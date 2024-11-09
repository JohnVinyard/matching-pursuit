from typing import Union, Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from config import Config
from conjure import LmdbCollection, serve_conjure, loggers
from data import AudioIterator
from modules import stft, sparsify, sparsify_vectors, iterative_loss, select_items, interpolate_last_axis, \
    sparse_softmax, NeuralReverb, LinearOutputStack, max_norm, flattened_multiband_spectrogram, UnitNorm, UNet
from modules.anticausal import AntiCausalAnalysis
from modules.infoloss import MultiWindowSpectralInfoLoss
from modules.iterative import TensorTransform
from modules.reds import F0Resonance
from modules.transfer import make_waves
from modules.upsample import upsample_with_holes
from util import device, encode_audio, make_initializer
from torch.nn import functional as F
from modules.fft import fft_convolve
from argparse import ArgumentParser

# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 16

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256
n_events = 32
context_dim = 128

n_frames = n_samples // transform_step_size

initializer = make_initializer(0.05)

def transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, n_frames, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x

# def loss_transform(x: torch.Tensor) -> torch.Tensor:
#     x = stft(x, transform_window_size, transform_step_size, pad=True)
#     return x


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def mix(dry: torch.Tensor, wet: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    batch, n_events, time = dry.shape

    mix = torch.softmax(mix, dim=-1)
    mix = mix[:, :, None, :]
    stacked = torch.stack([dry, wet], dim=-1)
    x = stacked * mix
    x = torch.sum(x, dim=-1)
    return x


def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # this is here to make the shift value interpretable
    shift = (1 - shift)

    n_samples = a.shape[-1]

    shift_samples = (shift * n_samples * 0.5)

    # a = F.pad(a, (0, n_samples * 2))

    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs

    shift = torch.exp(shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    # samples = torch.relu(samples)
    return samples


class MultiHeadTransform(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            hidden_channels: int,
            shapes: Dict[str, Tuple[int]],
            n_layers: int):

        super().__init__()

        self.latent_dim = latent_dim
        self.shapes = shapes
        self.n_layers = n_layers
        self.hidden_channels =hidden_channels
        self.shapes = shapes

        modules = {name: LinearOutputStack(
            channels=hidden_channels,
            layers=n_layers,
            in_channels=latent_dim,
            shortcut=False,
            out_channels=np.prod(shapes[name]),
            norm=nn.LayerNorm((hidden_channels,))
        ) for name, shape in shapes.items()}

        self.mods = nn.ModuleDict(modules)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, n_events, latent = x.shape

        return {
            name: module.forward(x).view(batch, n_events, *self.shapes[name])
            for name, module
            in self.mods.items()
        }


class Lookup(nn.Module):

    def __init__(
            self,
            n_items: int,
            n_samples: int,
            initialize: Union[None, TensorTransform] = None,
            fixed: bool = False):
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples
        data = torch.zeros(n_items, n_samples)
        self.fixed = fixed
        initialized = data.uniform_(-0.02, 0.02) if initialize is None else initialize(data)

        if self.fixed:
            self.register_buffer('items', initialized)
        else:
            self.items = nn.Parameter(initialized)

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        return items

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        return items

    def forward(self, selections: torch.Tensor) -> torch.Tensor:
        items = self.preprocess_items(self.items)
        selected = select_items(selections, items, type='sparse_softmax')
        processed = self.postprocess_results(selected)
        return processed


def flatten_envelope(x: torch.Tensor, kernel_size: int, step_size: int):
    """
    Given a signal with time-varying amplitude, give it as uniform an amplitude
    over time as possible
    """
    env = torch.abs(x)

    normalized = x / (env.max(dim=-1, keepdim=True)[0] + 1e-3)
    env = F.max_pool1d(
        env,
        kernel_size=kernel_size,
        stride=step_size,
        padding=step_size)
    env = 1 / env
    env = interpolate_last_axis(env, desired_size=x.shape[-1])
    result = normalized * env
    return result


class F0ResonanceLookup(Lookup):
    def __init__(self, n_items: int, n_samples: int):
        super().__init__(n_items, n_samples=3)
        self.f0 = F0Resonance(n_octaves=16, n_samples=n_samples)
        self.audio_samples = n_samples

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        batch, n_events, expressivity, _ = items.shape

        items = items.view(batch * n_events, expressivity, 3)

        f0 = items[..., :1]
        spacing = items[..., 1:2]
        decays = items[..., 2:]
        res = self.f0.forward(
            f0=f0, decay_coefficients=decays, freq_spacing=spacing, sigmoid_decay=True, apply_exponential_decay=True)
        res = res.view(batch, n_events, expressivity, self.audio_samples)
        return res


class WavetableLookup(Lookup):
    def __init__(
            self,
            n_items: int,
            n_samples: int,
            n_resonances: int,
            samplerate: int,
            learnable: bool = False):

        super().__init__(n_items, n_resonances)
        w = make_waves(n_samples, np.linspace(20, 4000, num=n_resonances // 4), samplerate)
        if learnable:
            self.waves = nn.Parameter(w)
        else:
            self.register_buffer('waves', w)

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        return items

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        # items is of dimension (batch, n_events, n_resonances)
        items = torch.relu(items)
        x = items @ self.waves
        return x


class SampleLookup(Lookup):

    def __init__(
            self,
            n_items: int,
            n_samples: int,
            flatten_kernel_size: Union[int, None] = None,
            initial: Union[torch.Tensor, None] = None,
            randomize_phases: bool = True,
            windowed: bool = False):


        if initial is not None:
            initializer = lambda x: initial
        else:
            initializer = None

        super().__init__(n_items, n_samples, initialize=initializer)
        self.randomize_phases = randomize_phases
        self.flatten_kernel_size = flatten_kernel_size
        self.windowed = windowed

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have audio-rate samples at a relatively uniform
        amplitude throughout
        """
        if self.flatten_kernel_size:
            x = flatten_envelope(
                items,
                kernel_size=self.flatten_kernel_size,
                step_size=self.flatten_kernel_size // 2)
        else:
            x = items

        spec = torch.fft.rfft(x, dim=-1)

        if self.randomize_phases:
            # randomize phases
            mags = torch.abs(spec)
            phases = torch.angle(spec)
            phases = torch.zeros_like(phases).uniform_(-np.pi, np.pi)
            imag = torch.cumsum(phases, dim=1)
            imag = (imag + np.pi) % (2 * np.pi) - np.pi
            spec = mags * torch.exp(1j * imag)

        x = torch.fft.irfft(spec, dim=-1)

        if self.windowed:
            x *= torch.hamming_window(x.shape[-1], device=x.device)

        return x


class Decays(Lookup):
    def __init__(self, n_items: int, n_samples: int, full_size: int, base_resonance: float = 0.5):
        super().__init__(n_items, n_samples)
        self.full_size = full_size
        self.base_resonance = base_resonance
        self.diff = 1 - self.base_resonance

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have all values between 0 and 1
        """
        items = items - items.min()
        items = items / (items.max() + 1e-3)
        return self.base_resonance + (items * self.diff)

    def postprocess_results(self, decay: torch.Tensor) -> torch.Tensor:
        """Apply a scan in log-space to end up with exponential decay
        """

        decay = torch.log(decay + 1e-12)
        decay = torch.cumsum(decay, dim=-1)
        decay = torch.exp(decay)
        amp = interpolate_last_axis(decay, desired_size=self.full_size)
        return amp


class Envelopes(Lookup):
    def __init__(
            self,
            n_items: int,
            n_samples: int,
            full_size: int,
            padded_size: int):
        def init(x):
            return \
                    torch.zeros(n_items, n_samples).uniform_(-1, 1) \
                    * (torch.linspace(1, 0, steps=n_samples)[None, :] ** torch.zeros(n_items, 1).uniform_(50, 100))

        super().__init__(n_items, n_samples, initialize=init)
        self.full_size = full_size
        self.padded_size = padded_size

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have all values between 0 and 1
        """
        items = items - items.min()
        items = items / (items.max() + 1e-3)
        return items

    def postprocess_results(self, decay: torch.Tensor) -> torch.Tensor:
        """Scale up to sample rate and multiply with noise
        """

        amp = interpolate_last_axis(decay, desired_size=self.full_size)
        amp = amp * torch.zeros_like(amp).uniform_(-0.02, 0.02)
        diff = self.padded_size - self.full_size
        padding = torch.zeros((amp.shape[:-1] + (diff,)), device=amp.device)
        amp = torch.cat([amp, padding], dim=-1)
        return amp


class Deformations(Lookup):

    def __init__(self, n_items: int, channels: int, frames: int, full_size: int):
        super().__init__(n_items, channels * frames)
        self.full_size = full_size
        self.channels = channels
        self.frames = frames

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        """Reshape so that the values are (..., channels, frames).  Apply
        softmax to the channel dimension and then upscale to full samplerate
        """
        shape = items.shape[:-1]
        x = items.reshape(*shape, self.channels, self.frames)
        x = torch.softmax(x, dim=-2)
        x = interpolate_last_axis(x, desired_size=self.full_size)
        return x


class DiracScheduler(nn.Module):
    def __init__(self, n_events: int, start_size: int, n_samples: int, pre_sparse: bool = False):
        super().__init__()
        self.n_events = n_events
        self.start_size = start_size
        self.n_samples = n_samples
        self.pos = nn.Parameter(
            torch.zeros(1, n_events, start_size).uniform_(-0.02, 0.02)
        )
        self.pre_sparse = pre_sparse

    def random_params(self):
        pos = torch.zeros(1, self.n_events, self.start_size, device=device).uniform_(-0.02, 0.02)
        if self.pre_sparse:
            pos = sparse_softmax(pos, normalize=True, dim=-1)
        return pos

    @property
    def params(self):
        return self.pos

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        if not self.pre_sparse:
            pos = sparse_softmax(pos, normalize=True, dim=-1)
        pos = upsample_with_holes(pos, desired_size=self.n_samples)
        final = fft_convolve(events, pos)
        return final

    # def forward(self, events: torch.Tensor) -> torch.Tensor:
    #     return self.schedule(self.pos, events)


# class FFTShiftScheduler(nn.Module):
#     def __init__(self, n_events):
#         super().__init__()
#         self.n_events = n_events
#         self.pos = nn.Parameter(torch.zeros(1, n_events, 1).uniform_(0, 1))
#
#     def random_params(self):
#         return torch.zeros(1, self.n_events, 1, device=device).uniform_(0, 1)
#
#     @property
#     def params(self):
#         return self.pos
#
#     def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
#         final = fft_shift(events, pos)
#         return final
#
#     def forward(self, events: torch.Tensor) -> torch.Tensor:
#         return self.schedule(self.pos, events)


# class HierarchicalDiracModel(nn.Module):
#     def __init__(self, n_events: int, signal_size: int):
#         super().__init__()
#         self.n_events = n_events
#         self.signal_size = signal_size
#         n_elements = int(np.log2(signal_size))
#
#         self.elements = nn.Parameter(
#             torch.zeros(1, n_events, n_elements, 2).uniform_(-0.02, 0.02))
#
#         self.n_elements = n_elements
#
#     def random_params(self):
#         return torch.zeros(1, self.n_events, self.n_elements, 2, device=device).uniform_(-0.02, 0.02)
#
#     @property
#     def params(self):
#         return self.elements
#
#     def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
#         x = hierarchical_dirac(pos)
#         x = fft_convolve(x, events)
#         return x
#
#     def forward(self, events: torch.Tensor) -> torch.Tensor:
#         return self.schedule(self.elements, events)


class OverfitResonanceModel(nn.Module):
    """
    A model that compresses an audio segment into n_events with the following:

        - (1) one-hot choice of envelope
        - (2) one-hot choice of noise resonance
        - (3) one-hot choice of noise deformation
        - (4) one-hot choice of noise mix
        - (5) one-hot choice of resonance
        - (6) one-hot choice of decay
        - (7) one hot choice of deformation
        - (8) one-hot choice of resonance mix
        - (9) scalar amplitude (also could be quantized over a log-scale)
        - (10) log2(n_samples) event time (for this experiment, around 2 bytes)

    Assuming each of these has < 256 choices, then we'd have
    """

    def __init__(
            self,
            n_noise_filters: int,
            noise_expressivity: int,
            noise_filter_samples: int,
            noise_deformations: int,
            instr_expressivity: int,
            n_events: int,
            n_resonances: int,
            n_envelopes: int,
            n_decays: int,
            n_deformations: int,
            n_samples: int,
            n_frames: int,
            samplerate: int,
            hidden_channels: int):
        super().__init__()

        self.noise_filter_samples = noise_filter_samples
        self.noise_expressivity = noise_expressivity
        self.n_noise_filters = n_noise_filters
        self.noise_deformations = noise_deformations

        self.samplerate = samplerate
        self.n_events = n_events
        self.n_samples = n_samples

        self.resonance_shape = (1, n_events, instr_expressivity, n_resonances)
        self.noise_resonance_shape = (1, n_events, noise_expressivity, n_noise_filters)

        self.envelope_shape = (1, n_events, n_envelopes)
        self.decay_shape = (1, n_events, n_decays)

        self.deformation_shape = (1, n_events, n_deformations)
        self.noise_deformation_shape = (1, n_events, noise_deformations)

        self.mix_shape = (1, n_events, 2)
        self.amplitude_shape = (1, n_events, 1)

        verbs = NeuralReverb.tensors_from_directory(Config.impulse_response_path(), n_samples)
        n_verbs = verbs.shape[0]

        self.room_shape = (1, n_events, n_verbs)

        self.multihead = MultiHeadTransform(
            latent_dim=context_dim,
            hidden_channels=hidden_channels,
            n_layers=2,
            shapes=dict(
                noise_resonance=(noise_expressivity, n_noise_filters),
                noise_deformations=(noise_deformations,),
                noise_mixes=(2,),
                envelopes=(n_envelopes,),
                resonances=(instr_expressivity, n_resonances),
                res_filter=(noise_expressivity, n_noise_filters),
                deformations=(n_deformations,),
                decays=(n_decays,),
                mixes=(2,),
                amplitudes=(1,),
                room_choice=(n_verbs,),
                room_mix=(2,)
            )
        )


        self.r = WavetableLookup(
            n_resonances, n_samples, n_resonances=4096, samplerate=samplerate, learnable=False)
        # self.r = SampleLookup(n_resonances, n_samples, flatten_kernel_size=512)
        # self.r = F0ResonanceLookup(n_resonances, n_samples)

        self.n = SampleLookup(
            n_noise_filters, noise_filter_samples, windowed=True, randomize_phases=False)

        self.verb = Lookup(n_verbs, n_samples, initialize=lambda x: verbs, fixed=True)

        self.e = Envelopes(
            n_envelopes,
            n_samples=128,
            full_size=8192,
            padded_size=self.n_samples)

        self.d = Decays(n_decays, n_frames, n_samples)
        self.warp = Deformations(n_deformations, instr_expressivity, n_frames, n_samples)

        self.noise_warp = Deformations(noise_deformations, noise_expressivity, n_frames, n_samples)

        self.scheduler = DiracScheduler(
            self.n_events, start_size=n_frames, n_samples=self.n_samples, pre_sparse=True)

        # self.scheduler = HierarchicalDiracModel(
        #     self.n_events, self.n_samples)

        # self.scheduler = FFTShiftScheduler(self.n_events)

    def random_sequence(self):
        return self.forward(
            noise_resonance=torch.zeros(*self.noise_resonance_shape, device=device).uniform_(-0.02, 0.02),
            noise_deformations=torch.zeros(*self.noise_deformation_shape, device=device).uniform_(),
            noise_mixes=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02),
            envelopes=torch.zeros(*self.envelope_shape, device=device).uniform_(-0.02, 0.02),
            resonances=torch.zeros(*self.resonance_shape, device=device).uniform_(-0.02, 0.02),
            res_filter=torch.zeros(*self.noise_resonance_shape, device=device).uniform_(-0.02, 0.02),
            deformations=torch.zeros(*self.deformation_shape, device=device).uniform_(-0.02, 0.02),
            decays=torch.zeros(*self.decay_shape, device=device).uniform_(-0.02, 0.02),
            mixes=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02),
            amplitudes=torch.zeros(*self.amplitude_shape, device=device).uniform_(0, 1),
            times=self.scheduler.random_params(),
            room_choice=torch.zeros(*self.room_shape, device=device).uniform_(-0.02, 0.02),
            room_mix=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02)
        )

    def forward(
            self,
            noise_resonance: torch.Tensor,
            noise_deformations: torch.Tensor,
            noise_mixes: torch.Tensor,
            envelopes: torch.Tensor,
            resonances: torch.Tensor,
            res_filter: torch.Tensor,
            deformations: torch.Tensor,
            decays: torch.Tensor,
            mixes: torch.Tensor,
            amplitudes: torch.Tensor,
            times: torch.Tensor,
            room_choice: torch.Tensor,
            room_mix: torch.Tensor) -> torch.Tensor:

        # Begin layer ==========================================

        # calculate impulses or energy injected into a system
        impulses = self.e.forward(envelopes)

        # choose filters to be convolved with energy/noise
        noise_res = self.n.forward(noise_resonance)
        noise_res = torch.cat([
            noise_res,
            torch.zeros(*noise_res.shape[:-1], self.n_samples - noise_res.shape[-1], device=impulses.device)
        ], dim=-1)

        # choose deformations to be applied to the initial noise resonance
        noise_def = self.noise_warp.forward(noise_deformations)

        # choose a dry/wet mix
        noise_mix = noise_mixes[:, :, None, :]
        noise_mix = torch.softmax(noise_mix, dim=-1)

        # convolve the initial impulse with all filters, then mix together
        noise_wet = fft_convolve(impulses[:, :, None, :], noise_res)
        noise_wet = noise_wet * noise_def
        noise_wet = torch.sum(noise_wet, dim=2)

        # mix dry and wet
        # stacked = torch.stack([impulses, noise_wet], dim=-1)
        # mixed = stacked * noise_mix
        # mixed = torch.sum(mixed, dim=-1)
        mixed = noise_wet

        # initial filtered noise is now the input to our longer resonances
        impulses = mixed

        # choose a number of resonances to be convolved with
        # those impulses
        resonance = self.r.forward(resonances)
        res_filters = self.n.forward(res_filter)
        res_filters = torch.cat([
            res_filters,
            torch.zeros(*res_filters.shape[:-1], resonance.shape[-1] - res_filters.shape[-1], device=res_filters.device)
        ], dim=-1)
        resonance = fft_convolve(resonance, res_filters)

        # describe how we interpolate between different
        # resonances over time
        deformations = self.warp.forward(deformations)

        # determine how each resonance decays or leaks energy
        decays = self.d.forward(decays)
        decaying_resonance = resonance * decays[:, :, None, :]

        dry = impulses[:, :, None, :]

        # convolve the impulse with all the resonances and
        # interpolate between them
        conv = fft_convolve(dry, decaying_resonance)
        with_deformations = conv * deformations
        audio_events = torch.sum(with_deformations, dim=2, keepdim=True)

        # mix the dry and wet signals
        mixes = mixes[:, :, None, None, :]
        mixes = torch.softmax(mixes, dim=-1)

        stacked = torch.stack([dry, audio_events], dim=-1)
        mixed = stacked * mixes
        final = torch.sum(mixed, dim=-1)

        # apply reverb
        verb = self.verb.forward(room_choice)
        wet = fft_convolve(verb, final.view(*verb.shape))
        verb_mix = torch.softmax(room_mix, dim=-1)[:, :, None, :]
        stacked = torch.stack([wet, final.view(*verb.shape)], dim=-1)
        stacked = stacked * verb_mix

        final = stacked.sum(dim=-1)

        # apply amplitudes
        final = final.view(-1, self.n_events, self.n_samples)
        final = final * torch.abs(amplitudes)

        # End layer ==========================================

        scheduled = self.scheduler.schedule(times, final)

        return scheduled

    # def forward(self):
    #     return self.apply_forces(
    #         noise_resonance=self.noise_resonances,
    #         noise_deformations=self.noise_deformations,
    #         noise_mixes=self.noise_mixes,
    #         envelopes=self.envelopes,
    #         resonances=self.resonances,
    #         deformations=self.deformations,
    #         decays=self.decays,
    #         mixes=self.mixes,
    #         amplitudes=self.amplitudes,
    #         times=self.scheduler.params,
    #         room_mix=self.room_mix,
    #         room_choice=self.rooms)



class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 1024, hidden_channels: int = 256):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.encoder = AntiCausalAnalysis(
            in_channels=in_channels,
            channels=hidden_channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 1],
            pos_encodings=False,
            do_norm=True)
        # self.encoder = UNet(in_channels, is_disc=True)
        self.judge = nn.Linear(hidden_channels, 1)
        self.apply(initializer)

    def forward(self, transformed: torch.Tensor):
        batch_size = transformed.shape[0]

        if transformed.shape[1] == 1:
            transformed = transform(transformed)

        x = transformed

        encoded = self.encoder.forward(x)
        encoded = torch.sum(encoded, dim=-1)
        x = self.judge(encoded)

        return x


class Model(nn.Module):
    def __init__(
            self,
            in_channels: int = 1024,
            hidden_channels: int = 256):

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.encoder = AntiCausalAnalysis(
            in_channels=in_channels,
            channels=hidden_channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 1],
            pos_encodings=False,
            do_norm=True)
        self.to_event_vectors = nn.Conv1d(hidden_channels, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(hidden_channels, 1, 1, 1, 0)

        self.resonance = OverfitResonanceModel(
            n_noise_filters=32,
            noise_expressivity=8,
            noise_filter_samples=128,
            noise_deformations=16,
            instr_expressivity=8,
            n_events=1,
            n_resonances=512,
            n_envelopes=128,
            n_decays=32,
            n_deformations=32,
            n_samples=n_samples,
            n_frames=n_frames,
            samplerate=samplerate,
            hidden_channels=hidden_channels
        )

        self.apply(initializer)

    def encode(self, transformed: torch.Tensor):
        n_events = 1

        batch_size = transformed.shape[0]

        if transformed.shape[1] == 1:
            transformed = transform(transformed)

        x = transformed

        encoded = self.encoder.forward(x)

        event_vecs = self.to_event_vectors(encoded).permute(0, 2, 1)  # batch, time, channels

        event_switch = self.to_event_switch(encoded)
        attn = torch.relu(event_switch).permute(0, 2, 1).view(batch_size, 1, -1)

        attn, attn_indices, values = sparsify(attn, n_to_keep=n_events, return_indices=True)

        vecs, indices = sparsify_vectors(event_vecs.permute(0, 2, 1), attn, n_to_keep=n_events)

        scheduling = torch.zeros(batch_size, n_events, encoded.shape[-1], device=encoded.device)
        for b in range(batch_size):
            for j in range(n_events):
                index = indices[b, j]
                scheduling[b, j, index] = attn[b, 0][index]

        return vecs, scheduling

    def generate(self, vecs: torch.Tensor, scheduling: torch.Tensor):
        choices = self.resonance.multihead.forward(vecs)
        choices_with_scheduling = dict(**choices, times=scheduling)
        events = self.resonance.forward(**choices_with_scheduling)
        return events

    def iterative(self, audio: torch.Tensor):
        channels = []
        schedules = []
        vecs = []

        spec = transform(audio)

        for i in range(n_events):
            v, sched = self.encode(spec)
            vecs.append(v)
            schedules.append(sched)
            ch = self.generate(v, sched)
            current = transform(ch)
            spec = (spec - current).clone().detach()
            channels.append(ch)

        channels = torch.cat(channels, dim=1)
        vecs = torch.cat(vecs, dim=1)
        schedules = torch.cat(schedules, dim=1)

        return channels, vecs, schedules

    def forward(self, audio: torch.Tensor):
        raise NotImplementedError()



def train_and_monitor(
        batch_size: int = 8,
        overfit: bool = False,
        sparsity_loss: bool = False,
        adv_loss: bool = False):

    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=overfit)


    collection = LmdbCollection(path='iterativedecomposition')

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        orig_audio,
        recon_audio,
        random_audio
    ], port=9999, n_workers=1)

    def train():
        model = Model(in_channels=1024, hidden_channels=512).to(device)
        optim = Adam(model.parameters(), lr=1e-3)

        disc = Discriminator(in_channels=1024, hidden_channels=512).to(device)
        disc_optim = Adam(model.parameters(), lr=1e-3)


        for i, item in enumerate(iter(stream)):
            optim.zero_grad()
            disc_optim.zero_grad()

            target = item.view(batch_size, 1, n_samples).to(device)
            orig_audio(target)
            recon, encoded, scheduling = model.iterative(target)
            recon_summed = torch.sum(recon, dim=1, keepdim=True)
            recon_audio(max_norm(recon_summed))


            loss = iterative_loss(target, recon, loss_transform)

            if sparsity_loss:
                loss = loss + (torch.abs(encoded).sum() * 1e-3)

            if adv_loss:
                mask = torch.zeros(target.shape[0], n_events, 1, device=recon.device).bernoulli_(p=0.5)
                for_disc = torch.sum(recon * mask, dim=1, keepdim=True)
                j = disc.forward(for_disc)
                d_loss = torch.abs(1 - j).mean()
                print('G', d_loss.item())
                loss = loss + d_loss


            loss.backward()
            optim.step()
            print(i, loss.item())

            if adv_loss:
                disc_optim.zero_grad()
                r_j = disc.forward(target)
                f_j = disc.forward(recon_summed.clone().detach())
                d_loss = torch.abs(0 - f_j).mean() + torch.abs(1 - r_j).mean()
                d_loss.backward()
                print('D', d_loss.item())
                disc_optim.step()

            with torch.no_grad():
                rnd = model.resonance.random_sequence()
                rnd = torch.sum(rnd, dim=1, keepdim=True)
                rnd = max_norm(rnd)
                random_audio(rnd)

    train()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--overfit',
        required=False,
        action='store_true',
        default=False,
    )
    parser.add_argument(
        '--loss-type',
        type=str,
        required=False,
        choices=['stft', 'multresolution'],
        default='multiresolution'
    )
    parser.add_argument(
        '--sparsity-loss',
        action='store_true',
        required=False,
        default=False
    )
    parser.add_argument(
        '--synthetic-loss',
        action='store_true',
        required=False,
        default=False
    )
    parser.add_argument(
        '--adversarial-loss',
        action='store_true',
        required=False,
        default=False
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        required=False
    )
    args = parser.parse_args()
    train_and_monitor(
        batch_size=1 if args.overfit else args.batch_size,
        overfit=args.overfit,
        sparsity_loss=args.sparsity_loss,
        adv_loss=args.adversarial_loss)

