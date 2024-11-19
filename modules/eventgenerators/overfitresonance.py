import torch
from torch import nn
from typing import Union

from config import Config
from modules import interpolate_last_axis, sparse_softmax, select_items, NeuralReverb
from modules.eventgenerators.generator import EventGenerator
from modules.eventgenerators.schedule import DiracScheduler
from modules.iterative import TensorTransform
from modules.multiheadtransform import ShapeSpec
from modules.reds import F0Resonance
from modules.transfer import fft_convolve, make_waves
from modules.upsample import upsample_with_holes
from util import device
import numpy as np
from torch.nn import functional as F


def mix(dry: torch.Tensor, wet: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    batch, n_events, time = dry.shape

    mix = torch.softmax(mix, dim=-1)
    mix = mix[:, :, None, :]
    stacked = torch.stack([dry, wet], dim=-1)
    x = stacked * mix
    x = torch.sum(x, dim=-1)
    return x





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
        selected = select_items(selections, items, selection_type='sparse_softmax')
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
        # items = torch.relu(items)
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

        if self.randomize_phases:
            spec = torch.fft.rfft(x, dim=-1)
            # randomize phases
            mags = torch.abs(spec)
            phases = torch.angle(spec)
            phases = torch.zeros_like(phases).uniform_(-np.pi, np.pi)
            imag = torch.cumsum(phases, dim=1)
            imag = (imag + np.pi) % (2 * np.pi) - np.pi
            spec = mags * torch.exp(1j * imag)
            x = torch.fft.irfft(spec, dim=-1)

        if self.windowed:
            x = x * torch.hamming_window(x.shape[-1], device=x.device)

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
        # items = items - items.min()
        # items = items / (items.max() + 1e-3)
        return self.base_resonance + (torch.sigmoid(items) * self.diff)

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
                    * (torch.linspace(1, 0, steps=n_samples)[None, :] ** torch.zeros(n_items, 1).uniform_(50, 150))

        super().__init__(n_items, n_samples, initialize=init)
        self.full_size = full_size
        self.padded_size = padded_size

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have all values between 0 and 1
        """
        # items = items - items.min()
        # items = items / (items.max() + 1e-3)
        return items ** 2

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



class OverfitResonanceModel(nn.Module, EventGenerator):

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
        self.n_envelopes = n_envelopes

        self.samplerate = samplerate
        self.n_events = n_events
        self.n_samples = n_samples

        self.instr_expressivity = instr_expressivity
        self.n_deformations = n_deformations

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
        self.n_verbs = n_verbs

        self.room_shape = (1, n_events, n_verbs)

        self.n_resonances = n_resonances

        self.r = WavetableLookup(
            n_resonances, n_samples, n_resonances=4096, samplerate=samplerate, learnable=False)

        # TODO: This is an issue because we have gradients for every sample, even though
        # they are largely redundant (think exponential vs.
        # self.r = SampleLookup(n_resonances, n_samples, flatten_kernel_size=2048, randomize_phases=False)
        # self.r = F0ResonanceLookup(n_resonances, n_samples)

        self.n = SampleLookup(
            n_noise_filters, noise_filter_samples, windowed=True, randomize_phases=False)

        self.verb = Lookup(n_verbs, n_samples, initialize=lambda x: verbs, fixed=True)

        self.e = Envelopes(
            n_envelopes,
            n_samples=128,
            full_size=8192,
            padded_size=self.n_samples)

        self.n_decays = n_decays

        self.d = Decays(n_decays, n_frames, n_samples)
        self.warp = Deformations(n_deformations, instr_expressivity, n_frames, n_samples)

        self.noise_warp = Deformations(noise_deformations, noise_expressivity, n_frames, n_samples)

        self.scheduler = DiracScheduler(
            self.n_events, start_size=n_frames, n_samples=self.n_samples, pre_sparse=True)

        # self.scheduler = HierarchicalDiracModel(
        #     self.n_events, self.n_samples)

        # self.scheduler = FFTShiftScheduler(self.n_events)

    @property
    def shape_spec(self) -> ShapeSpec:
        return dict(
            noise_resonance=(self.noise_expressivity, self.n_noise_filters),
            noise_deformations=(self.noise_deformations,),
            noise_mixes=(2,),
            envelopes=(self.n_envelopes,),
            resonances=(self.instr_expressivity, self.n_resonances),
            res_filter=(self.noise_expressivity, self.n_noise_filters),
            deformations=(self.n_deformations,),
            decays=(self.n_decays,),
            mixes=(2,),
            amplitudes=(1,),
            room_choice=(self.n_verbs,),
            room_mix=(2,)
        )


    # def random_sequence(self):
    #     # TODO: This should be moved to top-level model and should support specifying an arbitrary number of events
    #     return self.forward(
    #         noise_resonance=torch.zeros(*self.noise_resonance_shape, device=device).uniform_(-0.02, 0.02),
    #         noise_deformations=torch.zeros(*self.noise_deformation_shape, device=device).uniform_(),
    #         noise_mixes=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02),
    #         envelopes=torch.zeros(*self.envelope_shape, device=device).uniform_(-0.02, 0.02),
    #         resonances=torch.zeros(*self.resonance_shape, device=device).uniform_(-0.02, 0.02),
    #         res_filter=torch.zeros(*self.noise_resonance_shape, device=device).uniform_(-0.02, 0.02),
    #         deformations=torch.zeros(*self.deformation_shape, device=device).uniform_(-0.02, 0.02),
    #         decays=torch.zeros(*self.decay_shape, device=device).uniform_(-0.02, 0.02),
    #         mixes=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02),
    #         amplitudes=torch.zeros(*self.amplitude_shape, device=device).uniform_(0, 1),
    #         times=self.scheduler.random_params(),
    #         room_choice=torch.zeros(*self.room_shape, device=device).uniform_(-0.02, 0.02),
    #         room_mix=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02)
    #     )

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