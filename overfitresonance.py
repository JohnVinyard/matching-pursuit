from typing import Union, Dict
import torch
from torch import nn

from config import Config
from data.audioiter import AudioIterator
from modules import stft, fft_frequency_decompose, NeuralReverb
from modules.fft import fft_convolve
from conjure import LmdbCollection, audio_conjure, serve_conjure, numpy_conjure, SupportedContentType
from io import BytesIO
from soundfile import SoundFile
from modules.iterative import TensorTransform, iterative_loss
from modules.quantize import select_items
from modules.reds import F0Resonance
from modules.softmax import sparse_softmax
from modules.transfer import hierarchical_dirac, make_waves, ExponentialTransform
from modules.upsample import interpolate_last_axis, upsample_with_holes
from torch.nn import functional as F
from torch.optim import Adam
from itertools import count
import numpy as np
from modules.normalization import max_norm
from util import device

collection = LmdbCollection(path='overfitresonance')

samplerate = 22050
n_samples = 2 ** 16
n_frames = 256
n_events = 32

"""
TODOs:

- exponential transform as part of loss
- spiking pif as part of loss
- exponential decay for sparser energy
- damped oscillator scan
- full, deeper architecture with impulse and room convolutions
"""


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
            windowed: bool = False):


        if initial is not None:
            initializer = lambda x: initial
        else:
            initializer = None

        super().__init__(n_items, n_samples, initialize=initializer)
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

        mags = torch.abs(spec)

        # randomize phases
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
    def __init__(self, n_events: int, start_size: int, n_samples: int):
        super().__init__()
        self.n_events = n_events
        self.start_size = start_size
        self.n_samples = n_samples
        self.pos = nn.Parameter(
            torch.zeros(1, n_events, start_size).uniform_(-0.02, 0.02)
        )

    def random_params(self):
        return torch.zeros(1, self.n_events, self.start_size, device=device).uniform_(-0.02, 0.02)

    @property
    def params(self):
        return self.pos

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        pos = sparse_softmax(pos, normalize=True, dim=-1)
        pos = upsample_with_holes(pos, desired_size=self.n_samples)
        final = fft_convolve(events, pos)
        return final

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.schedule(self.pos, events)


class FFTShiftScheduler(nn.Module):
    def __init__(self, n_events):
        super().__init__()
        self.n_events = n_events
        self.pos = nn.Parameter(torch.zeros(1, n_events, 1).uniform_(0, 1))

    def random_params(self):
        return torch.zeros(1, self.n_events, 1, device=device).uniform_(0, 1)

    @property
    def params(self):
        return self.pos

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        final = fft_shift(events, pos)
        return final

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.schedule(self.pos, events)


class HierarchicalDiracModel(nn.Module):
    def __init__(self, n_events: int, signal_size: int):
        super().__init__()
        self.n_events = n_events
        self.signal_size = signal_size
        n_elements = int(np.log2(signal_size))

        self.elements = nn.Parameter(
            torch.zeros(1, n_events, n_elements, 2).uniform_(-0.02, 0.02))

        self.n_elements = n_elements

    def random_params(self):
        return torch.zeros(1, self.n_events, self.n_elements, 2, device=device).uniform_(-0.02, 0.02)

    @property
    def params(self):
        return self.elements

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        x = hierarchical_dirac(pos)
        x = fft_convolve(x, events)
        return x

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        return self.schedule(self.elements, events)


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
            samplerate: int):
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
        self.noise_deformation_shape = (1 ,n_events, noise_deformations)

        self.mix_shape = (1, n_events, 2)
        self.amplitude_shape = (1, n_events, 1)

        verbs = NeuralReverb.tensors_from_directory(Config.impulse_response_path(), n_samples)
        n_verbs = verbs.shape[0]

        self.room_shape = (1, n_events, n_verbs)


        # noise choices/selections
        self.noise_resonances = nn.Parameter(
            torch.zeros(*self.noise_resonance_shape).uniform_())
        self.noise_deformations = nn.Parameter(
            torch.zeros(*self.noise_deformation_shape).uniform_(-0.02, 0.02))
        self.noise_mixes = nn.Parameter(
            torch.zeros(*self.mix_shape).uniform_(-0.02, 0.02))

        # choices/selections
        self.resonances = nn.Parameter(
            torch.zeros(*self.resonance_shape).uniform_(-0.02, 0.02))

        self.envelopes = nn.Parameter(
            torch.zeros(*self.envelope_shape).uniform_(-0.02, 0.02))

        self.decays = nn.Parameter(
            torch.zeros(*self.decay_shape).uniform_(-0.02, 0.02))

        self.deformations = nn.Parameter(
            torch.zeros(*self.deformation_shape).uniform_(-0.02, 0.02))

        self.mixes = nn.Parameter(
            torch.zeros(*self.mix_shape).uniform_(-0.02, 0.02))

        self.amplitudes = nn.Parameter(
            torch.zeros(*self.amplitude_shape).uniform_(0, 0.02))

        self.res_filter = nn.Parameter(
            torch.zeros(1, n_events, instr_expressivity, n_noise_filters).uniform_(0.02, 0.02)
        )

        # room choices and mix
        self.rooms = nn.Parameter(torch.zeros(*self.room_shape).uniform_(-0.02, 0.02))
        self.room_mix = nn.Parameter(torch.zeros(*self.mix_shape).uniform_(-0.02, 0.02))

        self.r = WavetableLookup(
            n_resonances, n_samples, n_resonances=4096, samplerate=samplerate, learnable=False)

        # self.r = SampleLookup(n_resonances, n_samples, flatten_kernel_size=512)
        # self.r = F0ResonanceLookup(n_resonances, n_samples)
        self.n = SampleLookup(n_noise_filters, noise_filter_samples, windowed=True)


        self.verb = Lookup(n_verbs, n_samples, initialize=lambda x: verbs, fixed=True)

        self.e = Envelopes(
            n_envelopes,
            n_samples=128,
            full_size=8192,
            padded_size=self.n_samples)

        self.d = Decays(n_decays, n_frames, n_samples)
        self.warp = Deformations(n_deformations, instr_expressivity, n_frames, n_samples)

        self.noise_warp = Deformations(noise_deformations, noise_expressivity, n_frames, n_samples)

        # self.scheduler = DiracScheduler(
        #     self.n_events, start_size=self.n_samples // 32, n_samples=self.n_samples)

        self.scheduler = HierarchicalDiracModel(
            self.n_events, self.n_samples)

        # self.scheduler = FFTShiftScheduler(self.n_events)

    def random_sequence(self):
        return self.apply_forces(
            noise_resonance=torch.zeros(*self.noise_resonance_shape, device=device).uniform_(-0.02, 0.02),
            noise_deformations=torch.zeros(*self.noise_deformation_shape, device=device).uniform_(),
            noise_mixes=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02),
            envelopes=torch.zeros(*self.envelope_shape, device=device).uniform_(-0.02, 0.02),
            resonances=torch.zeros(*self.resonance_shape, device=device).uniform_(-0.02, 0.02),
            deformations=torch.zeros(*self.deformation_shape, device=device).uniform_(-0.02, 0.02),
            decays=torch.zeros(*self.decay_shape, device=device).uniform_(-0.02, 0.02),
            mixes=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02),
            amplitudes=torch.zeros(*self.amplitude_shape, device=device).uniform_(0, 1),
            times=self.scheduler.random_params(),
            room_choice=torch.zeros(*self.room_shape, device=device).uniform_(-0.02, 0.02),
            room_mix=torch.zeros(*self.mix_shape, device=device).uniform_(-0.02, 0.02))

    def apply_forces(
            self,
            noise_resonance: torch.Tensor,
            noise_deformations: torch.Tensor,
            noise_mixes: torch.Tensor,
            envelopes: torch.Tensor,
            resonances: torch.Tensor,
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
        stacked = torch.stack([impulses, noise_wet], dim=-1)
        mixed = stacked * noise_mix
        mixed = torch.sum(mixed, dim=-1)

        # initial filtered noise is now the input to our longer resonances
        impulses = mixed

        # choose a number of resonances to be convolved with
        # those impulses
        resonance = self.r.forward(resonances)
        res_filters = self.n.forward(self.res_filter)
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

    def forward(self):
        return self.apply_forces(
            noise_resonance=self.noise_resonances,
            noise_deformations=self.noise_deformations,
            noise_mixes=self.noise_mixes,
            envelopes=self.envelopes,
            resonances=self.resonances,
            deformations=self.deformations,
            decays=self.decays,
            mixes=self.mixes,
            amplitudes=self.amplitudes,
            times=self.scheduler.params,
            room_mix=self.room_mix,
            room_choice=self.rooms)




def audio(x: torch.Tensor):
    x = x.data.cpu().numpy()[0].reshape((-1,))
    io = BytesIO()

    with SoundFile(
            file=io,
            mode='w',
            samplerate=samplerate,
            channels=1,
            format='WAV',
            subtype='PCM_16') as sf:
        sf.write(x)

    io.seek(0)
    return io.read()


@audio_conjure(storage=collection)
def recon_audio(x: torch.Tensor):
    return audio(x)


@audio_conjure(storage=collection)
def orig_audio(x: torch.Tensor):
    return audio(x)


@audio_conjure(storage=collection)
def random_audio(x: torch.Tensor):
    return audio(x)

@numpy_conjure(storage=collection, content_type=SupportedContentType.Spectrogram.value)
def envelopes(x: torch.Tensor):
    return x.data.cpu().numpy()

# TODO: consider multi-band transform or PIF here.
def transform(audio: torch.Tensor) -> torch.Tensor:
    return stft(audio, ws=2048, step=256, pad=True)



def spec_loss(recon_audio: torch.Tensor, real_audio: torch.Tensor) -> torch.Tensor:
    recon_spec = transform(torch.sum(recon_audio, dim=1, keepdim=True))
    real_spec = transform(real_audio)
    loss = torch.abs(recon_spec - real_spec).sum()
    return loss

# exp_transform = ExponentialTransform(32, 16, n_exponents=16, n_frames=n_samples // 16).to(device)

# def transform(x: torch.Tensor) -> torch.Tensor:
#     batch_size, channels, _ = x.shape
#     bands = multiband_transform(x)
#     return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)
#
#
# def multiband_transform(x: torch.Tensor) -> Dict[str, torch.Tensor]:
#     bands = fft_frequency_decompose(x, 512)
#     # TODO: each band should have 256 frequency bins and also 256 time bins
#     # this requires a window size of (n_samples // 256) * 2
#     # and a window size of 512, 256
#
#     window_size = 512
#
#     d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
#     d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
#     d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
#
#     normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)
#
#     # et = exp_transform.forward(x)
#
#     return dict(
#         normal=normal,
#         # et=et * 1e-3,
#         **d1,
#         **d3,
#         **d4
#     )


def train(target: torch.Tensor):
    model = OverfitResonanceModel(
        noise_filter_samples=64,
        noise_deformations=16,
        noise_expressivity=2,
        n_noise_filters=16,
        instr_expressivity=4,
        n_events=n_events,
        n_resonances=16,
        n_envelopes=16,
        n_decays=16,
        n_deformations=16,
        n_samples=n_samples,
        n_frames=n_frames,
        samplerate=samplerate
    ).to(device)

    optim = Adam(model.parameters(), lr=1e-3)

    for iteration in count():
        optim.zero_grad()
        recon = model.forward()

        # logging
        recon_audio(max_norm(torch.sum(recon, dim=1, keepdim=True)))

        loss = iterative_loss(target, recon, transform)
        # loss = spec_loss(recon, target)

        envelopes(model.e.items)

        loss.backward()
        optim.step()
        print(iteration, loss.item())

        with torch.no_grad():
            rnd = model.random_sequence()
            # logging
            random_audio(max_norm(torch.sum(rnd, dim=1, keepdim=True)))


if __name__ == '__main__':
    ai = AudioIterator(
        batch_size=1,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=True, )
    target: torch.Tensor = next(iter(ai)).to(device).view(-1, 1, n_samples)

    # logging
    orig_audio(target)

    serve_conjure(
        conjure_funcs=[
            recon_audio,
            orig_audio,
            random_audio,
            envelopes
        ],
        port=9999,
        n_workers=1
    )
    train(target)
