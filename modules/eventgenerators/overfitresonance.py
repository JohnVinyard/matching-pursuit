from typing import Union, Tuple, Dict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import Config
from modules import interpolate_last_axis, select_items, NeuralReverb, SSM, sparse_softmax, sparsify, max_norm, \
    fft_frequency_recompose, unit_norm, LinearOutputStack, gammatone_filter_bank
from modules.ddsp import AudioModel
from modules.eventgenerators.generator import EventGenerator
from modules.eventgenerators.schedule import DiracScheduler, HierarchicalDiracModel
from modules.iterative import TensorTransform
from modules.multiheadtransform import ShapeSpec
from modules.overlap_add import overlap_add
from modules.phase import mag_phase_recomposition
from modules.transfer import fft_convolve, make_waves_vectorized, freq_domain_transfer_function_to_resonance, fft_shift
from modules.upsample import ensure_last_axis_length, ConvUpsample
from util import device


def mix(dry: torch.Tensor, wet: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    batch, n_events, time = dry.shape

    mix = torch.softmax(mix, dim=-1)
    mix = mix[:, :, None, :]
    stacked = torch.stack([dry, wet], dim=-1)
    x = stacked * mix
    x = torch.sum(x, dim=-1)
    return x


# def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
#     # this is here to make the shift value interpretable
#     shift = (1 - shift)
#
#     n_samples = a.shape[-1]
#
#     shift_samples = (shift * n_samples * 0.5)
#
#     # a = F.pad(a, (0, n_samples * 2))
#
#     spec = torch.fft.rfft(a, dim=-1, norm='ortho')
#
#     n_coeffs = spec.shape[-1]
#     shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
#
#     shift = torch.exp(shift * shift_samples)
#
#     spec = spec * shift
#
#     samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
#     # samples = samples[..., :n_samples]
#     # samples = torch.relu(samples)
#     return samples


class Lookup(nn.Module):

    def __init__(
            self,
            n_items: int,
            n_samples: int,
            initialize: Union[None, TensorTransform] = None,
            fixed: bool = False,
            selection_type: str = 'softmax'):

        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples
        data = torch.zeros(n_items, n_samples)
        self.fixed = fixed
        self.selection_type = selection_type
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
        selected = select_items(selections, items, selection_type=self.selection_type)
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


def project_and_limit_norm(
        vector: torch.Tensor,
        forward: TensorTransform,
        max_efficiency: float = 0.999) -> torch.Tensor:
    # get the original norm, this is the absolute max norm/energy we should arrive at,
    # given a perfectly efficient physical system
    # original_norm = torch.norm(vector, dim=-1, keepdim=True)

    # project
    x = forward(vector)
    return x

    # TODO: clamp norm should be a utility that lives in normalization
    # find the norm of the projection
    new_norm = torch.norm(x, dim=-1, keepdim=True)

    # clamp the norm between the allowed values
    mx_value = original_norm.reshape(*new_norm.shape) * max_efficiency
    clamped_norm = torch.clamp(new_norm, min=None, max=mx_value)

    # give the projected vector the clamped norm, such that it
    # can have lost some or all energy, but not _gained_ any
    normalized = unit_norm(x, axis=-1)
    x = normalized * clamped_norm
    return x


class MultiSSM(nn.Module, EventGenerator):

    @property
    def shape_spec(self) -> ShapeSpec:
        return dict(
            control_plane_choice=(1, self.n_control_planes)
        )

    def __init__(
            self, context_dim: int,
            control_plane_dim: int,
            n_frames: int,
            state_dim: int,
            window_size: int,
            n_models: int,
            n_control_planes: int,
            n_samples: int):
        super().__init__()
        self.context_dim = context_dim
        self.control_plane_dim = control_plane_dim
        self.n_frames = n_frames
        self.state_dim = state_dim
        self.window_size = window_size
        self.n_control_planes = n_control_planes
        self.n_models = n_models
        self.n_samples = n_samples

        self.control_plane_selection = Lookup(
            self.n_control_planes,
            n_samples=control_plane_dim * n_frames,
            initialize=lambda x: torch.zeros_like(x).uniform_(-1, 1),
            selection_type='sparse_softmax')

        # self.models = nn.ModuleList(
        #     [SSM(control_plane_dim, window_size, state_dim, windowed=True) for _ in range(n_models)])

        self.ssm = SSM(control_plane_dim, window_size, state_dim, windowed=True)

        self.scheduler = DiracScheduler(1, n_frames, n_samples)

    def forward(self, control_plane_choice: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        batch = control_plane_choice.shape[0]

        cp = self.control_plane_selection \
            .forward(control_plane_choice) \
            .view(batch, self.control_plane_dim, self.n_frames)

        cp = torch.softmax(cp.view(batch, -1), dim=-1).view(batch, self.control_plane_dim, self.n_frames)
        cp = sparsify(cp, n_to_keep=8)

        samples = self.ssm.forward(cp)
        samples = self.scheduler.schedule(times, samples)
        return samples


class SampleResonanceLookup(Lookup):
    def __init__(self, n_items: int, n_samples: int):
        def init(x: torch.Tensor) -> torch.Tensor:
            noise = torch.zeros_like(x).uniform_(-1, 1)
            ramp = torch.linspace(1, 0, n_samples, device=x.device)[None, :]
            decays = torch.linspace(2, 80, n_items, device=x.device)[:, None]
            resonances = (ramp ** decays) * noise
            return resonances

        super().__init__(n_items, n_samples, initialize=init, selection_type='relu')

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        # items = max_norm(items, dim=-1)
        items = unit_norm(items)
        return items


class MultibandResonanceLookup(Lookup):

    def __init__(
            self,
            n_items: int,
            n_samples: int,
            smallest_band_size: int = 512,
            base_resonance: float = 0.2,
            window_size: int = 64):
        def init(x):
            return torch.zeros_like(x).uniform_(-6, 6) * torch.zeros_like(x).bernoulli_(p=0.01)

        step_size = window_size // 2

        full_size_log2 = int(np.log2(n_samples))
        small_size_log2 = int(np.log2(smallest_band_size))
        band_sizes = [2 ** x for x in range(small_size_log2, full_size_log2, 1)]
        n_bands = len(band_sizes)

        n_coeffs = window_size // 2 + 1
        # magnitude and phase for each band
        params_per_band = n_coeffs * 3
        total_params_per_item = params_per_band * n_bands
        super().__init__(n_items, total_params_per_item, selection_type='relu', initialize=init)

        self.base_resonance = base_resonance
        self.resonance_span = 1 - base_resonance
        self.frames_per_band = [(size // step_size) for size in band_sizes]
        self.total_frames = (n_samples // step_size) * 2
        self.band_sizes = band_sizes
        self.n_bands = n_bands
        self.params_per_band = params_per_band
        self.n_coeffs = n_coeffs
        self.window_size = window_size
        self.padded_samples = n_samples * 2

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        batch, n_events, expressivity, n_params = items.shape

        bands = dict()

        for i, size in enumerate(self.band_sizes):
            start = i * self.params_per_band
            stop = start + self.params_per_band
            band_params = items[:, :, :, start: stop]

            mag = band_params[:, :, :, :self.n_coeffs]
            phase = band_params[:, :, :, self.n_coeffs:self.n_coeffs * 2]
            start = band_params[:, :, :, -self.n_coeffs:]

            mag = self.base_resonance + ((torch.sigmoid(mag) * self.resonance_span) * 0.9999)
            phase = torch.tanh(phase) * np.pi
            start = torch.sigmoid(start)

            band = freq_domain_transfer_function_to_resonance(
                window_size=self.window_size,
                coeffs=mag,
                n_frames=self.frames_per_band[i],
                apply_decay=True,
                start_phase=phase,
                start_mags=start
            )
            bands[size] = ensure_last_axis_length(band, size * 2)

        full = fft_frequency_recompose(bands, desired_size=self.padded_samples)
        full = full[..., :self.padded_samples // 2]
        full = full.view(batch, n_events, expressivity, -1)
        full = unit_norm(full)
        return full



class FFTResonanceLookup(Lookup):
    def __init__(self, n_items: int, n_samples: int, window_size: int, base_resonance: float = 0.5):
        def init(x):
            return torch.zeros_like(x).uniform_(-6, 6) * torch.zeros_like(x).bernoulli_(p=0.01)

        self.chunk_size = (window_size // 2 + 1)
        n_coeffs = self.chunk_size * 3

        step_size = window_size // 2

        super().__init__(n_items, n_coeffs, initialize=init, selection_type='relu')


        self.base_resonance = base_resonance
        self.window_size = window_size
        self.n_coeffs = n_coeffs
        self.step_size = step_size
        self.n_frames = n_samples // self.step_size
        self.span = 1 - self.base_resonance

    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        batch, n_events, expressivity, n_coeffs = items.shape

        # mags = torch.sigmoid(items) * 0.9999

        mags = self.base_resonance + ((torch.sigmoid(items[..., :self.chunk_size]) * 0.9999) * self.span)

        phases = torch.tanh(items[..., self.chunk_size: self.chunk_size * 2]) * np.pi

        starts = torch.sigmoid(items[..., -self.chunk_size:])

        items = freq_domain_transfer_function_to_resonance(
            self.window_size,
            mags,
            self.n_frames,
            apply_decay=True,
            start_phase=phases,
            start_mags=starts,
            log_space_scan=True
        )

        items = items.view(batch, n_events, expressivity, -1)
        items = unit_norm(items, dim=-1)
        return items



class WavetableLookup(Lookup):
    def __init__(
            self,
            n_items: int,
            n_samples: int,
            n_resonances: int,
            samplerate: int,
            learnable: bool = False,
            wavetable_device=None):

        super().__init__(n_items, n_resonances)

        # w = make_waves(n_samples, np.linspace(20, 4000, num=n_resonances // 4), samplerate)

        w = make_waves_vectorized(n_samples, np.linspace(20, 4000, num=n_resonances // 4), samplerate)

        if learnable:
            self.waves = nn.Parameter(w)
        else:
            # TODO: This should be registered as a buffer, but I'm trying to understand what
            # the total parameter size is, and this one is easily compute-able/recreate-able
            # self.register_buffer('waves', w)
            self.waves = w.to(wavetable_device or device)

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

        def sample_lookup_init(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x).uniform_(-1, 1)

        super().__init__(n_items, n_samples, initialize=sample_lookup_init, selection_type='relu')
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

        # TODO: Unit norm
        x = unit_norm(x)
        return x


class EnvelopesMLP(nn.Module):
    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            env_samples: int,
            full_size: int,
            padded_size: int):

        super().__init__()
        self.input_channels = input_channels
        self.env_samples = env_samples
        self.full_size = full_size
        self.padded_size = padded_size

        self.network = LinearOutputStack(
            channels=hidden_channels,
            layers=3,
            out_channels=env_samples,
            in_channels=input_channels,
            norm=lambda channels: nn.LayerNorm([channels, ])
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = self.network(x)
        envelope = envelope.view(x.shape[0], 1, -1)
        amp = interpolate_last_axis(envelope, desired_size=self.full_size)
        amp = amp * torch.zeros_like(amp).uniform_(-1, 1)
        amp = ensure_last_axis_length(amp, self.padded_size)
        return amp


class Envelopes(Lookup):
    def __init__(
            self,
            n_items: int,
            n_samples: int,
            full_size: int,
            padded_size: int,
            max_events: int,
            with_noise: bool = False):

        # def init(x: torch.Tensor):
        #     ramp = torch.linspace(1, 0, n_samples, device=x.device)[None, :]
        #     exp = torch.linspace(2, 40, n_items, device=x.device)[:, None]
        #     x = ramp ** exp
        #     return x


        super().__init__(
            n_items,
            n_samples * max_events,
            # initialize=init,
            selection_type='relu'
        )
        self.with_noise = with_noise
        self.max_events = max_events
        self.full_size = full_size
        self.padded_size = padded_size

    # def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
    #     """Ensure that we have all values between 0 and 1
    #     """
    #     return items ** 2

    def postprocess_results(self, envelope: torch.Tensor) -> torch.Tensor:
        """Scale up to sample rate and multiply with noise
        """

        amp = envelope.view(*envelope.shape[:-1], self.max_events, -1)
        # amp = torch.relu(envelope)
        if not self.with_noise:
            amp = sparse_softmax(amp, dim=-1, normalize=False)
        amp = torch.sum(amp, dim=-2)

        # scale so we always have a max of one and min of zero
        # envelope = envelope - torch.min(envelope, dim=-1, keepdim=True)[0]
        # envelope = envelope / (torch.max(envelope, dim=-1, keepdim=True)[0] + 1e-8)

        # amp = sparsify(envelope, n_to_keep=64)
        amp = interpolate_last_axis(amp, desired_size=self.full_size)

        if self.with_noise:
            amp = amp * torch.zeros_like(amp).uniform_(-1, 1)

        amp = ensure_last_axis_length(amp, self.padded_size)
        # diff = self.padded_size - self.full_size
        # padding = torch.zeros((amp.shape[:-1] + (diff,)), device=amp.device)
        # amp = torch.cat([amp, padding], dim=-1)
        return amp


# class ConvDeformations(nn.Module):
#     def __init__(self, context_dim: int, channels: int, frames: int, full_size: int, hidden_channels: int):
#         super().__init__()
#         self.context_dim = context_dim
#         self.channels = channels
#         self.frames = frames
#         self.full_size = full_size
#         self.hidden_channels = hidden_channels
#
#         self.net = ConvUpsample(
#             latent_dim=self.context_dim,
#             channels=self.hidden_channels,
#             start_size=8,
#             end_size=self.frames,
#             mode='nearest',
#             out_channels=self.channels,
#             from_latent=True,
#             weight_norm=True
#         )
#
#     def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         x = self.net(x)
#         x = torch.softmax(x, dim=-2)
#         x = x[:, None, :, :]
#         before_upsample = x
#         x = interpolate_last_axis(x, desired_size=self.full_size)
#         return x, before_upsample


# class PeriodicDeformations(nn.Module):
#     def __init__(self, in_channels: int, channels: int, frames: int, full_size):
#         super().__init__()
#         self.in_channels = in_channels
#         self.channels = channels
#         self.frames = frames

class DeformationsMLP(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int, expressivity: int, frames: int, full_size: int):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.expressivity = expressivity
        self.frames = frames
        self.full_size = full_size

        self.network = LinearOutputStack(
            channels=hidden_channels,
            layers=3,
            out_channels=expressivity * frames,
            in_channels=input_channels,
            norm=lambda channels: nn.LayerNorm([channels, ])
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_events, context_dim = x.shape

        x = self.network(x)
        x = x.view(batch, n_events, self.expressivity, self.frames)
        # TODO: sparsify and then do cumsum
        x = torch.softmax(x, dim=-2)
        before_upsample = x
        x = interpolate_last_axis(x, desired_size=self.full_size)
        return x, before_upsample

class Deformations(Lookup):

    def __init__(self, n_items: int, channels: int, frames: int, full_size: int):
        # values_per_item = 3 # phase, frequency, amplitude

        super().__init__(n_items, channels * frames, selection_type='relu')
        self.full_size = full_size
        self.channels = channels
        self.frames = frames
        # self.values_per_item = values_per_item

    # def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
    #     t = torch.linspace(-np.pi, np.pi, self.frames, device=items.device)[None, None, None, :]
    #     items = items.view(*items.shape[:-1], self.channels, self.values_per_item)
    #
    #     phases = items[..., 0:1] * np.pi
    #     freqs = items[..., 1:2] * np.pi
    #     amps = items[..., 2:3]
    #
    #     x = amps * torch.sin((phases + freqs) * t)
    #     x = torch.softmax(x, dim=-2)
    #     before_upsample = x
    #     x = interpolate_last_axis(x, desired_size=self.full_size)
    #     return x, before_upsample

    def postprocess_results(self, items: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape so that the values are (..., channels, frames).  Apply
        softmax to the channel dimension and then upscale to full samplerate
        """
        shape = items.shape[:-1]
        x = items.reshape(*shape, self.channels, self.frames)
        x = torch.cumsum(x, dim=-1)
        x = torch.softmax(x, dim=-2)
        # x = torch.relu(x)
        before_upsample = x
        x = interpolate_last_axis(x, desired_size=self.full_size)
        return x, before_upsample





class WavetableModel(nn.Module, EventGenerator):

    def __init__(
            self,
            n_items: int,
            n_samples: int,
            n_frames: int,
            n_events: int):
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.n_events = n_events

        self.items = Lookup(n_items, 16384, fixed=False, selection_type='softmax')
        verbs = NeuralReverb.tensors_from_directory(Config.impulse_response_path(), n_samples)
        n_verbs = verbs.shape[0]
        self.n_verbs = n_verbs
        self.verb = Lookup(n_verbs, n_samples, initialize=lambda x: verbs, fixed=True)
        self.scheduler = DiracScheduler(
            self.n_events, start_size=n_frames, n_samples=self.n_samples, pre_sparse=True)

    @property
    def shape_spec(self) -> ShapeSpec:
        return dict(
            amplitudes=(1,),
            mix=(self.n_items,),
            room_choice=(self.n_verbs,),
            room_mix=(2,),
        )

    def forward(
            self,
            amplitudes: torch.Tensor,
            mix: torch.Tensor,
            room_choice: torch.Tensor,
            room_mix: torch.Tensor,
            times: torch.Tensor) -> torch.Tensor:
        batch_size = amplitudes.shape[0]

        dry = self.items.forward(mix)
        dry = dry + torch.zeros_like(dry).uniform_(-1e-7, 1e-7)

        dry = F.pad(dry, (0, self.n_samples - dry.shape[-1]))
        verb = self.verb.forward(room_choice)
        wet = fft_convolve(dry, verb)

        stacked = torch.stack([dry, wet], dim=-1)
        final = stacked @ torch.softmax(room_mix, dim=-1)[:, :, :, None]
        final = final.reshape(batch_size, -1, self.n_samples)

        # final = max_norm(final)
        final = final * torch.abs(amplitudes)
        final = self.scheduler.schedule(times, final)
        return final

class SimpleEventGenerator(nn.Module, EventGenerator):

    @property
    def shape_spec(self) -> ShapeSpec:
        params = dict(
            param=(self.context_dim,)
        )


        return params

    def __init__(self, context_dim: int, n_frames: int, n_samples: int, n_events: int, channels: int):
        super().__init__()
        self.context_dim = context_dim
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.n_events = n_events
        self.channels = channels

        self.window_size = 512

        self.n_coeffs = self.window_size // 2 + 1
        self.total_coeffs = self.n_coeffs * 2

        self.pos = nn.Parameter(torch.zeros(1, n_frames, channels).uniform_(-0.01, 0.01))
        self.proj = nn.Linear(context_dim, channels)
        self.net = LinearOutputStack(
            channels=channels,
            layers=3,
            out_channels=self.total_coeffs,
            in_channels=channels,
            # norm=lambda channels: nn.LayerNorm([channels,])
        )

        self.scheduler = DiracScheduler(
            self.n_events, start_size=n_frames, n_samples=self.n_samples, pre_sparse=True)

    def forward(self, param: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        batch_size = param.shape[0]
        x = self.proj(param)
        x = x.view(batch_size, 1, self.channels)
        x = x + self.pos
        x = self.net(x)

        x = x.view(batch_size, self.n_frames, self.n_coeffs, 2)
        mags = torch.abs(x[..., 0: 1])
        phase = x[..., 1:]

        base_phase = torch.ones_like(phase)
        noise = torch.zeros_like(base_phase).uniform_(-1, 1)
        phase = base_phase + (phase * noise)

        x = torch.cat([mags, phase], dim=-1)

        # x = torch.view_as_complex(x)
        x = mag_phase_recomposition(x, torch.linspace(0, 1, 257, device=x.device))

        x = torch.fft.irfft(x)
        x = x.view(batch_size, 1, self.n_frames, self.window_size)
        x = overlap_add(x, apply_window=True)[..., :self.n_samples]

        x = self.scheduler.schedule(times, x)
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
            hidden_channels: int,
            context_dim: int,
            fine_positioning: bool = False,
            wavetable_device=None,
            fft_resonance: bool = False,
            hierarchical_scheduling: bool = False):

        super().__init__()

        self.hierarchical_scheduling = hierarchical_scheduling
        self.noise_filter_samples = noise_filter_samples
        self.noise_expressivity = noise_expressivity
        self.n_noise_filters = n_noise_filters
        self.noise_deformations = noise_deformations
        self.n_envelopes = n_envelopes
        self.fine_positioning = fine_positioning
        self.samples_per_frame = n_samples // n_frames
        self.frame_ratio = self.samples_per_frame / n_samples
        self.context_dim = context_dim

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

        verbs = NeuralReverb.tensors_from_directory(Config.impulse_response_path(), n_samples, normalize=True)
        n_verbs = verbs.shape[0]
        self.n_verbs = n_verbs

        self.room_shape = (1, n_events, n_verbs)

        self.n_resonances = n_resonances

        if fft_resonance:
            # self.r = MultibandResonanceLookup(
            #     n_resonances,
            #     n_samples,
            #     base_resonance=0.01,
            #     smallest_band_size=2048)

            self.r = FFTResonanceLookup(n_resonances, n_samples, 2048, base_resonance=0.02)
            # self.r = SampleResonanceLookup(n_resonances, n_samples)
            # self.r = F0ResonanceLookup(n_resonances, n_samples)
        else:
            self.r = WavetableLookup(
                n_resonances,
                n_samples,
                n_resonances=n_resonances,
                samplerate=samplerate,
                learnable=False,
                wavetable_device=wavetable_device)

        # TODO: This is an issue because we have gradients for every sample, even though
        # they are largely redundant (think exponential vs.
        # self.r = SampleLookup(n_resonances, n_samples, flatten_kernel_size=2048, randomize_phases=False)
        # self.r = F0ResonanceLookup(n_resonances, n_samples)

        self.n = SampleLookup(
            n_noise_filters, noise_filter_samples, windowed=False, randomize_phases=False)

        self.verb = Lookup(n_verbs, n_samples, initialize=lambda x: verbs, fixed=True, selection_type='relu')

        self.e = Envelopes(
            n_envelopes,
            n_samples=128,
            full_size=8192,
            padded_size=self.n_samples,
            max_events=32,
            with_noise=True
        )



        # self.e = ConvEnvelopes(
        #     n_samples=128, full_size=8192, padded_size=self.n_samples, context_dim=self.context_dim, channels=32)

        self.n_decays = n_decays

        # deformation_frames = n_samples // 1024
        deformation_frames = n_frames

        # self.warp = DeformationsMLP(
        #     input_channels=context_dim,
        #     hidden_channels=256,
        #     expressivity=instr_expressivity,
        #     frames=deformation_frames,
        #     full_size=n_samples)

        self.warp = Deformations(n_deformations, instr_expressivity, deformation_frames, n_samples)

        self.noise_warp = Deformations(noise_deformations, noise_expressivity, deformation_frames, n_samples)

        if self.hierarchical_scheduling:
            self.scheduler = HierarchicalDiracModel(
                self.n_events, self.n_samples)
        else:
            self.scheduler = DiracScheduler(
                self.n_events, start_size=n_frames, n_samples=self.n_samples, pre_sparse=True)



        # self.scheduler = FFTShiftScheduler(self.n_events)

    @property
    def shape_spec(self) -> ShapeSpec:
        params = dict(
            noise_resonance=(self.noise_expressivity, self.n_noise_filters),

            # TODO: Replace these with convolutional upsampled things
            noise_deformations=(self.noise_deformations,),
            deformations=(self.n_deformations,),
            envelopes=(self.n_envelopes,),

            # noise_deformations=(self.context_dim,),
            # deformations=(self.context_dim,),
            # envelopes=(self.context_dim,),
            # resonances=(self.instr_expressivity, self.context_dim),

            noise_mixes=(2,),
            resonances=(self.instr_expressivity, self.n_resonances),
            res_filter=(self.noise_expressivity, self.n_noise_filters),
            # decays=(self.instr_expressivity, self.n_decays,),
            mixes=(2,),
            amplitudes=(1,),
            room_choice=(self.n_verbs,),
            room_mix=(2,),
        )

        if self.fine_positioning:
            params['fine'] = (1,)

        return params

    def forward_with_intermediate_steps(
            self,
            noise_resonance: torch.Tensor,
            noise_deformations: torch.Tensor,
            noise_mixes: torch.Tensor,
            envelopes: torch.Tensor,
            resonances: torch.Tensor,
            res_filter: torch.Tensor,
            deformations: torch.Tensor,
            # decays: torch.Tensor,
            mixes: torch.Tensor,
            amplitudes: torch.Tensor,
            times: torch.Tensor,
            room_choice: torch.Tensor,
            room_mix: torch.Tensor,
            fine: Union[torch.Tensor, None] = None) -> Tuple[
        torch.Tensor, Dict[str, torch.Tensor]]:

        intermediates = {}

        # calculate impulses or energy injected into a system
        impulses = self.e.forward(envelopes)

        # choose filters to be convolved with energy/noise
        noise_res = self.n.forward(noise_resonance)
        noise_res = torch.cat([
            noise_res,
            torch.zeros(*noise_res.shape[:-1], self.n_samples - noise_res.shape[-1], device=impulses.device)
        ], dim=-1)

        # choose deformations to be applied to the initial noise resonance
        noise_def, _ = self.noise_warp.forward(noise_deformations)

        # choose a dry/wet mix
        noise_mix = noise_mixes[:, :, None, :]
        noise_mix = torch.softmax(noise_mix, dim=-1)

        # convolve the initial impulse with all filters, then mix together
        noise_wet = fft_convolve(impulses[:, :, None, :], noise_res)
        noise_wet = noise_wet * noise_def
        noise_wet = torch.sum(noise_wet, dim=2)

        intermediates['impulse'] = noise_wet

        # mix dry and wet
        stacked = torch.stack([impulses, noise_wet], dim=-1)
        mixed = stacked * noise_mix
        mixed = torch.sum(mixed, dim=-1)
        # mixed = noise_wet

        # initial filtered noise is now the input to our longer resonances
        impulses = mixed

        # choose a number of resonances to be convolved with
        # those impulses
        resonance = self.r.forward(resonances)
        # res_filters = self.n.forward(res_filter)

        # res_filters = torch.cat([
        #     res_filters,
        #     torch.zeros(*res_filters.shape[:-1], resonance.shape[-1] - res_filters.shape[-1], device=res_filters.device)
        # ], dim=-1)

        # resonance = fft_convolve(resonance, res_filters)

        # describe how we interpolate between different
        # resonances over time
        deformations, before_upsample = self.warp.forward(deformations)
        intermediates['deformations'] = before_upsample

        # determine how each resonance decays or leaks energy
        # decays = self.d.forward(decays)
        decaying_resonance = resonance

        dry = impulses[:, :, None, :]

        # convolve the impulse with all the resonances and
        # interpolate between them
        conv = fft_convolve(dry, decaying_resonance)
        # conv = dry
        with_deformations = conv * deformations
        audio_events = torch.sum(with_deformations, dim=2, keepdim=True)

        # mix the dry and wet signals
        mixes = mixes[:, :, None, None, :]
        mixes = torch.softmax(mixes, dim=-1)

        stacked = torch.stack([dry, audio_events], dim=-1)
        mixed = stacked * mixes
        final = torch.sum(mixed, dim=-1)

        intermediates['dry'] = final

        # apply reverb
        verb = self.verb.forward(room_choice)
        wet = fft_convolve(verb, final.view(*verb.shape))
        verb_mix = torch.softmax(room_mix, dim=-1)[:, :, None, :]
        stacked = torch.stack([wet, final.view(*verb.shape)], dim=-1)
        stacked = stacked * verb_mix

        final = stacked.sum(dim=-1)


        intermediates['wet'] = final

        # apply amplitudes
        final = final.view(-1, self.n_events, self.n_samples)

        if self.hierarchical_scheduling:
            final = final * torch.abs(amplitudes)

        scheduled = self.scheduler.schedule(times, final)

        if fine is not None:
            fine_shifts = torch.tanh(fine) * self.frame_ratio
            scheduled = fft_shift(scheduled, fine_shifts)
            scheduled = scheduled[..., :self.n_samples]

        return scheduled, intermediates

    def forward(
            self,
            noise_resonance: torch.Tensor,
            noise_deformations: torch.Tensor,
            noise_mixes: torch.Tensor,
            envelopes: torch.Tensor,
            resonances: torch.Tensor,
            res_filter: torch.Tensor,
            deformations: torch.Tensor,
            # decays: torch.Tensor,
            mixes: torch.Tensor,
            amplitudes: torch.Tensor,
            times: torch.Tensor,
            room_choice: torch.Tensor,
            room_mix: torch.Tensor,
            fine: Union[torch.Tensor, None] = None) -> torch.Tensor:

        scheduled, intermediates = self.forward_with_intermediate_steps(
            noise_resonance,
            noise_deformations,
            noise_mixes,
            envelopes,
            resonances,
            res_filter,
            deformations,
            # decays,
            mixes,
            amplitudes,
            times,
            room_choice,
            room_mix,
            fine)

        return scheduled
