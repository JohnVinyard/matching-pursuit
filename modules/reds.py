from typing import Callable, List, Union
import torch
from torch import nn

from modules.fft import fft_convolve
from modules.normal_pdf import gamma_pdf, pdf2
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax
from modules.transfer import gaussian_bandpass_filtered, make_waves
from torch.nn import functional as F
from enum import Enum
import numpy as np

from util.music import musical_scale_hz


def interpolate_last_axis(low_sr: torch.Tensor, desired_size) -> torch.Tensor:
    """A convenience wrapper around `torch.nn.functional.interpolate` to allow
    for an arbitrary number of leading dimensions
    """
    orig_shape = low_sr.shape
    new_shape = orig_shape[:-1] + (desired_size,)
    last_dim = low_sr.shape[-1]

    reshaped = low_sr.reshape(-1, 1, last_dim)
    upsampled = F.interpolate(reshaped, mode='linear', size=desired_size)
    upsampled = upsampled.reshape(*new_shape)
    return upsampled


def fft_shift(a: torch.Tensor, shift: torch.Tensor):
    n_samples = a.shape[-1]
    shift_samples = shift * n_samples
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')
    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs) * 2j * np.pi).to(a.device) / n_coeffs
    shift = torch.exp(-shift * shift_samples)
    spec = spec * shift
    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples


class EnvelopeType(Enum):
    Gaussian = 'Gaussian'
    Gamma = 'Gamma'


def exponential_decay(
        decay_values: torch.Tensor,
        n_atoms: int,
        n_frames: int,
        base_resonance: float,
        n_samples: int):
    decay_values = torch.sigmoid(decay_values.view(-1, n_atoms, 1).repeat(1, 1, n_frames))
    resonance_factor = (1 - base_resonance) * 0.99
    decay = base_resonance + (decay_values * resonance_factor)
    decay = torch.log(decay + 1e-12)
    decay = torch.cumsum(decay, dim=-1)
    decay = torch.exp(decay).view(-1, n_atoms, n_frames)
    amp = F.interpolate(decay, size=n_samples, mode='linear')
    return amp


class BandPassFilteredNoise(nn.Module):
    def __init__(self, n_samples: int):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, means: torch.Tensor, stds: torch.Tensor):
        batch, n_events = means.shape

        assert means.shape == stds.shape
        noise = torch.zeros(1, n_events, self.n_samples, device=means.device).uniform_(-1, 1)

        filtered_noise = gaussian_bandpass_filtered(means, stds, noise)
        assert filtered_noise.shape == (batch, n_events, self.n_samples)
        return filtered_noise


class F0Resonance(nn.Module):
    """
    TODO: Try the following:
    
    - wavetable with fft_shift
    - wavetable with gaussian sampling kernel
    """

    def __init__(
            self,
            n_octaves: int,
            n_samples: int,
            min_hz: int = 20,
            max_hz: int = 3000,
            samplerate: int = 22050):
        super().__init__()
        self.samplerate = samplerate
        self.min_hz = min_hz
        self.max_hz = max_hz
        factors = torch.arange(1, 1 + n_octaves, 1)
        self.n_octaves = n_octaves
        self.n_samples = n_samples

        # self.register_buffer('powers', 1 / 2 ** (torch.arange(1, 1 + n_f0_elements, 1)))
        self.register_buffer('factors', factors)

        self.min_freq = self.min_hz / (samplerate // 2)
        self.max_freq = self.max_hz / (samplerate // 2)
        self.freq_range = self.max_freq - self.min_freq

    def forward(
            self,
            f0: torch.Tensor,
            decay_coefficients: torch.Tensor,
            freq_spacing: torch.Tensor,
            sigmoid_decay: bool = True,
            apply_exponential_decay: bool = True,
            time_decay: Union[torch.Tensor, None] = None) -> torch.Tensor:

        batch, n_events, n_elements = f0.shape

        # assert n_elements == self.n_f0_elements

        f0 = torch.abs(
            f0  # @ self.powers
        )
        f0 = f0.view(batch, n_events, 1)

        # print(decay_coefficients.shape, f0.shape)

        assert decay_coefficients.shape == f0.shape
        # assert phase_offsets.shape == (batch, n_events, 1)

        # phase_offsets = torch.sigmoid(phase_offsets) * np.pi

        exp_decays = exponential_decay(
            torch.sigmoid(decay_coefficients) if sigmoid_decay else decay_coefficients,
            n_atoms=n_events,
            n_frames=self.n_octaves,
            base_resonance=0.01,
            n_samples=self.n_octaves)
        assert exp_decays.shape == (batch, n_events, self.n_octaves)

        # frequencies in radians
        f0 = self.min_freq + (f0 * self.freq_range)
        f0 = f0 * np.pi

        # octaves
        factors = freq_spacing.repeat(1, 1, self.n_octaves)

        # factors.fill_(1)

        factors = torch.cumsum(factors, dim=-1)
        f0s = f0 * factors
        assert f0s.shape == (batch, n_events, self.n_octaves)

        # filter out anything above nyquist
        mask = f0s < 1
        f0s = f0s * mask

        # generate sine waves
        f0s = f0s.view(batch, n_events, self.n_octaves, 1).repeat(1, 1, 1, self.n_samples)
        osc = torch.sin(
            torch.cumsum(f0s, dim=-1)  # + phase_offsets[..., None]
        )

        assert osc.shape == (batch, n_events, self.n_octaves, self.n_samples)

        # apply decaying value
        if apply_exponential_decay:
            osc = osc * exp_decays[..., None]

        if time_decay is not None:
            frames = time_decay.shape[-1]
            ramp = torch.linspace(1, 0, frames, device=time_decay.device)
            ramp = ramp ** time_decay
            ramp = interpolate_last_axis(ramp, desired_size=self.n_samples)
            ramp = ramp.view(-1, 1, self.n_samples)
            print(osc.shape, ramp.shape)
            osc = osc * ramp

        osc = torch.sum(osc, dim=2)
        osc = max_norm(osc, dim=-1)

        assert osc.shape == (batch, n_events, self.n_samples)
        return osc


class ExponentialDecayEnvelope(nn.Module):
    def __init__(self, base_resonance: float, n_frames: int, n_samples: int):
        super().__init__()
        self.base_resonance = base_resonance
        self.n_samples = n_samples
        self.n_frames = n_frames

    def forward(self, decay_values: torch.Tensor):
        batch, n_events, _ = decay_values.shape
        envelopes = exponential_decay(
            decay_values,
            n_atoms=n_events,
            n_frames=self.n_frames,
            base_resonance=self.base_resonance,
            n_samples=self.n_samples)
        return envelopes


class EvolvingFilteredResonance(nn.Module):
    def __init__(self, base_crossfade_resonance: float, crossfade_frames: int, n_samples: int):
        super().__init__()
        self.n_samples = n_samples
        self.base_crossfade_resonance = base_crossfade_resonance
        self.crossfade_frames = crossfade_frames

    def forward(
            self,
            resonances: torch.Tensor,
            decays: torch.Tensor,
            start_filter_means: torch.Tensor,
            start_filter_stds: torch.Tensor,
            end_filter_means: torch.Tensor,
            end_filter_stds: torch.Tensor):
        batch, n_events, n_samples = resonances.shape
        assert n_samples == self.n_samples

        batch, n_events, _ = decays.shape

        assert resonances.shape[:2] == decays.shape[:2]

        start_resonance = gaussian_bandpass_filtered(
            start_filter_means, start_filter_stds, resonances)
        end_resonance = gaussian_bandpass_filtered(
            end_filter_means, end_filter_stds, resonances)

        filt_crossfade = exponential_decay(
            decays,
            n_atoms=n_events,
            n_frames=self.crossfade_frames,
            base_resonance=self.base_crossfade_resonance,
            n_samples=self.n_samples)
        filt_crossfade_inverse = 1 - filt_crossfade

        filt_crossfade_stacked = torch.cat([
            filt_crossfade[..., None],
            filt_crossfade_inverse[..., None]], dim=-1)

        assert filt_crossfade_stacked.shape == (batch, n_events, self.n_samples, 2)

        return start_resonance, end_resonance, filt_crossfade_stacked


class EnvelopeAndPosition(nn.Module):
    def __init__(self, n_samples: int, envelope_type: EnvelopeType):
        super().__init__()
        self.n_samples = n_samples
        self.envelope_type = envelope_type

        self.gamma_ramp_size = 128
        self.gamma_ramp_exponent = 2
        self.gaussian_envelope_factor = 0.1

    def forward(
            self,
            signals: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor,
            adjustment: Union[torch.Tensor, None],
            shifts: Union[torch.Tensor, None]):

        batch, n_events, n_samples = signals.shape
        assert n_samples == self.n_samples

        batch, n_events = a.shape
        assert a.shape == b.shape

        if self.envelope_type == EnvelopeType.Gaussian.value:
            envelopes = pdf2(a, (torch.abs(b) + 1e-12) * self.gaussian_envelope_factor, self.n_samples)
        elif self.envelope_type == EnvelopeType.Gamma.value:
            envelopes = gamma_pdf((torch.abs(a) + 1e-12), (torch.abs(b) + 1e-12), self.n_samples)
            ramp = torch.zeros_like(envelopes)
            ramp[..., :self.gamma_ramp_size] = torch.linspace(0, 1, self.gamma_ramp_size)[None, None,
                                               :] ** self.gamma_ramp_exponent
            envelopes = envelopes * ramp
        else:
            raise ValueError(f'{self.envelope_type.value} is not supported')

        assert envelopes.shape == (batch, n_events, self.n_samples)

        positioned_signals = signals * envelopes

        if adjustment is not None:
            # shifts = sparse_softmax(adjustment, dim=-1, normalize=True)
            # shifts = hard_shift_choice(adjustment)
            shifts = adjustment
            positioned_signals = fft_convolve(positioned_signals, shifts)

        if shifts is not None:
            positioned_signals = fft_shift(positioned_signals, shifts)

        return positioned_signals


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, signals: List[torch.Tensor], mix: torch.Tensor):
        stacked_signals = torch.cat([s[..., None] for s in signals], dim=-1)
        batch, n_events, n_samples, n_channels = stacked_signals.shape

        assert n_channels == len(signals)

        # this is a mix that varies over time
        is_time_series = mix.shape == (batch, n_events, n_samples, n_channels)
        # this is a mix that is constant over time
        is_global = mix.shape == (batch, n_events, 1, n_channels)

        assert is_time_series or is_global

        result = torch.sum(stacked_signals * mix, dim=-1)
        assert result.shape == (batch, n_events, n_samples)
        return result


class Resonance(nn.Module):
    def __init__(
            self,
            n_resonances: int,
            n_samples: int,
            hard_choice: Union[bool, Callable],
            samplerate: int):

        super().__init__()
        self.n_resonances = n_resonances
        self.n_samples = n_samples
        self.hard_choice = hard_choice

        # we generate pure sines, triangle, sawtooth and square waves with these fundamental
        # frequencies, hence `n_resonances // 4`
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=n_resonances // 4)
        waves = make_waves(n_samples, f0s, samplerate)
        self.register_buffer('waves', waves.view(1, n_resonances, n_samples))

    def forward(self, choice: torch.Tensor):
        print(choice.shape)

        batch, n_events, n_resonances = choice.shape
        assert n_resonances == self.n_resonances

        if self.hard_choice:
            if isinstance(self.hard_choice, bool):
                resonances = sparse_softmax(choice, normalize=True, dim=-1)
            else:
                resonances = self.hard_choice(choice)
        else:
            resonances = torch.relu(choice)

        resonances = resonances @ self.waves
        assert resonances.shape == (batch, n_events, self.n_samples)
        return resonances


class RedsLikeModel(nn.Module):
    """
    A model representing audio with the following parameters
    
    n_atoms * (env(2) + mix(2) + decay(1) + decay(1) + res_choice(1) + noise_filter(2) + res_filter(2) + res_filter2(2) + amps(1) + verb_choice(1) + verb_mix(1))
    
    n_atoms * 16
    """

    def __init__(
            self,
            n_resonance_octaves=64,
            n_samples: int = 2 ** 15,
            samplerate: int = 22050,
            use_wavetables: bool = False):

        super().__init__()

        self.use_wavetables = use_wavetables
        self.n_resonance_octaves = n_resonance_octaves
        self.n_samples = n_samples
        self.samplerate = samplerate

        n_octaves = self.n_resonance_octaves
        self.resonance_generator = F0Resonance(
            n_octaves=n_octaves, n_samples=n_samples)

        hard_res_choice = lambda x: x

        self.wavteable_resonance_generator = Resonance(
            4096, n_samples, hard_choice=hard_res_choice, samplerate=samplerate)

        self.noise_generator = BandPassFilteredNoise(n_samples)
        self.amp_envelope_generator = ExponentialDecayEnvelope(
            base_resonance=0.02,
            n_frames=128,
            n_samples=n_samples)
        self.evolving_resonance = EvolvingFilteredResonance(
            base_crossfade_resonance=0.02,
            crossfade_frames=128,
            n_samples=n_samples)

        self.env_and_position = EnvelopeAndPosition(
            n_samples=n_samples,
            envelope_type=EnvelopeType.Gamma.value)

        self.mixer = Mixer()

        hard_reverb_choice = lambda x: sparse_softmax(x, normalize=True, dim=-1)  # locked

        self.verb = ReverbGenerator(
            4, 2, samplerate, n_samples, norm=nn.LayerNorm(4, ), hard_choice=False)

    @property
    def n_reverb_rooms(self):
        return self.verb.n_rooms

    def generate_test_data(self,
                           noise_osc_mix: torch.Tensor,
                           f0_choice: torch.Tensor,
                           decay_choice: torch.Tensor,
                           freq_spacing: torch.Tensor,
                           noise_filter: torch.Tensor,
                           filter_decays: torch.Tensor,
                           resonance_filter: torch.Tensor,
                           resonance_filter2: torch.Tensor,
                           decays: torch.Tensor,
                           shifts: torch.Tensor,
                           env: torch.Tensor,
                           verb_room_choice: torch.Tensor,
                           verb_mix: torch.Tensor,
                           amplitudes: torch.Tensor) -> torch.Tensor:

        batch, n_atoms, _ = f0_choice.shape

        overall_mix = torch.softmax(noise_osc_mix, dim=-1)

        resonances = self.resonance_generator.forward(
            f0_choice, decay_choice, freq_spacing)

        filtered_noise = self.noise_generator.forward(
            noise_filter[:, :, 0],
            (torch.abs(noise_filter[:, :, 1]) + 1e-12))

        filtered_resonance, filt_res_2, filt_crossfade_stacked = self.evolving_resonance.forward(
            resonances=resonances,
            decays=filter_decays,
            start_filter_means=torch.zeros_like(resonance_filter[:, :, 0]),
            start_filter_stds=torch.abs(resonance_filter[:, :, 1]) + 1e-12,
            end_filter_means=torch.zeros_like(resonance_filter2[:, :, 0]),
            end_filter_stds=torch.abs(resonance_filter2[:, :, 1]) + 1e-12
        )

        decays = self.amp_envelope_generator.forward(decays)

        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays

        positioned_noise = self.env_and_position.forward(
            signals=filtered_noise,
            a=env[:, :, 0],
            b=env[:, :, 1],
            adjustment=None,
            shifts=shifts)

        res = fft_convolve(
            positioned_noise,
            decaying_resonance)

        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2
        )

        mixed = self.mixer.forward([res, res2], filt_crossfade_stacked)

        final = self.mixer.forward([positioned_noise, mixed], overall_mix[:, :, None, :])

        assert final.shape[1:] == (n_atoms, self.n_samples)

        final = final.view(-1, n_atoms, self.n_samples)
        final = unit_norm(final, dim=-1)
        amps = torch.abs(amplitudes)
        final = final * amps

        # final, rm, mx = self.verb.forward(verb_params, final, return_parameters=True)
        final = self.verb.direct(final, verb_room_choice, verb_mix)

        return final

    def forward(
            self,
            mix: torch.Tensor,
            f0_choice: torch.Tensor,
            decay_choice: torch.Tensor,
            freq_spacing: torch.Tensor,
            noise_filter: torch.Tensor,
            filter_decays: torch.Tensor,
            resonance_filter: torch.Tensor,
            resonance_filter2: torch.Tensor,
            decays: torch.Tensor,
            shifts: torch.Tensor,
            env: torch.Tensor,
            verb_params: torch.Tensor,
            amplitudes: torch.Tensor):

        batch, n_atoms, _ = f0_choice.shape

        overall_mix = torch.softmax(mix, dim=-1)

        if self.use_wavetables:
            resonances = self.wavteable_resonance_generator.forward(f0_choice)
        else:
            resonances = self.resonance_generator.forward(
                f0_choice, decay_choice, freq_spacing)

        filtered_noise = self.noise_generator.forward(
            noise_filter[:, :, 0],
            (torch.abs(noise_filter[:, :, 1]) + 1e-12))

        filtered_resonance, filt_res_2, filt_crossfade_stacked = self.evolving_resonance.forward(
            resonances=resonances,
            decays=filter_decays,
            start_filter_means=torch.zeros_like(resonance_filter[:, :, 0]),
            start_filter_stds=torch.abs(resonance_filter[:, :, 1]) + 1e-12,
            end_filter_means=torch.zeros_like(resonance_filter2[:, :, 0]),
            end_filter_stds=torch.abs(resonance_filter2[:, :, 1]) + 1e-12
        )

        decays = self.amp_envelope_generator.forward(decays)

        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays

        positioned_noise = self.env_and_position.forward(
            signals=filtered_noise,
            a=env[:, :, 0],
            b=env[:, :, 1],
            adjustment=shifts)

        res = fft_convolve(
            positioned_noise,
            decaying_resonance)

        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2
        )

        mixed = self.mixer.forward([res, res2], filt_crossfade_stacked)

        final = self.mixer.forward([positioned_noise, mixed], overall_mix[:, :, None, :])

        assert final.shape[1:] == (n_atoms, self.n_samples)

        final = final.view(-1, n_atoms, self.n_samples)
        # final = unit_norm(final, dim=-1)
        amps = torch.abs(amplitudes)
        # final = final * amps

        # rm is a one-hot room choice
        # mx is a two-element, softmax distribution
        final, rm, mx = self.verb.forward(verb_params, final, return_parameters=True)

        return final, amps
