from enum import Enum
from typing import Callable, List, Union
import torch
from torch import nn
from torch.nn import functional as F

from modules.eventgenerators.generator import EventGenerator, ShapeSpec
from modules.fft import fft_convolve
from modules.normal_pdf import gamma_pdf, pdf2
from modules.normalization import unit_norm
from modules.reds import F0Resonance
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax
from modules.transfer import gaussian_bandpass_filtered, make_waves
from overfitresonance import DiracScheduler, HierarchicalDiracModel
from util.music import musical_scale_hz


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
    def __init__(self, n_samples: int, n_atoms: int = 1):
        super().__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms

    def forward(self, means: torch.Tensor, stds: torch.Tensor):
        batch, n_events = means.shape

        assert means.shape == stds.shape
        noise = torch.zeros(1, self.n_atoms, self.n_samples, device=means.device).uniform_(-1, 1)

        filtered_noise = gaussian_bandpass_filtered(means, stds, noise)
        assert filtered_noise.shape == (batch, n_events, self.n_samples)
        return filtered_noise


class Resonance(nn.Module):
    def __init__(self, n_resonances: int, n_samples: int, samplerate: int, hard_choice: Union[bool, Callable]):
        super().__init__()
        self.n_resonances = n_resonances
        self.n_samples = n_samples
        self.hard_choice = hard_choice
        self.samplerate = samplerate

        # we generate pure sines, triangle, sawtooth and square waves with these fundamental
        # frequencies, hence `n_resonances // 4`
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=n_resonances // 4)
        waves = make_waves(n_samples, f0s, samplerate)
        self.register_buffer('waves', waves.view(1, n_resonances, n_samples))

    def forward(self, choice: torch.Tensor):
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
    def __init__(
            self,
            n_samples: int,
            envelope_type: EnvelopeType,
            gaussian_envelope_factor: float = 0.1):

        super().__init__()
        self.n_samples = n_samples
        self.envelope_type = envelope_type
        self.gaussian_envelope_factor = gaussian_envelope_factor

        self.gamma_ramp_size = 128
        self.gamma_ramp_exponent = 2

    def forward(
            self,
            signals: torch.Tensor,
            a: torch.Tensor,
            b: torch.Tensor):

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


class SplattingEventGenerator(nn.Module, EventGenerator):

    def __init__(
            self,
            n_samples: int,
            samplerate: int,
            n_resonance_octaves: int,
            n_frames: int,
            hard_reverb_choice: bool = False,
            hierarchical_scheduler: bool = False,
            wavetable_resonance: bool = False):

        super().__init__()
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.n_resonance_octaves = n_resonance_octaves
        self.n_octaves = n_resonance_octaves
        self.wavetable_resonance = wavetable_resonance

        if wavetable_resonance:
            self.resonance_generator = Resonance(
                4096, n_samples, hard_choice=False, samplerate=samplerate)
        else:
            self.resonance_generator = F0Resonance(
                n_resonance_octaves, n_samples, min_hz=20, max_hz=3000, samplerate=samplerate)

        self.hierarchical_scheduler = hierarchical_scheduler

        self.noise_generator = BandPassFilteredNoise(n_samples)
        self.amp_envelope_generator = ExponentialDecayEnvelope(
            base_resonance=0.02,
            n_frames=n_frames,
            n_samples=n_samples)
        self.evolving_resonance = EvolvingFilteredResonance(
            base_crossfade_resonance=0.02,
            crossfade_frames=n_frames,
            n_samples=n_samples)

        self.env_and_position = EnvelopeAndPosition(
            n_samples=n_samples,
            envelope_type=EnvelopeType.Gaussian.value,
            gaussian_envelope_factor=0.5)

        self.mixer = Mixer()

        self.verb = ReverbGenerator(
            4,
            2,
            samplerate,
            n_samples,
            norm=nn.LayerNorm(4, ),
            hard_choice=hard_reverb_choice)

        if hierarchical_scheduler:
            self.scheduler = HierarchicalDiracModel(
                n_events=1, signal_size=self.n_samples)
        else:
            self.scheduler = DiracScheduler(
                n_events=1, start_size=self.n_samples // 256, n_samples=self.n_samples)

    def forward(self, *args, **kwargs):
        if self.wavetable_resonance:
            return self.forward_wavetable(*args, **kwargs)
        else:
            return self.forward_f0(*args, **kwargs)

    def forward_wavetable(self,
                          env: torch.Tensor,
                          mix: torch.Tensor,
                          decay_choice: torch.Tensor,
                          filter_decay: torch.Tensor,
                          resonance_choice: torch.Tensor,
                          noise_filter: torch.Tensor,
                          resonance_filter_1: torch.Tensor,
                          resonance_filter_2: torch.Tensor,
                          amp: torch.Tensor,
                          verb_params: torch.Tensor,
                          times: torch.Tensor) -> torch.Tensor:

        batch = env.shape[0]

        overall_mix = torch.softmax(mix, dim=-1)

        resonances = self.resonance_generator.forward(resonance_choice)

        filtered_noise = self.noise_generator.forward(
            noise_filter[:, :, 0],
            (torch.abs(noise_filter[:, :, 1]) + 1e-12))

        filtered_resonance, filt_res_2, filt_crossfade_stacked = self.evolving_resonance.forward(
            resonances=resonances,
            decays=filter_decay,
            start_filter_means=torch.zeros_like(resonance_filter_1[:, :, 0]),
            start_filter_stds=torch.abs(resonance_filter_1[:, :, 1]) + 1e-12,
            end_filter_means=torch.zeros_like(resonance_filter_2[:, :, 0]),
            end_filter_stds=torch.abs(resonance_filter_2[:, :, 1]) + 1e-12

        )

        decays = self.amp_envelope_generator.forward(decay_choice)

        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays

        positioned_noise = self.env_and_position.forward(
            signals=filtered_noise,
            a=env[:, :, 0],
            b=env[:, :, 1])

        res = fft_convolve(
            positioned_noise,
            decaying_resonance)

        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2

        )

        mixed = self.mixer.forward([res, res2], filt_crossfade_stacked)

        final = self.mixer.forward([positioned_noise, mixed], overall_mix[:, :, None, :])
        # assert final.shape == (1, n_atoms, exp.n_samples)

        final = final.view(batch, -1, self.n_samples)
        final = unit_norm(final, dim=-1)

        amps = torch.abs(amp)
        final = final * amps

        final = self.scheduler.schedule(times, final)

        # rm is a one-hot room choice
        # mx is a two-element, softmax distribution
        final = self.verb.forward(verb_params, final)

        return final

    def forward_f0(
            self,
            env: torch.Tensor,
            mix: torch.Tensor,
            decay: torch.Tensor,
            filter_decay: torch.Tensor,
            f0_choice: torch.Tensor,
            decay_choice: torch.Tensor,
            freq_spacing: torch.Tensor,
            noise_filter: torch.Tensor,
            resonance_filter_1: torch.Tensor,
            resonance_filter_2: torch.Tensor,
            amp: torch.Tensor,
            verb_params: torch.Tensor,
            times: torch.Tensor) -> torch.Tensor:

        batch = env.shape[0]

        overall_mix = torch.softmax(mix, dim=-1)

        resonances = self.resonance_generator.forward(
            f0_choice, decay, freq_spacing, sigmoid_decay=True)

        filtered_noise = self.noise_generator.forward(
            noise_filter[:, :, 0],
            (torch.abs(noise_filter[:, :, 1]) + 1e-12))

        filtered_resonance, filt_res_2, filt_crossfade_stacked = self.evolving_resonance.forward(
            resonances=resonances,
            decays=filter_decay,
            start_filter_means=torch.zeros_like(resonance_filter_1[:, :, 0]),
            start_filter_stds=torch.abs(resonance_filter_1[:, :, 1]) + 1e-12,
            end_filter_means=torch.zeros_like(resonance_filter_2[:, :, 0]),
            end_filter_stds=torch.abs(resonance_filter_2[:, :, 1]) + 1e-12
        )

        decays = self.amp_envelope_generator.forward(decay_choice)

        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays

        positioned_noise = self.env_and_position.forward(
            signals=filtered_noise,
            a=env[:, :, 0],
            b=env[:, :, 1])

        res = fft_convolve(
            positioned_noise,
            decaying_resonance)

        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2
        )

        mixed = self.mixer.forward([res, res2], filt_crossfade_stacked)

        final = self.mixer.forward([positioned_noise, mixed], overall_mix[:, :, None, :])
        # assert final.shape == (1, n_atoms, exp.n_samples)

        final = final.view(batch, -1, self.n_samples)
        final = unit_norm(final, dim=-1)

        amps = torch.abs(amp)
        final = final * amps

        final = self.scheduler.schedule(times, final)

        # rm is a one-hot room choice
        # mx is a two-element, softmax distribution
        final = self.verb.forward(verb_params, final)

        return final

    @property
    def shape_spec(self) -> ShapeSpec:
        if not self.wavetable_resonance:
            return dict(
                env=(2,),
                mix=(2,),
                decay=(1,),
                filter_decay=(1,),
                f0_choice=(1,),
                decay_choice=(1,),
                freq_spacing=(1,),
                noise_filter=(2,),
                resonance_filter_1=(2,),
                resonance_filter_2=(2,),
                amp=(1,),
                verb_params=(4,),
            )
        else:
            return dict(
                env=(2,),
                mix=(2,),
                # decay=(1,),
                filter_decay=(1,),
                # f0_choice=(1,),
                decay_choice=(1,),
                # freq_spacing=(1,),
                resonance_choice=(4096,),
                noise_filter=(2,),
                resonance_filter_1=(2,),
                resonance_filter_2=(2,),
                amp=(1,),
                verb_params=(4,),
            )
