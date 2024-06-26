import os
from typing import Callable, List, Optional, Union
from conjure import SupportedContentType, numpy_conjure
import torch
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve, fft_shift
from modules.normal_pdf import gamma_pdf, pdf2
from modules.normalization import max_norm, unit_norm
# from modules.quantize import QuantizedResonanceMixture
from modules.quantize import QuantizedResonanceMixture
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax
from modules.transfer import gaussian_bandpass_filtered, make_waves
from scratchpad.time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.nn import functional as F
from enum import Enum
import numpy as np


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

class LossType(Enum):
    PhaseInvariantFeature = 'PIF'
    IterativeMultiband = 'IMB'
    AllAtOnceMultiband = 'AAOMB'
    Hybrid = 'HYBRID'
    
class EnvelopeType(Enum):
    Gaussian = 'Gaussian'
    Gamma = 'Gamma'
    

n_atoms = 64
envelope_dist = EnvelopeType.Gamma

# force_pos_adjustment = False
# For gamma distributions, the center of gravity is always near zero,
# so further adjustment is required
# softmax_positioning = envelope_dist == EnvelopeType.Gamma or force_pos_adjustment
softmax_positioning = True # locked
use_unit_shifts = False # locked


# hard_resonance_choice = False
loss_type = LossType.Hybrid.value # locked
# For iterative multiband loss, determine if channels are first sorted by descending norm
sort_by_norm = True # locked

optimize_f0 = True # locked
nyquist_cutoff = True # locked
fixed_f0_spacing = False
n_resonance_octaves = 128


static_learning_rate = 1e-3
total_iterations = 4000
schedule_learning_rate = True
learning_rates = torch.linspace(1e-2, 1e-4, steps=total_iterations)

gaussian_envelope_factor = 0.1

gsm = lambda x: F.gumbel_softmax(x, tau=0.1, hard=True, dim=-1)
hsm = lambda x: sparse_softmax(x, normalize=True, dim=-1)
sm = lambda x: torch.softmax(x, dim=-1)
sparse_choice = hsm

# hard_reverb_choice = lambda x: sparse_softmax(x, normalize=True, dim=-1) # locked
# hard_shift_choice = lambda x: sparse_softmax(x, normalize=True, dim=-1) # locked
# hard_resonance_choice = lambda x: sparse_softmax(x, normalize=True, dim=-1) # locked

hard_reverb_choice = sparse_choice
hard_shift_choice = sparse_choice
hard_resonance_choice = sm




def transform(x: torch.Tensor):
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)

        
def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_xl': stft(v, 512, 64, pad=True) for k, v in bands.items()}
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)
    return dict(
        **d1, 
        **d3, 
        **d4, 
        normal=normal
    )


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
        noise = torch.zeros(1, n_atoms, self.n_samples, device=means.device).uniform_(-1, 1)
        
        filtered_noise = gaussian_bandpass_filtered(means, stds, noise)
        assert filtered_noise.shape == (batch, n_events, self.n_samples)
        return filtered_noise
        

class F0Resonance(nn.Module):
    """
    TODO: Try the following:
    
    - wavetable with fft_shift
    - wavetable with gaussian sampling kernel
    """
    def __init__(self, n_f0_elements: int, n_octaves: int, n_samples: int, min_hz: int = 20, max_hz: int = 3000):
        super().__init__()
        self.min_hz = min_hz
        self.max_hz = max_hz
        factors = torch.arange(1, 1 + n_octaves, 1)
        self.n_octaves = n_octaves
        self.n_samples = n_samples
        self.n_f0_elements = n_f0_elements
        
        self.register_buffer('powers', 1 / 2 ** (torch.arange(1, 1 + n_f0_elements, 1)))
        self.register_buffer('factors', factors)
        
        self.min_freq = self.min_hz / (exp.samplerate // 2)
        self.max_freq = self.max_hz / (exp.samplerate // 2)
        self.freq_range = self.max_freq - self.min_freq
    
    def forward(
            self, 
            f0: torch.Tensor, 
            decay_coefficients: torch.Tensor, 
            _: torch.Tensor,
            freq_spacing: torch.Tensor,
            sigmoid_decay: bool = True):
        
        batch, n_events, n_elements = f0.shape
        
        # assert n_elements == self.n_f0_elements
        
        f0 = torch.abs(
            f0 #@ self.powers
        )
        f0 = f0.view(batch, n_events, 1)
        
        print(decay_coefficients.shape, f0.shape)
        
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
        
        if fixed_f0_spacing:
            factors.fill_(1)
        
        factors = torch.cumsum(factors, dim=-1)
        f0s = f0 * factors
        assert f0s.shape == (batch, n_events, self.n_octaves)
        
        if nyquist_cutoff:
            # filter out anything above nyquist
            mask = f0s < 1
            f0s = f0s * mask
        
        # generate sine waves
        f0s = f0s.view(batch, n_events, self.n_octaves, 1).repeat(1, 1, 1, self.n_samples)
        osc = torch.sin(
            torch.cumsum(f0s, dim=-1) # + phase_offsets[..., None]
        )
        
        assert osc.shape == (batch, n_events, self.n_octaves, self.n_samples)
        
        # apply decaying value
        osc = osc * exp_decays[..., None]
        osc = torch.sum(osc, dim=2)
        osc = max_norm(osc, dim=-1)
        
        assert osc.shape == (batch, n_events, self.n_samples)
        return osc
        

class Resonance(nn.Module):
    def __init__(self, n_resonances: int, n_samples: int, hard_choice: Union[bool, Callable]):
        super().__init__()
        self.n_resonances = n_resonances
        self.n_samples = n_samples
        self.hard_choice = hard_choice
        
        
        # we generate pure sines, triangle, sawtooth and square waves with these fundamental
        # frequencies, hence `n_resonances // 4`
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=n_resonances // 4)
        waves = make_waves(exp.n_samples, f0s, exp.samplerate)
        self.register_buffer('waves', waves.view(1, n_resonances, exp.n_samples))
    
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
    def __init__(self, n_samples: int, envelope_type: EnvelopeType):
        super().__init__()
        self.n_samples = n_samples
        self.envelope_type = envelope_type
        
        self.gamma_ramp_size = 128
        self.gamma_ramp_exponent = 2
    
    def forward(
            self, 
            signals: torch.Tensor, 
            a: torch.Tensor, 
            b: torch.Tensor, 
            adjustment: Union[torch.Tensor, None],
            unit_shifts: Union[torch.Tensor, None]):
        
        batch, n_events, n_samples = signals.shape
        assert n_samples == self.n_samples
        
        batch, n_events = a.shape
        assert a.shape == b.shape
        
        if self.envelope_type == EnvelopeType.Gaussian.value:
            envelopes = pdf2(a, (torch.abs(b) + 1e-12) * gaussian_envelope_factor, self.n_samples)
        elif self.envelope_type == EnvelopeType.Gamma.value:
            envelopes = gamma_pdf((torch.abs(a) + 1e-12), (torch.abs(b) + 1e-12), self.n_samples)
            ramp = torch.zeros_like(envelopes)
            ramp[..., :self.gamma_ramp_size] = torch.linspace(0, 1, self.gamma_ramp_size)[None, None, :] ** self.gamma_ramp_exponent
            envelopes = envelopes * ramp
        else:
            raise ValueError(f'{self.envelope_type.value} is not supported')
        
        assert envelopes.shape == (batch, n_events, self.n_samples)
        
        positioned_signals = signals * envelopes
        
        if adjustment is not None:
            # shifts = sparse_softmax(adjustment, dim=-1, normalize=True)
            shifts = hard_shift_choice(adjustment)
            positioned_signals = fft_convolve(positioned_signals, shifts)
        
        if unit_shifts is not None:
            positioned_signals = fft_shift(positioned_signals, unit_shifts)
        
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
    

def single_channel_loss_3(
        target: torch.Tensor, 
        recon: torch.Tensor, 
        sort_by_norm: bool = True,
        coarse_loss: bool = False) -> torch.Tensor:
    
    
    target = transform(target).reshape(target.shape[0], -1)
    
    
    channels = transform(recon)
    
    residual = target
    
    # Try L1 norm instead of L@
    # Try choosing based on loudest patch/segment
    
    if sort_by_norm:
        # sort channels from loudest to softest
        diff = torch.norm(channels, dim=(-1), p = 1)
        indices = torch.argsort(diff, dim=-1, descending=True)
        srt = torch.take_along_dim(channels, indices[:, :, None], dim=1)
    else:
        srt = channels
    
    
    loss = 0
    
    if coarse_loss:
        summed = torch.sum(channels, dim=1, keepdim=True)
        loss = loss + F.mse_loss(summed.view(*target.shape), target)
    
    for i in range(n_atoms):
        current = srt[:, i, :]
        start_norm = torch.norm(residual, dim=-1, p=1)
        # TODO: should the residual be cloned and detached each time,
        # so channels are optimized independently?
        residual = residual - current
        end_norm = torch.norm(residual, dim=-1, p=1)
        diff = -(start_norm - end_norm)
        loss = loss + diff.sum()
        
    
    return loss


class Model(nn.Module):
    """
    A model representing audio with the following parameters
    
    n_atoms * (env(2) + mix(2) + decay(1) + decay(1) + res_choice(1) + noise_filter(2) + res_filter(2) + res_filter2(2) + amps(1) + verb_choice(1) + verb_mix(1))
    
    n_atoms * 16
    """
    
    def __init__(self, n_resonance_octaves=64):
        super().__init__()
        
        self.n_resonance_octaves = n_resonance_octaves
        
        # means and stds for envelope
        self.env = nn.Parameter((torch.zeros(1, n_atoms, 2).uniform_(1e-8, 1)))
        
        self.shifts = nn.Parameter(torch.zeros(1, n_atoms, exp.n_samples).uniform_(0, 1))
        
        # two-channel mixer for noise + resonance
        self.mix = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 1))
        
        self.decays = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(-6, 6))
        self.filter_decays = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(-6, 6))
        
        if use_unit_shifts:
            self.unit_shifts = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(0, 1))
        
        if optimize_f0:
            n_f0_elements = 16
            n_octaves = self.n_resonance_octaves
            self.resonance_generator = F0Resonance(
                n_f0_elements=n_f0_elements, n_octaves=n_octaves, n_samples=exp.n_samples)
            
            self.f0_choice = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(0.001, 0.1))
            self.decay_choice = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(-6, -6))
            self.phase_offsets = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(-6, -6))
            self.freq_spacing = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(0.5, 2))
        else:
            total_resonances = 4096
            # hard_resonance_choice = True
            
            quantize_dim = n_atoms
            # hard_func = lambda x: sparse_softmax(x, normalize=True, dim=-1)
            # hard_func = lambda x: x
            
            # self.resonance_generator = Resonance(
            #     n_resonances=total_resonances, 
            #     n_samples=exp.n_samples, 
            #     hard_choice=hard_resonance_choice)
            
            self.resonance_generator = QuantizedResonanceMixture(
                n_resonances=total_resonances,
                quantize_dim=quantize_dim,
                n_samples=exp.n_samples,
                samplerate=exp.samplerate,
                hard_func=lambda x: sparse_softmax(x, normalize=True, dim=-1)
            )
            # one-hot choice of resonance for each atom
            self.resonance_choice = nn.Parameter(torch.zeros(1, n_atoms, total_resonances).uniform_(0, 1))
        
        self.noise_generator = BandPassFilteredNoise(exp.n_samples)
        self.amp_envelope_generator = ExponentialDecayEnvelope(
            base_resonance=0.02, 
            n_frames=128, 
            n_samples=exp.n_samples)
        self.evolving_resonance = EvolvingFilteredResonance(
            base_crossfade_resonance=0.02, 
            crossfade_frames=128, 
            n_samples=exp.n_samples)
        
        self.env_and_position = EnvelopeAndPosition(
            n_samples=exp.n_samples, 
            envelope_type=EnvelopeType.Gaussian.value)
        
        self.mixer = Mixer()
        
        
        # means and stds for bandpass noise filter
        self.noise_filter = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 1))
        
        # means and stds for bandpass resonance filter
        self.resonance_filter = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 0.1))
        self.resonance_filter2 = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 0.1))
        
        # amplitudes
        self.amplitudes = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(0, 0.1) ** 2)
        
        self.verb_params = nn.Parameter(torch.zeros(1, n_atoms, 4).uniform_(-1, 1))
        
        # reverb parameters
        self.rm: Optional[torch.Tensor] = None
        self.mx: Optional[torch.Tensor] = None
        
        self.verb = ReverbGenerator(
            4, 2, exp.samplerate, exp.n_samples, norm=nn.LayerNorm(4,), hard_choice=hard_reverb_choice)
    
    def get_parameters(self) -> torch.Tensor:
        
        
        if self.rm is None or self.mx is None:
            standin = torch.zeros((1, n_atoms, exp.n_samples), device=self.amplitudes.device)
            _, rm, mx = self.verb.forward(self.verb_params, standin, return_parameters=True)
            self.rm = rm
            self.mx = mx
            
        
        atoms = []
        for i in range(n_atoms):
            new_atom = [
                
                # mean and std for the envelope
                self.env[0, i, 0].item(),
                self.env[0, i, 1].item(),
                
                # unit value for shift
                float(torch.argmax(self.shifts[0, i], dim=-1).item() / self.shifts.shape[-1]),
                
                # since the mix is two-elements and passed through softmax, the other element
                # can be derived
                self.mix[0, i, 0].item(),
                
                # decay value for this atom
                self.decays[0, i, 0].item(),
                
                # decay value that determines how we crossfade from one filter
                # to another
                self.filter_decays[0, i, 0].item(),
                
                
                self.f0_choice[0, i, 0].item(),
                self.decay_choice[0, i, 0].item(),
                self.freq_spacing[0, i, 0].item(),
                
                # unit value for resonance choice
                # float(torch.argmax(self.resonance_choice[0, i], dim=-1).item() / self.resonance_choice.shape[-1]),
                
                # mean for noise filter
                self.noise_filter[0, i, 0].item(),
                # std for noise filter
                self.noise_filter[0, i, 1].item(),
                
                # mean for resonance_filter
                self.resonance_filter[0, i, 0].item(),
                # std for resonance filter
                self.resonance_filter[0, i, 1].item(),
                
                # atom amplitude
                self.amplitudes[0, i, 0].item(),
                
                # unit value for reverb choice
                float(torch.argmax(self.rm[i], dim=-1).item() / self.verb.n_rooms),
                
                # since the reverb mix is two elements passed through a softmax,
                # the other value can be derived
                self.mx[0, i, 0, 0].item()
                
            ]
            new_atom = np.array(new_atom)
            new_atom = torch.from_numpy(new_atom)
            atoms.append(new_atom[None, ...])
        
        atoms = torch.cat(atoms, dim=0)
        return atoms
    
    def forward(self, x, return_unpositioned_atoms: bool = False):
        overall_mix = torch.softmax(self.mix, dim=-1)
        
        
        if optimize_f0:
            resonances = self.resonance_generator.forward(
                self.f0_choice, self.decay_choice, self.phase_offsets, self.freq_spacing)
        else:
            resonances = self.resonance_generator.forward(self.resonance_choice)
        
        
        filtered_noise = self.noise_generator.forward(
            self.noise_filter[:, :, 0], 
            (torch.abs(self.noise_filter[:, :, 1]) + 1e-12))
        
        
        filtered_resonance, filt_res_2, filt_crossfade_stacked = self.evolving_resonance.forward(
            resonances=resonances,
            decays=self.filter_decays,
            start_filter_means=torch.zeros_like(self.resonance_filter[:, :, 0]),
            start_filter_stds=torch.abs(self.resonance_filter[:, :, 1]) + 1e-12,
            end_filter_means=torch.zeros_like(self.resonance_filter2[:, :, 0]),
            end_filter_stds=torch.abs(self.resonance_filter2[:, :, 1]) + 1e-12
        )
        
        
        decays = self.amp_envelope_generator.forward(self.decays)
        
        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays
        

        if return_unpositioned_atoms:
            positioned_noise = self.env_and_position.forward(
                signals=filtered_noise, 
                a=self.env[:, :, 0], 
                b=self.env[:, :, 1], 
                adjustment=None,
                unit_shifts=None)
        else:
            positioned_noise = self.env_and_position.forward(
                signals=filtered_noise, 
                a=self.env[:, :, 0], 
                b=self.env[:, :, 1], 
                adjustment=self.shifts if softmax_positioning else None,
                unit_shifts=self.unit_shifts if use_unit_shifts else None)        
            
        res = fft_convolve(
            positioned_noise, 
            decaying_resonance)
        
        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2
        )
        # stacked = torch.cat([res[..., None], res2[..., None]], dim=-1)
        # mixed = torch.sum(filt_crossfade_stacked * stacked, dim=-1)
        
        mixed = self.mixer.forward([res, res2], filt_crossfade_stacked)
        
        
        # stacked = torch.cat([
        #     positioned_noise[..., None], 
        #     mixed[..., None]], dim=-1)
        
        # # TODO: This is a dot product
        # final = torch.sum(stacked * overall_mix[:, :, None, :], dim=-1)
        
        final = self.mixer.forward([positioned_noise, mixed], overall_mix[:, :, None, :])
        assert final.shape == (1, n_atoms, exp.n_samples)
        
        final = final.view(1, n_atoms, exp.n_samples)
        final = unit_norm(final, dim=-1)
        
        amps = torch.abs(self.amplitudes)
        final = final * amps
        
        # rm is a one-hot room choice
        # mx is a two-element, softmax distribution
        final, rm, mx = self.verb.forward(self.verb_params, final, return_parameters=True)
        
        self.rm = rm
        self.mx = mx
        
        return final, amps
        

model = Model(n_resonance_octaves=n_resonance_octaves).to(device)
optim = optimizer(model, lr=static_learning_rate)

def perceptual_loss(recon: torch.Tensor, orig: torch.Tensor):
    loss = exp.perceptual_loss(torch.sum(recon, dim=1, keepdim=True), orig) #+ sparsity
    return loss

def multiband_loss(recon: torch.Tensor, orig: torch.Tensor):
    real = transform(orig)
    fake = transform(torch.sum(recon, dim=1, keepdim=True))
    loss = F.mse_loss(fake, real) #+ sparsity
    return loss

scaler = torch.cuda.amp.GradScaler()

def train(batch, i):
    optim.zero_grad()
    recon, amps = model.forward(None)
    
    # hinge loss to encourage a sparse solution
    mask = amps > 1e-6
    sparsity = torch.abs(amps * mask).sum() * 0.1
    
    nz = mask.sum() / amps.nelement()
    print(f'{nz} percent sparsity with min {amps.min().item()} and max {amps.max().item()}')
    
    if loss_type == LossType.PhaseInvariantFeature.value:
        loss = exp.perceptual_loss(torch.sum(recon, dim=1, keepdim=True), batch) #+ sparsity
    elif loss_type == LossType.AllAtOnceMultiband.value:
        with torch.cuda.amp.autocast():
            real = transform(batch)
            fake = transform(torch.sum(recon, dim=1, keepdim=True))
            loss = F.mse_loss(fake, real) #+ sparsity
    elif loss_type == LossType.IterativeMultiband.value:
        loss = single_channel_loss_3(batch, recon, sort_by_norm=sort_by_norm) + sparsity
    elif loss_type == LossType.Hybrid.value:
        loss = single_channel_loss_3(batch, recon, sort_by_norm=sort_by_norm, coarse_loss=True) + sparsity
    else:
        raise ValueError(f'Unsupported loss {loss_type}')
    
    if schedule_learning_rate:
        try:
            new_learning_rate = learning_rates[i]
            print(f'new learning rate is {new_learning_rate.item()}')
            for g in optim.param_groups:
                g['lr'] = new_learning_rate
        except IndexError:
            pass
    
    scaler.scale(loss).backward()
    scaler.step(optim)
    scaler.update()
    # loss.backward()
    # optim.step()
    
    recon = max_norm(recon.sum(dim=1, keepdim=True), dim=-1)
    return loss, recon, amps


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def amps(x: torch.Tensor):
        x = x.data.cpu().numpy()[0].reshape(1, n_atoms)
        return x / (x.max() + 1e-8)

    return (amps,)


@readme
class GaussianSplatting(BaseExperimentRunner):
    amps = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, a = train(item, i)

            self.real = item
            self.fake = r
            self.amps = a
            
            
            print(i, l.item())
            self.after_training_iteration(l, i)
            
            # if i % 100 == 0:
            #     print('SAVING!')
            #     path = os.path.join(self.trained_weights_path, 'splat_4.dat')
            #     torch.save(model.state_dict(), path)
            
            if i == total_iterations:
                break