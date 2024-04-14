
from enum import Enum
from typing import Callable, List, Union
import torch
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules import stft
from modules.anticausal import AntiCausalStack
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve, fft_shift
from modules.linear import LinearOutputStack
from modules.normal_pdf import gamma_pdf, pdf2
from modules.normalization import max_norm, unit_norm
from modules.quantize import QuantizedResonanceMixture
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax
from modules.sparse import sparsify_vectors
from modules.transfer import gaussian_bandpass_filtered, make_waves
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_atoms = 64

class EnvelopeType(Enum):
    Gaussian = 'Gaussian'
    Gamma = 'Gamma'

envelope_dist = EnvelopeType.Gaussian
force_pos_adjustment = True
# For gamma distributions, the center of gravity is always near zero,
# so further adjustment is required
softmax_positioning = envelope_dist == EnvelopeType.Gamma or force_pos_adjustment
# max_guassian_shift = 256 / exp.n_samples

def experiment_spectrogram(x: torch.Tensor):
    batch_size = x.shape[0]
    
    x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
    return x


def gaussian_std_epsilon(min_width: int, total_width: int):
    epsilon = min_width / total_width
    return epsilon

def exponential_decay(
        decay_values: torch.Tensor, 
        n_atoms: int, 
        n_frames: int, 
        base_resonance: float,
        n_samples: int):
    
    decay_values = torch.sigmoid(decay_values.view(-1, n_atoms, 1).repeat(1, 1, n_frames))
    resonance_factor = (1 - base_resonance) * 0.95
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
        


        


class Resonance(nn.Module):
    def __init__(self, n_resonances: int, n_samples: int, hard_choice: bool):
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
            resonances = sparse_softmax(choice, normalize=True, dim=-1)
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
            gaussian_mean_offset: Union[torch.Tensor, None]):
        
        batch, n_events, n_samples = signals.shape
        assert n_samples == self.n_samples
        
        batch, n_events = a.shape
        assert a.shape == b.shape
        
        # print(a.shape, b.shape, gaussian_mean_offset.shape)
        
        
        epsilon = gaussian_std_epsilon(64, exp.n_samples)
        
        if self.envelope_type == EnvelopeType.Gaussian.value:
            
            
            # if gaussian_mean_offset is not None:
            #     envelopes = pdf2(a + gaussian_mean_offset.view(*a.shape), (torch.abs(b) + epsilon), self.n_samples)
            # else:
            envelopes = pdf2(torch.sigmoid(a) * 0.1, (torch.abs(b) + epsilon), self.n_samples)
                
        elif self.envelope_type == EnvelopeType.Gamma.value:
            envelopes = gamma_pdf((torch.abs(a) + epsilon), (torch.abs(b) + epsilon), self.n_samples)
            ramp = torch.zeros_like(envelopes)
            ramp[..., :self.gamma_ramp_size] = torch.linspace(0, 1, self.gamma_ramp_size)[None, None, :] ** self.gamma_ramp_exponent
            envelopes = envelopes * ramp
        else:
            raise ValueError(f'{self.envelope_type.value} is not supported')

        # print(envelopes.shape)
                
        assert envelopes.shape == (batch, n_events, self.n_samples)
        
        positioned_signals = signals * envelopes
        
        if adjustment is not None:
            shifts = sparse_softmax(adjustment, dim=-1, normalize=True)
            positioned_signals = fft_convolve(positioned_signals, shifts)
        
        if gaussian_mean_offset is not None:
            positioned_signals = fft_shift(positioned_signals, gaussian_mean_offset)[..., :exp.n_samples]
        
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


class Model(nn.Module):
    def __init__(self, channels: int, total_resonances: int, quantize_dim: int, hard_func: Callable):
        super().__init__()
        self.channels = channels
        self.to_channels = nn.Conv1d(1, self.channels, 25, 1, 12)
        self.encoder = AntiCausalStack(
            channels, 
            kernel_size=2, 
            dilations=[1, 2, 4, 8, 16, 32, 64, 1])
        self.attn = nn.Conv1d(self.channels, 1, 7, 1, 3)
        
        
        self.resonance_generator = QuantizedResonanceMixture(
                n_resonances=total_resonances,
                quantize_dim=quantize_dim,
                n_samples=exp.n_samples,
                samplerate=exp.samplerate,
                hard_func=hard_func
            )
        
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
        
        self.verb = ReverbGenerator(
            self.channels, 
            layers=2, 
            samplerate=exp.samplerate, 
            n_samples=exp.n_samples, 
            norm=nn.LayerNorm(self.channels,), 
            hard_choice=True)
        
        self.mixer = Mixer()
        
        self.generate_mix_params = LinearOutputStack(channels, 3, out_channels=2, norm=nn.LayerNorm((channels,)))
        self.generate_env = LinearOutputStack(channels, 3, out_channels=2, norm=nn.LayerNorm((channels,)))
        self.generate_resonance_choice = LinearOutputStack(channels, 3, out_channels=total_resonances, norm=nn.LayerNorm((channels,)))
        self.generate_noise_filter = LinearOutputStack(channels, 3, out_channels=2, norm=nn.LayerNorm((channels,)))
        self.generate_filter_decays = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
        self.generate_filter_1 = LinearOutputStack(channels, 3, out_channels=2, norm=nn.LayerNorm((channels,)))
        self.generate_filter_2 = LinearOutputStack(channels, 3, out_channels=2, norm=nn.LayerNorm((channels,)))
        self.generate_decays = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
        self.generate_amplitudes = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
        self.generate_verb_params = LinearOutputStack(channels, 3, out_channels=channels, norm=nn.LayerNorm((channels,)))
        
        
        # self.generate_mix_params = nn.Linear(channels, 2)
        # self.generate_env = nn.Linear(channels, 2)
        # self.generate_resonance_choice = nn.Linear(channels, total_resonances)
        # self.generate_noise_filter = nn.Linear(channels, 2)
        # self.generate_filter_decays = nn.Linear(channels, 1)
        # self.generate_filter_1 = nn.Linear(channels, 2)
        # self.generate_filter_2 = nn.Linear(channels, 2)
        # self.generate_decays = nn.Linear(channels, 1)
        # self.generate_amplitudes = nn.Linear(channels, 1)
        # self.generate_verb_params = nn.Linear(channels, channels)
        
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        
        x = x.view(-1, 1, exp.n_samples)
        
        spec = experiment_spectrogram(x)
        
        full_decoded = torch.zeros(batch_size, n_atoms, exp.n_samples, device=x.device)
        
        for i in range(n_atoms):
            
            print(f'batch {i} spec norm {spec.norm().item()}')
        
            x = self.encoder(spec)
        
            attn = self.attn(x)
        
            step_size = exp.n_samples // spec.shape[-1]
            shifts = torch.zeros(x.shape[0], 1, exp.n_samples, device=x.device)
        
            events, indices = sparsify_vectors(x, attn=attn, n_to_keep=1, normalize=False)
            event_norms = torch.norm(unit_norm(events), dim=-1, keepdim=True)
        
            for b in range(batch_size):
                for j in range(1):
                    norm = event_norms[b, j]
                    index = indices[b, j]
                    shifts[b, j, index * step_size] = norm
        
            # (batch, n_events, 2)
            mx = self.generate_mix_params.forward(events)
            
            # (batch, n_events, 2)
            env = self.generate_env(events)
            # (batch, n_events, 4096) (one-hot)
            res_choice = self.generate_resonance_choice.forward(events)
            # (batch, n_events, 2)
            noise_filt = self.generate_noise_filter(events)
            # (batch, n_events, 1)
            filt_decays = self.generate_filter_decays(events)
            # (batch, n_events, 2)
            res_filt_1 = self.generate_filter_1(events)
            # (batch, n_events, 2)
            res_filt_2 = self.generate_filter_2(events)
            # (batch, n_events, 1)
            decays = self.generate_decays.forward(events)
            # (batch, n_events, 1)
            amps = self.generate_amplitudes(events)
            
            # (batch, n_events, 16)
            verb = self.generate_verb_params(events)
            
            decoded, amps = self.decode(
                mix=mx, 
                env=env,
                resonance_choice=res_choice, 
                noise_filter=noise_filt, 
                filter_decays=filt_decays, 
                resonance_filter=res_filt_1, 
                resonance_filter2=res_filt_2, 
                decays=decays, 
                shifts=shifts, 
                amplitudes=amps, 
                verb_params=verb,
            )
            
            
            decoded_spec = experiment_spectrogram(decoded)
            spec = (spec - decoded_spec).clone().detach()
            full_decoded[:, i: i + 1, :] = decoded
            
        return full_decoded
    
    def decode(
            self, 
            mix: torch.Tensor,
            env: torch.Tensor,
            resonance_choice: torch.Tensor,
            noise_filter: torch.Tensor,
            filter_decays: torch.Tensor,
            resonance_filter: torch.Tensor,
            resonance_filter2: torch.Tensor,
            decays: torch.Tensor,
            shifts: torch.Tensor,
            amplitudes: torch.Tensor,
            verb_params: torch.Tensor):
        
        batch_size, n_events, _ = mix.shape
        
        
        
        overall_mix = torch.softmax(mix, dim=-1)
        
        
        resonances = self.resonance_generator.forward(resonance_choice)
        
        
        filtered_noise = self.noise_generator.forward(
            noise_filter[:, :, 0], 
            (torch.abs(noise_filter[:, :, 1]) + 1e-4))
        
        
        epsilon = gaussian_std_epsilon(10, exp.n_samples // 2 + 1)
        filtered_resonance, filt_res_2, filt_crossfade_stacked = self.evolving_resonance.forward(
            resonances=resonances,
            decays=filter_decays,
            start_filter_means=torch.zeros_like(resonance_filter[:, :, 0]),
            start_filter_stds=torch.abs(resonance_filter[:, :, 1]) + epsilon,
            end_filter_means=torch.zeros_like(resonance_filter2[:, :, 0]),
            end_filter_stds=torch.abs(resonance_filter2[:, :, 1]) + epsilon
        )
        
        
        decays = self.amp_envelope_generator.forward(decays)
        
        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays
        

        positioned_noise = self.env_and_position.forward(
            signals=filtered_noise, 
            a=env[:, :, 0], 
            b=env[:, :, 1], 
            adjustment=shifts,
            gaussian_mean_offset=None)        
        
        res = fft_convolve(
            positioned_noise, 
            decaying_resonance)
        
        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2
        )
        
        mixed = self.mixer.forward([res, res2], filt_crossfade_stacked)
        
        
        
        final = self.mixer.forward([positioned_noise, mixed], overall_mix[:, :, None, :])
        # assert final.shape == (batch_size, n_atoms, exp.n_samples)
        
        final = final.view(batch_size, n_events, exp.n_samples)
        final = unit_norm(final, dim=-1)
        
        amps = torch.abs(amplitudes)
        final = final * amps
        
        final = self.verb.forward(verb_params, final)
        
        return final, amps
    
    

model = Model(
    channels=1024, 
    total_resonances=4096, 
    quantize_dim=4096, 
    # hard_func=lambda x: sparse_softmax(x, normalize=True, dim=-1)
    hard_func=lambda x: torch.relu(x)
).to(device)

optim = optimizer(model, lr=1e-3)

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
    return dict(**d1, **d3, **d4, normal=normal)

def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
    
    target = transform(target).reshape(target.shape[0], -1)
    
    
    channels = transform(recon)
    
    residual = target
    
    # Try L1 norm instead of L@
    # Try choosing based on loudest patch/segment
    
    # sort channels from loudest to softest
    diff = torch.norm(channels, dim=(-1), p = 1)
    indices = torch.argsort(diff, dim=-1, descending=True)
    
    srt = torch.take_along_dim(channels, indices[:, :, None], dim=1)
    
    loss = 0
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

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    
    summed = torch.sum(recon, dim=1, keepdim=True)
    
    # real = transform(batch)
    # fake = transform(summed)
    # loss = torch.abs(real - fake).sum() / batch.shape[0]
    loss = single_channel_loss_3(batch, recon)
    loss.backward()
    optim.step()
    return loss, max_norm(summed, dim=-1)
    

@readme
class SplatEncoder(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    