from conjure import SupportedContentType, numpy_conjure
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve, fft_shift
from modules.normal_pdf import pdf2
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax
from modules.transfer import make_waves
from scratchpad.time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.nn import functional as F


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_atoms = 128

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
    
    # return dict(normal=stft(x, 2048, 256, pad=True))
    



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



def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
    
    target = transform(target).reshape(target.shape[0], -1)
    
    # full = torch.sum(recon, dim=1, keepdim=True)
    # full = transform(full).view(*target.shape)
    
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


class Model(nn.Module):
    """
    A model representing audio with the following parameters
    
    n_atoms * (env(2) + mix(2) + decay(1) + decay(1) + res_choice(1) + noise_filter(2) + res_filter(2) + res_filter2(2) + amps(1) + verb_choice(1) + verb_mix(1))
    
    n_atoms * 16
    """
    
    def __init__(self):
        super().__init__()
        
        # means and stds for envelope
        self.env = nn.Parameter((torch.zeros(1, n_atoms, 2).uniform_(0, 1)))
        
        self.shifts = nn.Parameter(torch.zeros(1, n_atoms, exp.n_samples).uniform_(0, 1))
        
        # two-channel mixer for noise + resonance
        self.mix = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 1))
        
        self.decays = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(-6, 6))
        self.filter_decays = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(-6, 6))
        
        total_resonances = 4096
        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=total_resonances // 4)
        waves = make_waves(exp.n_samples, f0s, exp.samplerate)
        self.register_buffer('waves', waves.view(1, total_resonances, exp.n_samples))
        
        # one-hot choice of resonance for each atom
        self.resonance_choice = nn.Parameter(torch.zeros(1, n_atoms, total_resonances).uniform_(0, 1))
        
        # means and stds for bandpass noise filter
        self.noise_filter = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 1))
        
        # means and stds for bandpass resonance filter
        self.resonance_filter = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 1))
        self.resonance_filter2 = nn.Parameter(torch.zeros(1, n_atoms, 2).uniform_(0, 1))
        
        # amplitudes
        self.amplitudes = nn.Parameter(torch.zeros(1, n_atoms, 1).uniform_(0, 0.01))
        
        self.verb_params = nn.Parameter(torch.zeros(1, n_atoms, 4).uniform_(-1, 1))
        
        self.verb = ReverbGenerator(4, 2, exp.samplerate, exp.n_samples, norm=nn.LayerNorm(4,), hard_choice=True)
    
    def forward(self, x):
        overall_mix = torch.softmax(self.mix, dim=-1)
        
        resonances = sparse_softmax(self.resonance_choice, normalize=True, dim=-1)
        
        resonances = resonances @ self.waves
        assert resonances.shape == (1, n_atoms, exp.n_samples)
        
        
        noise = torch.zeros(1, n_atoms, exp.n_samples, device=resonances.device).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(noise, dim=-1)
        assert noise_spec.shape == (1, n_atoms, exp.n_samples // 2 + 1)
        
        noise_spec_filter = pdf2(
            self.noise_filter[:, :, 0], 
            (torch.abs(self.noise_filter[:, :, 1]) + 1e-12),
            noise_spec.shape[-1]
        )
        
        assert noise_spec_filter.shape == (1, n_atoms, exp.n_samples // 2 + 1)
        
        filtered_noise = noise_spec * noise_spec_filter
        filtered_noise = torch.fft.irfft(filtered_noise)
        assert filtered_noise.shape == (1, n_atoms, exp.n_samples)
        
        # resonance 1
        resonance_spec = torch.fft.rfft(resonances, dim=-1)
        resonance_filter = pdf2(
            # note: always low-pass
            torch.zeros_like(self.resonance_filter[:, :, 0]),
            torch.abs(self.resonance_filter[:, :, 1]) + 1e-12,
            resonance_spec.shape[-1]
        )
        
        filtered_resonance = resonance_spec * resonance_filter
        filtered_resonance = torch.fft.irfft(filtered_resonance)
        
        
        # resonance 2
        res_filter_2 = pdf2(
            # note: always low-pass
            torch.zeros_like(self.resonance_filter2[:, :, 0]),
            torch.abs(self.resonance_filter2[:, :, 1]) + 1e-12,
            resonance_spec.shape[-1]
        )
        filt_res_2 = resonance_spec * res_filter_2
        filt_res_2 = torch.fft.irfft(filt_res_2)
        assert filt_res_2.shape == filtered_resonance.shape
        
        # filter crossfade
        filt_crossfade = exponential_decay(self.filter_decays, n_atoms, 128, 0.02, exp.n_samples)
        filt_crossfade_inverse = 1 - filt_crossfade
        filt_crossfade_stacked = torch.cat([filt_crossfade[..., None], filt_crossfade_inverse[..., None]], dim=-1)
        assert filt_crossfade_stacked.shape == (1, n_atoms, exp.n_samples, 2)
        
        
        
        decays = exponential_decay(self.decays, n_atoms, 128, 0.02, exp.n_samples)
        decaying_resonance = filtered_resonance * decays
        decaying_resonance2 = filt_res_2 * decays
        
        envelopes = pdf2(
            self.env[:, :, 0], 
            torch.abs(self.env[:, :, 1] + 1e-12) * 0.1, 
            exp.n_samples).view(1, n_atoms, exp.n_samples)
        
        positioned_noise = filtered_noise * envelopes
        
        shifts = sparse_softmax(self.shifts, dim=-1, normalize=True)
        positioned_noise = fft_convolve(positioned_noise, shifts)
        
        
        res = fft_convolve(
            positioned_noise, 
            decaying_resonance)
        
        res2 = fft_convolve(
            positioned_noise,
            decaying_resonance2
        )
        stacked = torch.cat([res[..., None], res2[..., None]], dim=-1)
        mixed = torch.sum(filt_crossfade_stacked * stacked, dim=-1)
        
        
        stacked = torch.cat([
            positioned_noise[..., None], 
            mixed[..., None]], dim=-1)
        
        # # TODO: This is a dot product
        final = torch.sum(stacked * overall_mix[:, :, None, :], dim=-1)
        assert final.shape == (1, n_atoms, exp.n_samples)
        
        final = final.view(1, n_atoms, exp.n_samples)
        final = unit_norm(final, dim=-1)
        
        amps = torch.abs(self.amplitudes)
        final = final * amps
        # final = torch.sum(final, dim=1, keepdim=True)
        
        final = self.verb.forward(self.verb_params, final)
        
        return final, amps
        

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon, amps = model.forward(None)
    
    # hinge loss to encourage a sparse solution
    mask = amps > 1e-6
    sparsity = torch.abs(amps * mask).sum() * 1e-3
    
    nz = mask.sum() / amps.nelement()
    print(f'{nz} percent sparsity')
    
    # loss = exp.perceptual_loss(torch.sum(recon, dim=1, keepdim=True), batch) + sparsity
    
    # Does not work well
    # real = transform(batch)
    # fake = transform(torch.sum(recon, dim=1, keepdim=True))
    # loss = F.mse_loss(fake, real) + sparsity
    
    loss = single_channel_loss_3(batch, recon) + sparsity
    
    loss.backward()
    optim.step()
    
    recon = max_norm(recon.sum(dim=1, keepdim=True), dim=-1)
    return loss, recon, amps


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def amps(x: torch.Tensor):
        x = x.data.cpu().numpy()[0].reshape(1, n_atoms)
        return x

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