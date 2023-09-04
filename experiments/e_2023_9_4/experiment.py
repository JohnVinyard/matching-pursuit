
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.atoms import unit_norm
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.reverb import ReverbGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        x = F.dropout(x, 0.1)

        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x

class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=2, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 2)
        x = torch.softmax(x, dim=-1)
        return x

class GenerateImpulse(nn.Module):

    def __init__(self, latent_dim, channels, n_samples, n_filter_bands, encoding_channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_samples = n_samples
        self.n_frames = n_samples // 256
        self.n_filter_bands = n_filter_bands
        self.channels = channels
        self.filter_kernel_size = 32
        self.encoding_channels = encoding_channels


        scale = zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist), n_filter_bands)
        filters = morlet_filter_bank(
            exp.samplerate, self.filter_kernel_size, scale, 0.25, normalize=True).real.astype(np.float32)
        self.register_buffer('filters', torch.from_numpy(filters).view(n_filter_bands, self.filter_kernel_size))

        self.to_filter_mix = LinearOutputStack(
            channels, 3, out_channels=self.n_filter_bands, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        
        self.to_envelope = ConvUpsample(
            latent_dim,
            channels,
            start_size=8,
            mode='nearest',
            end_size=self.n_frames,
            out_channels=1,
            batch_norm=True,
        )

    
    def forward(self, x):

        batch_size = x.shape[0]

        # generate envelopes
        frames = self.to_envelope(x.view(-1, self.latent_dim))
        frames = torch.abs(frames)
        frames = F.interpolate(frames, size=self.n_samples, mode='linear')
        frames = frames.view(-1, self.encoding_channels, self.n_samples)

        # generate filters
        mix = self.to_filter_mix(x).view(-1, self.n_filter_bands, 1)


        filt = self.filters.view(-1, self.n_filter_bands, self.filter_kernel_size)
        filters = (mix * filt).view(-1, self.n_filter_bands, self.filter_kernel_size).sum(dim=1)
        filters = F.pad(filters, (0, self.n_samples - self.filter_kernel_size))
        filters = filters.view(-1, self.encoding_channels, self.n_samples)


        # generate noise
        noise = torch.zeros(batch_size, self.encoding_channels, self.n_samples, device=x.device).uniform_(-1, 1)

        # filter the noise
        filtered = fft_convolve(noise, filters)[..., :self.n_samples]

        # apply envelope
        impulse = filtered * frames
        return impulse


class Model(nn.Module):
    def __init__(
            self, 
            channels, 
            encoding_channels, 
            impulse_samples, 
            resonance_samples,
            latent_dim):

        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.impulse_samples = impulse_samples
        self.resonance_samples = resonance_samples
        self.latent_dim = latent_dim

        self.stack = nn.Sequential(
            nn.Conv1d(exp.n_bands, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),
            DilatedBlock(channels, 1),    
            DilatedBlock(channels, 3),    
            DilatedBlock(channels, 9),    
            DilatedBlock(channels, 27),    
            DilatedBlock(channels, 81),    
            DilatedBlock(channels, 243),    
            DilatedBlock(channels, 1),    
        )

        self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)

        self.impulse_latent = nn.Parameter(
            torch.zeros(1, encoding_channels, latent_dim).uniform_(-1, 1))
        self.resonance_latent = nn.Parameter(
            torch.zeros(1, encoding_channels, latent_dim).uniform_(-1, 1))
        self.mix_latent = nn.Parameter(
            torch.zeros(1, encoding_channels, latent_dim).uniform_(-1, 1))

        self.embed_impulse_latent = LinearOutputStack(
             channels, 
             layers=3, 
             in_channels=encoding_channels, 
             out_channels=latent_dim, 
             norm=nn.LayerNorm((channels,))
        )

        self.embed_resonance_latent = LinearOutputStack(
             channels, 
             layers=3, 
             in_channels=encoding_channels, 
             out_channels=latent_dim, 
             norm=nn.LayerNorm((channels,))
        )

        self.embed_mix_latent = LinearOutputStack(
             channels, 
             layers=3, 
             in_channels=encoding_channels, 
             out_channels=latent_dim, 
             norm=nn.LayerNorm((channels,))
        )


        self.to_resonance = ConvUpsample(
            latent_dim, 
            channels, 
            8, 
            end_size=resonance_samples, 
            out_channels=1, 
            mode='nearest', 
            batch_norm=True)
        
        self.to_impulse = GenerateImpulse(
            latent_dim, channels, self.impulse_samples, 16, encoding_channels)
        
        self.to_mix = GenerateMix(
            latent_dim, channels, encoding_channels)

        self.apply(lambda x: exp.init_weights(x))
    
        
    def forward(self, x):
        batch_size = x.shape[0]

        x = exp.fb.forward(x, normalize=False)
        x = self.stack(x)

        x = self.up(x)
        x = F.dropout(x, 0.01)

        # TODO: lateral competition
        encoding = x = torch.relu(x)

        ctxt = torch.sum(encoding, dim=-1)

        impulse_latent = self.embed_impulse_latent(ctxt) + self.impulse_latent
        resonance_latent = self.embed_resonance_latent(ctxt) + self.resonance_latent
        mix_latent = self.embed_mix_latent(ctxt) + self.mix_latent

        assert impulse_latent.shape == (batch_size, self.encoding_channels, self.latent_dim)
        assert resonance_latent.shape == (batch_size, self.encoding_channels, self.latent_dim)
        assert mix_latent.shape == (batch_size, self.encoding_channels, self.latent_dim)

        impulses = self.to_impulse.forward(impulse_latent)
        assert impulses.shape == (batch_size, self.encoding_channels, self.impulse_samples)
        impulses = F.pad(impulses, (0, self.resonance_samples - self.impulse_samples))

        resonances = self.to_resonance.forward(resonance_latent).view(-1, self.encoding_channels, self.resonance_samples)
        assert resonances.shape == (batch_size, self.encoding_channels, self.resonance_samples)
        

        mix = self.to_mix(mix_latent)
        assert mix.shape == (batch_size, self.encoding_channels, 2)

        imp_amp = mix[..., :1]
        res_amp = mix[..., 1:]

        # scaled_impulses = impulses * imp_amp
        # scaled_resonances = resonances * res_amp

        d = fft_convolve(impulses, resonances)
        d = (impulses * imp_amp) + (d * res_amp)
        assert d.shape == (batch_size, self.encoding_channels, self.resonance_samples)

        d = F.pad(d, (0, exp.n_samples - self.resonance_samples))

        x = fft_convolve(d, encoding)[..., :exp.n_samples]
        x = torch.sum(x, dim=1, keepdim=True)
        return x, encoding

model = Model(
    channels=64, 
    encoding_channels=512, 
    impulse_samples=256, 
    resonance_samples=8192,
    latent_dim=16
).to(device)

optim = optimizer(model, lr=1e-3)

def train(batch, i):
    batch_size = batch.shape[0]

    optim.zero_grad()
    recon, encoding = model.forward(batch)

    encoding = encoding.view(batch_size, -1)
    non_zero = (encoding > 0).sum()
    sparsity = non_zero / encoding.nelement()
    print('sparsity', sparsity.item(), 'n_elements', (non_zero / batch_size).item())


    sparsity_loss = torch.abs(encoding).sum() * 0.0001

    loss = exp.perceptual_loss(recon, batch) + sparsity_loss

    loss.backward()
    optim.step()
    return loss, recon


@readme
class SparseResonance(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    