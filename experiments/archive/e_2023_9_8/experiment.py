
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import unit_norm
from modules.overlap_add import overlap_add
from modules.reverb import ReverbGenerator
from modules.sparse import encourage_sparsity_loss
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


convolve_impulse_and_resonance = False
full_atom_size = exp.n_samples

class RecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, impulse_samples, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.impulse_samples = impulse_samples
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.step = window_size // 2
        self.n_coeffs = window_size // 2 + 1
        self.resonance_samples = resonance_samples
        self.n_frames = resonance_samples // self.step
        self.encoding_channels = encoding_channels

        self.base_resonance = 0.5
        self.resonance_factor = (1 - self.base_resonance) * 0.999

        self.register_buffer('group_delay', torch.linspace(0, np.pi, self.n_coeffs))

        self.to_initial = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_resonance = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_phase_dither = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
    
    def forward(self, latents):
        batch_size = latents.shape[0]

        initial = self.to_initial(latents)
        res = self.base_resonance + (torch.sigmoid(self.to_resonance(latents)) * self.resonance_factor)
        dither = torch.sigmoid(self.to_phase_dither(latents))


        first_frame = initial[:, None, :]

        frames = [first_frame]
        phases = [torch.zeros_like(first_frame).uniform_(-np.pi, np.pi)]

        # TODO: This should also incorporate impulses, i.e., new excitations
        # beyond the original
        for i in range(self.n_frames - 1):

            mag = frames[i]
            phase = phases[i]

            # compute next polar coordinates
            nxt_mag = mag * res

            nxt_phase = \
                phase \
                + self.group_delay[None, None, None, :] \
                + (dither * torch.zeros_like(dither).uniform_(-np.pi, np.pi)[:, None, :, :]) 


            frames.append(nxt_mag)
            phases.append(nxt_phase)
        


        mags = torch.cat(frames, dim=1)
        phases = torch.cat(phases, dim=1)

        frames = torch.complex(
            mags * torch.cos(phases),
            mags * torch.sin(phases)
        )

        windowed = torch.fft.irfft(frames, dim=-1, norm='ortho')
        
        windowed = windowed.permute(0, 2, 1, 3)
        samples = overlap_add(windowed, apply_window=True)[..., :self.resonance_samples]

        assert samples.shape == (batch_size, self.encoding_channels, self.resonance_samples)
        return samples
                        


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
        self.filter_kernel_size = 16
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
            start_size=4,
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

        self.embed_periodicity = nn.Linear(257, 8)
        self.to_channel_dim = nn.Conv1d(exp.n_bands * 8, channels, 1, 1, 0)

        self.stack = nn.Sequential(
            DilatedBlock(channels, 1),    
            DilatedBlock(channels, 3),    
            DilatedBlock(channels, 9),    
            DilatedBlock(channels, 1),    
        )

        # self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)

        self.up = ConvUpsample(
            channels, 
            channels, 
            start_size=128, 
            end_size=exp.n_samples, 
            mode='learned', 
            out_channels=encoding_channels, 
            from_latent=False, 
            batch_norm=True)
        
        self.lateral_height = 7
        self.lateral_width = 25

        self.lateral_height_padding = self.lateral_height // 2
        self.lateral_width_padding = self.lateral_width // 2

        self.lateral_competition = nn.Parameter(torch.zeros(1, 1, self.lateral_height, self.lateral_width).uniform_(-1, 1))

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


        self.to_resonance = RecurrentResonanceModel(
            self.encoding_channels,
            self.impulse_samples,
            self.latent_dim,
            self.channels,
            512,
            self.resonance_samples)
    
        
        
        self.to_impulse = GenerateImpulse(
            latent_dim, channels, self.impulse_samples, 8, encoding_channels)
        
        self.to_mix = GenerateMix(
            latent_dim, channels, encoding_channels)
        
        self.embed_verb_latent = nn.Linear(encoding_channels, latent_dim)
        self.verb = ReverbGenerator(latent_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((latent_dim,)))

        self.apply(lambda x: exp.init_weights(x))
    
        
    def forward(self, x):
        batch_size = x.shape[0]

        # x = exp.fb.forward(x, normalize=False)
        if len(x.shape) == 4:
            pif = x
        else:
            pif = exp.perceptual_feature(x)
        pif = self.embed_periodicity(pif).permute(0, 3, 1, 2).reshape(batch_size, 8 * exp.n_bands, -1)
        x = self.to_channel_dim(pif)

        x = self.stack(x)

        x = self.up(x)

        x = F.dropout(x, 0.01)
        # rectification (only positive activations)
        encoding = x = torch.relu(x)

        # lateral comp.
        # encoding = encoding[:, None, :, :]
        # encoding = F.conv2d(encoding, self.lateral_competition, stride=(1, 1), padding=(self.lateral_height_padding, self.lateral_width_padding))
        # encoding = encoding.view(batch_size, self.encoding_channels, exp.n_samples)

        ctxt = torch.sum(encoding, dim=-1)

        # using the context, compute the vectors used to modulate the base latents
        impulse_latent = self.embed_impulse_latent(ctxt)[:, None, :] + self.impulse_latent
        resonance_latent = self.embed_resonance_latent(ctxt)[:, None, :] + self.resonance_latent
        mix_latent = self.embed_mix_latent(ctxt)[:, None, :] + self.mix_latent

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

        # TODO: what happens if I skip this convolution?
        # is it better to just add the two together
        if convolve_impulse_and_resonance:
            d = fft_convolve(impulses, resonances)
        else:
            d = resonances
        d = (impulses * imp_amp) + (d * res_amp)
        assert d.shape == (batch_size, self.encoding_channels, self.resonance_samples)
        
        d = F.pad(d, (0, exp.n_samples - self.resonance_samples))
        d = unit_norm(d, dim=-1)
        x = fft_convolve(d, encoding)[..., :exp.n_samples]
        x = torch.sum(x, dim=1, keepdim=True)

        verb_ctxt = self.embed_verb_latent(ctxt)
        x = self.verb.forward(verb_ctxt, x)
        return x, encoding

model = Model(
    channels=128, 
    encoding_channels=512, 
    impulse_samples=256 * 8, 
    resonance_samples=full_atom_size,
    latent_dim=16,
).to(device)

optim = optimizer(model, lr=1e-3)

def train(batch, i):
    # batch_size = batch.shape[0]
    optim.zero_grad()


    with torch.no_grad():
        feat = exp.perceptual_feature(batch)

    recon, encoding = model.forward(feat)

    # encoding = encoding.view(batch_size, -1)
    # srt, indices = torch.sort(encoding, dim=-1, descending=True)

    sparsity_loss = encourage_sparsity_loss(
        encoding, 
        128,
        0.00001
    )

    # loss = exp.perceptual_loss(recon, batch) + sparsity_loss
    fake_feat = exp.perceptual_feature(recon)

    loss = F.mse_loss(feat, fake_feat) + sparsity_loss

    loss.backward()
    optim.step()
    return loss, recon

@readme
class SparseResonance2(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    