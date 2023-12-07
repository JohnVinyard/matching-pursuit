from typing import Callable, Dict

import numpy as np
import torch
import zounds
from conjure import numpy_conjure, SupportedContentType
from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F

from config.experiment import Experiment
from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.ddsp import NoiseModel
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import sparsify2, sparsify_vectors
from modules.stft import stft
from modules.transfer import ImpulseGenerator, fft_shift
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from modules.upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from random import choice


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512)

n_events = 64
context_dim = 16
impulse_size = 4096
resonance_size = 32768



class ResonanceModel2(nn.Module):
    def __init__(
        self, 
        latent_dim, 
        channels, 
        resonance_size, 
        n_atoms, 
        n_piecewise, 
        init_atoms=None, 
        learnable_atoms=False,
        mixture_over_time=False,
        n_frames = 128):
        
        super().__init__()
        
        self.n_frames = n_frames
        self.latent_dim = latent_dim
        self.channels = channels
        self.resonance_size = resonance_size
        self.n_atoms = n_atoms
        self.n_piecewise = n_piecewise
        self.init_atoms = init_atoms
        self.learnable_atoms = learnable_atoms
        self.mixture_over_time = mixture_over_time
        
        self.n_coeffs = (self.resonance_size // 2) + 1
        self.coarse_coeffs = 32
        
        self.base_resonance = 0.02
        self.res_factor = (1 - self.base_resonance) * 0.95
        
        low_hz = 40
        high_hz = 4000
        
        low_samples = int(exp.samplerate) // low_hz
        high_samples = int(exp.samplerate) // high_hz
        spacings = torch.linspace(low_samples, high_samples, self.n_atoms)
        print('SMALLEST SPACING', low_samples, 'HIGHEST SPACING', high_samples)
        oversample_rate = 8
        
        if init_atoms is None:
            atoms = torch.zeros(self.n_atoms, self.resonance_size * oversample_rate)
            for i, spacing in enumerate(spacings):
                sp = int(spacing * oversample_rate)
                atoms[i, ::(sp + 1)] = 1
            
            atoms = F.avg_pool1d(atoms.view(1, 1, -1), kernel_size=oversample_rate, stride=oversample_rate).view(self.n_atoms, self.resonance_size)
            if learnable_atoms:
                self.atoms = nn.Parameter(atoms)
            else:
                self.register_buffer('atoms', atoms)
        else:
            if learnable_atoms:
                self.atoms = nn.Parameter(init_atoms)
            else:
                self.register_buffer('atoms', init_atoms)
        
        self.selections = nn.ModuleList([
            nn.Linear(latent_dim, self.n_atoms) for _ in range(self.n_piecewise)
        ])
        
        self.decay = nn.Linear(latent_dim, self.n_frames)
        
        
        self.to_filter = ConvUpsample(
            latent_dim,
            channels,
            start_size=8,
            end_size=self.n_frames,
            mode='nearest',
            out_channels=self.coarse_coeffs,
            from_latent=True,
            batch_norm=True
        )
        
        self.to_mixture = ConvUpsample(
            latent_dim, 
            channels, 
            start_size=8, 
            end_size=self.n_frames, 
            mode='learned', 
            out_channels=n_piecewise, 
            from_latent=True, 
            batch_norm=True)
        
        
        self.final_mix = nn.Linear(latent_dim, 2)
        
        self.register_buffer('env', torch.linspace(1, 0, self.resonance_size))
        self.max_exp = 20
    
    def forward(self, latent, impulse):
        """
        Generate:
            - n selections
            - n decay exponents
            - n filters
            - time-based mixture
        """
        
        batch_size = latent.shape[0]
        
        # TODO: There should be another collection for just resonances
        convs = []
        
        imp = F.pad(impulse, (0, self.resonance_size - impulse.shape[-1]))
        
        
        decay = torch.sigmoid(self.decay(latent))
        decay = self.base_resonance + (decay * self.res_factor)
        decay = torch.log(1e-12 + decay)
        decay = torch.cumsum(decay, dim=-1)
        decay = torch.exp(decay)
        decay = F.interpolate(decay, size=self.resonance_size, mode='linear')
        

        # produce time-varying, frequency-domain filter coefficients        
        filt = self.to_filter(latent).view(-1, self.coarse_coeffs, self.n_frames).permute(0, 2, 1)
        filt = torch.sigmoid(filt)
        filt = F.interpolate(filt, size=257, mode='linear')
        filt = filt.view(batch_size, n_events, self.n_frames, 257)
        
        
        for i in range(self.n_piecewise):
            # choose a linear combination of resonances
            sel = self.selections[i].forward(latent)
            sel = torch.softmax(sel, dim=-1)
            res = sel @ self.atoms
            res = res * decay
            conv = fft_convolve(res, imp)
            convs.append(conv[:, None, :, :])
            
        
        # TODO: Concatenate both the pure resonances and the convolved audio
        convs = torch.cat(convs, dim=1)
        
        # produce a linear mixture-over time
        mx = self.to_mixture(latent)
        mx = F.interpolate(mx, size=self.resonance_size, mode='linear')
        mx = F.avg_pool1d(mx, 9, 1, 4)
        mx = torch.softmax(mx, dim=1)
        mx = mx.view(-1, n_events, self.n_piecewise, self.resonance_size).permute(0, 2, 1, 3)
        
        final_convs = (mx * convs).sum(dim=1)
        
        # apply time-varying filter
        windowed = windowed_audio(final_convs, 512, 256)
        windowed = unit_norm(windowed, dim=-1)
        windowed = torch.fft.rfft(windowed, dim=-1)
        windowed = windowed * filt
        windowed = torch.fft.irfft(windowed)
        final_convs = overlap_add(windowed, apply_window=False)[..., :self.resonance_size]\
            .view(-1, n_events, self.resonance_size)
        
        final_mx = self.final_mix(latent)
        final_mx = torch.softmax(final_mx, dim=-1)
        
        final_convs = unit_norm(final_convs)
        imp = unit_norm(imp)
        
        stacked = torch.cat([final_convs[..., None], imp[..., None]], dim=-1)
        
        final = stacked @ final_mx[..., None]
        final = final.view(-1, n_events, self.resonance_size)
    
        return final



def make_waves(n_samples, f0s, samplerate):
    sawtooths = []
    squares = []
    triangles = []
    sines = []
    
    total_atoms = len(f0s) * 4

    for f0 in f0s:
        f0 = f0 / (samplerate // 2)
        rps = f0 * np.pi
        radians = np.linspace(0, rps * n_samples, n_samples)
        sq = square(radians)[None, ...]
        squares.append(sq)
        st = sawtooth(radians)[None, ...]
        sawtooths.append(st)
        tri = sawtooth(radians, 0.5)[None, ...]
        triangles.append(tri)
        sin = np.sin(radians)
        sines.append(sin[None, ...])
    
    waves = np.concatenate([sawtooths, squares, triangles, sines], axis=0)
    waves = torch.from_numpy(waves).view(total_atoms, n_samples).float()
    return waves




class GenerateMix(nn.Module):

    def __init__(self, latent_dim, channels, encoding_channels, mixer_channels=2):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        mixer_channels = mixer_channels

        self.to_mix = LinearOutputStack(
            channels, 3, out_channels=mixer_channels, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))

    def forward(self, x):
        x = self.to_mix(x)
        x = x.view(-1, self.encoding_channels, 1)
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

        self.to_frames = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='learned',
            end_size=self.n_frames,
            out_channels=channels,
            batch_norm=True
        )

        self.noise_model = NoiseModel(
            channels,
            self.n_frames,
            self.n_frames * 16,
            self.n_samples,
            self.channels,
            batch_norm=True,
            squared=True,
            activation=lambda x: torch.sigmoid(x),
            mask_after=1
        )
        
        self.to_env = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):
        
        env = self.to_env(x) ** 2
        env = F.interpolate(env, mode='linear', size=self.n_samples)
        
        x = self.to_frames(x)
        x = self.noise_model(x)
        x = x.view(-1, n_events, self.n_samples)
        
        x = x * env
        return x



class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        
        self.initial = nn.Conv1d(channels, channels, 1, 1, 0)
        self.pos_encoding = nn.Parameter(torch.zeros(1, channels, 128).uniform_(-0.1, 0.1))

        self.down = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 128
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.attn = nn.Conv1d(1024, 1, 1, 1, 0)
        self.proj = nn.Conv1d(1024, context_dim, 1, 1, 0)

    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}
        
        batch_size = x.shape[0]
        
        x = self.initial(x)
        x = x + self.pos_encoding

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x

        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]

        
        attn = torch.sigmoid(self.attn(x))
        x = self.proj(x)
        latents, indices = sparsify_vectors(x, attn, n_to_keep=n_events)
        
        return latents


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024)
    
        self.embed_context = nn.Linear(16, 256)
        self.embed_encoding = nn.Linear(16, 256)

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)

        
        total_atoms = 2048
        f0s = np.linspace(40, 4000, total_atoms // 4)
        waves = make_waves(resonance_size, f0s, int(exp.samplerate))
        
        self.res = ResonanceModel2(
            256, 
            128, 
            resonance_size, 
            n_atoms=total_atoms, 
            n_piecewise=4, 
            init_atoms=waves, 
            learnable_atoms=False, 
            mixture_over_time=True,
            n_frames=128)

        # self.mix = GenerateMix(256, 128, n_events, mixer_channels=3)
        self.to_amp = nn.Linear(256, 1)

        # self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            context_dim, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((context_dim,)))

        self.embed_memory_context = nn.Linear(context_dim, context_dim)

        self.from_context = nn.Linear(context_dim, 4096)
        
        self.to_pos = nn.Linear(256, 1)
        self.gen_impulse = ImpulseGenerator(exp.n_samples, softmax=lambda x: sparse_softmax(x))
        

        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(
            self.refractory_period, power=10, device=device))

        self.apply(lambda x: exp.init_weights(x))

    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded = self.encoder.forward(x)

        return encoded
    

    def generate(self, encoded):
        ctxt = torch.sum(encoded, dim=1)
        dense = self.embed_memory_context(ctxt)  # (batch, context_dim)

        
        embeddings = self.embed_encoding(encoded)
        pos = torch.sigmoid(self.to_pos(embeddings)).view(-1, n_events)

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        # padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        mixed = self.res.forward(embeddings, imp)
        # mixed = unit_norm(mixed)

        amps = torch.abs(self.to_amp(embeddings)).view(-1, n_events)
        # mixed = mixed * amps[..., None]
        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        final = self.verb.forward(dense, final)
        final = unit_norm(final)
        
        norms = amps
        
        new_final = torch.zeros_like(final)
        new_amps = torch.zeros_like(amps)
        new_pos = torch.zeros_like(pos)
        
        batch_size = encoded.shape[0]        
    
        # sort from loudest to quietest
        for b in range(batch_size):
            indices = torch.argsort(norms[b], descending=True)
            for i, index in enumerate(indices):
                new_final[b, i] = final[b, index]
                new_amps[b, i] = amps[b, index]
                new_pos[b, i] = pos[b, index]
                
        return new_final, new_pos, new_amps

    def forward(self, x):
        encoded = self.encode(x)

        atoms, pos, amp = self.generate(encoded)
        return atoms, pos, amp, encoded


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def loss_func(batch, atoms, positions, amps):
    loss = 0

    residual = batch
    
    
    for i in range(n_events):
        
        residual = residual.clone().detach()
        
        ch = atoms[:, i: i + 1, :]
        pos = positions[:, i: i + 1]
        amp = amps[:, i: i + 1]
        

        # find the best possible location for this atom        
        fm = fft_convolve(residual, ch)
        best = torch.argmax(fm, dim=-1, keepdim=True)
        best_amps = torch.gather(fm, dim=-1, index=best)
        best = best / exp.n_samples
        
        
        # the selected atom, given the benefit of the doubt about 
        # position and amplitude
        shifted = fft_shift(ch, best.view(-1, 1, 1)) * best_amps.view(-1, 1, 1)
        
        amp_loss = F.mse_loss(amp[..., None], best_amps)
        pos_loss = F.mse_loss(pos[..., None], best)
        atom_loss =  F.mse_loss(shifted, residual)
        
        # start_norm = torch.norm(residual, dim=-1)
        residual = residual - shifted
        # end_norm = torch.norm(residual, dim=-1)
        # atom_loss = (end_norm / (start_norm + 1e-4))
        
        # atom loss is weighted by the ground-truth best amplitude
        # atom_loss = atom_loss * (n_events - i)
        # atom_loss = atom_loss.mean()
        
        loss = loss + amp_loss + pos_loss + atom_loss
    
    return loss
        

def train(batch, i):
    optim.zero_grad()

    b = batch.shape[0]
    
    recon, pos, amp, encoded = model.forward(batch)
    
    adjusted = fft_shift(recon, pos[..., None]) * amp[..., None]
    recon_summed = torch.sum(adjusted, dim=1, keepdim=True)

    loss = loss_func(batch, recon, pos, amp)    
    loss.backward()
    optim.step()

    recon = max_norm(recon_summed)
    return loss, recon, encoded


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        # x = x[:, None, :, :]
        # x = F.max_pool2d(x, (16, 8), (16, 8))
        # x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)


@readme
class GraphRepresentation6(BaseExperimentRunner):
    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None, load_weights=True, save_weights=False, model=model):
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            load_weights=load_weights, 
            save_weights=save_weights, 
            model=model)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)

            self.encoded = e
            self.real = item
            self.fake = r
            print(i, l.item())
            self.after_training_iteration(l, i)
