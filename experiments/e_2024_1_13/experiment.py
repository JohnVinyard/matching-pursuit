from typing import Callable, Dict

import numpy as np
import torch

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
from modules.phase import AudioCodec, MelScale, morlet_filter_bank
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from modules.transfer import ResonanceChain
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from modules.upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from sklearn.cluster import MiniBatchKMeans

from vector_quantize_pytorch import VectorQuantize

exp = Experiment(
    samplerate=22050,
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512)

n_events = 16
context_dim = 16
impulse_size = 16384
resonance_size = 32768
samplerate = 22050
n_samples = 32768

        


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
        self.coarse_coeffs = 257
        
        self.base_resonance = 0.02
        self.res_factor = (1 - self.base_resonance) * 0.99
        
        low_hz = 40
        high_hz = 4000
        
        low_samples = int(samplerate) // low_hz
        high_samples = int(samplerate) // high_hz
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
            nn.Linear(latent_dim, self.n_atoms) 
            for _ in range(self.n_piecewise)
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
            mode='nearest', 
            out_channels=n_piecewise, 
            from_latent=True, 
            batch_norm=True)
        
        
        self.final_mix = nn.Linear(latent_dim, 2)
        
        
        # self.register_buffer('env', torch.linspace(1, 0, self.resonance_size))
        # self.max_exp = 20
        
    
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
            # and convolve the impulse with each
            
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
        # mx = F.avg_pool1d(mx, 9, 1, 4)
        mx = torch.softmax(mx, dim=1)
        mx = mx.view(-1, n_events, self.n_piecewise, self.resonance_size).permute(0, 2, 1, 3)
        
        final_convs = (mx * convs).sum(dim=1)
        
        # apply time-varying filter
        # TODO: To avoid windowing artifacts, this is really just the 
        # same process again:  Convole the entire signal with N different
        # filters and the produce a smooth mixture over time
        windowed = windowed_audio(final_convs, 512, 256)
        windowed = unit_norm(windowed, dim=-1)
        windowed = torch.fft.rfft(windowed, dim=-1)
        windowed = windowed * filt
        windowed = torch.fft.irfft(windowed)
        final_convs = overlap_add(windowed, apply_window=False)[..., :self.resonance_size]\
            .view(-1, n_events, self.resonance_size)
        
        final_mx = self.final_mix(latent)
        final_mx = torch.softmax(final_mx, dim=-1)
        
        # final_convs = unit_norm(final_convs)
        # imp = unit_norm(imp)
        
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
            self.n_frames * 4,
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
        
        
        self.embed_spec = nn.Conv1d(1024, 1024, 1, 1, 0)
        self.pos = nn.Parameter(torch.zeros(1, 1024, 128).uniform_(-0.01, 0.01))
        

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
        

        self.weighting = nn.Conv1d(1024, 4096, 1, 1, 0)
        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)
        
        
        

    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}
        
        batch_size = x.shape[0]
        
        if x.shape[1] == 1:
            x = stft(x, 2048, 256, pad=True).view(
                batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)
        
        x = self.embed_spec(x)
        x = x + self.pos
        
        batch_size = x.shape[0]

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x
        

        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]

        x = self.proj(x)
        
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = UNet(1024)
    
        self.embed_context = nn.Linear(4096, 256)
        self.embed_one_hot = nn.Linear(4096, 256)

        self.imp = GenerateImpulse(256, 128, impulse_size, 16, n_events)

        
        total_atoms = 2048
        f0s = np.linspace(40, 4000, total_atoms // 4)
        waves = make_waves(resonance_size, f0s, int(samplerate))
        
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
        
        
        # total_atoms = 1024
        # f0s = np.linspace(40, 4000, total_atoms // 4)
        # waves = make_waves(resonance_size, f0s, int(samplerate))
        
        # self.res = ResonanceChain(
        #     2, 
        #     n_atoms=1024, 
        #     window_size=512, 
        #     n_frames=128, 
        #     total_samples=resonance_size, 
        #     mix_channels=4, 
        #     channels=128, 
        #     latent_dim=256,
        #     initial=waves)
        

        # self.mix = GenerateMix(256, 128, n_events, mixer_channels=3)
        self.to_amp = nn.Linear(256, 1)

        # self.verb_context = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(
            context_dim, 3, samplerate, n_samples, norm=nn.LayerNorm((context_dim,)))

        self.to_context_mean = nn.Linear(4096, context_dim)
        self.to_context_std = nn.Linear(4096, context_dim)
        self.embed_memory_context = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 256)
        
        self.fine_shift = nn.Linear(256, 1)
        self.shift_factor = (256 / resonance_size) * 0.5
        

        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(
            self.refractory_period, power=10, device=device))
        
        self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-0.01, 0.01))

        # self.apply(lambda x: exp.init_weights(x))
    
    def from_sparse(self, sparse, ctxt):
        encoded, packed, one_hot = sparsify2(sparse, n_to_keep=n_events)
        x, imp = self.generate(encoded, one_hot, packed, ctxt)
        return x, imp, encoded


    def encode(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        encoded = self.encoder.forward(x)
        encoded = F.dropout(encoded, 0.05)

        ref = F.pad(self.refractory,
                    (0, encoded.shape[-1] - self.refractory_period))
        encoded = fft_convolve(encoded, ref)[..., :encoded.shape[-1]]

        return encoded
    
    def sparse_encode(self, x):
        encoded = self.encode(x)
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        return encoded
    
    def from_sparse(self, sparse, ctxt):
        encoded, packed, one_hot = sparsify2(sparse, n_to_keep=n_events)
        x, imp = self.generate(encoded, one_hot, packed, ctxt)
        return x, imp, encoded

    def generate(self, encoded, one_hot, packed, dense):
        
        
        # ctxt = torch.sum(encoded, dim=-1)
        # dense = self.embed_memory_context(ctxt)  # (batch, context_dim)

        # ctxt is a single vector
        # ce = self.embed_context(ctxt)

        # one hot is n_events vectors
        proj = self.from_context(dense)
        oh = self.embed_one_hot(one_hot)
        
        # print(dense.shape, proj.shape, oh.shape)

        embeddings = proj[:, None, :] + oh

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        # padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        mixed = self.res.forward(embeddings, imp)
        mixed = mixed.view(-1, n_events, resonance_size)
        mixed = unit_norm(mixed)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps
        
        # shift = torch.tanh(self.fine_shift(embeddings)) * self.shift_factor

        # coarse positioning
        final = F.pad(mixed, (0, n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, n_samples, device=final.device)
        up[:, :, ::256] = packed
        
        # fine positioning
        # up = fft_shift(up, shift)

        final = fft_convolve(final, up)[..., :n_samples]

        final = self.verb.forward(dense, final)

        return final, imp

    def forward(self, x):
        encoded = self.encode(x)
        
        dense = torch.mean(encoded, dim=-1)
        mean = self.to_context_mean(dense)
        std = self.to_context_std(dense)
        dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        encoded = encoded + self.atom_bias[None, :, None]
        encoded = torch.relu(encoded)
        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        
        final, imp = self.generate(encoded, one_hot, packed, dense)
        
        # print('ENCODED', encoded.max().item())
        # print('DENSE', dense.mean().item(), dense.std().item())
        
        
        return final, encoded, imp
    
    
    def random_generation(self, exponent=2):
        with torch.no_grad():
            # generate context latent
            z = torch.zeros(1, context_dim).normal_(0, 0.1)
            
            # generate events
            events = torch.zeros(1, 1, 4096, 128).uniform_(0, 1.6)
            events = F.avg_pool2d(events, (7, 7), (1, 1), (3, 3))
            events = events.view(1, 4096, 128)
            
            ch, _, encoded = self.from_sparse(events, z)
            ch = torch.sum(ch, dim=1, keepdim=True)
            ch = max_norm(ch)
        
        return ch, encoded
    
    def derive_events_and_context(self, x: torch.Tensor):
        
        print('STARTING WITH', x.shape)
        
        encoded = self.encode(x)

        dense = torch.mean(encoded, dim=-1)
        mean = self.to_context_mean(dense)
        std = self.to_context_std(dense)
        dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        
        
        return encoded, dense




model = Model().to(device)
optim = optimizer(model, lr=1e-3)


n_patch_centroids = 64

vq = VectorQuantize(9*9, n_patch_centroids, 128, heads=1, kmeans_init=True, threshold_ema_dead_code=2).to(device)
vq_optim = optimizer(vq, lr=1e-3)


import zounds
band = zounds.FrequencyBand(40, exp.samplerate)
scale = zounds.MelScale(band, 1024)
bank = morlet_filter_bank(exp.samplerate, 512, scale, scaling_factor=0.1, normalize=True).real.astype(np.float32)
bank = torch.from_numpy(bank).to(device).view(1024, 1, 512)

def spectral(a: torch.Tensor):
    return stft(a, 2048, 256, pad=True).view(a.shape[0], 128, 1025).permute(0, 2, 1)
    # # return codec.to_frequency_domain(a.view(-1, exp.n_samples))[..., 0]
    # spec = F.conv1d(a, bank, padding=256)
    # spec = torch.relu(spec)
    # spec = F.avg_pool1d(spec, 512, 256, padding=256)
    # return spec

def patches(spec: torch.Tensor, size: int = 9, step: int = 3):
    batch, channels, time = spec.shape
    
    p = spec.unfold(1, size, step).unfold(2, size, step)
    last_dim = np.prod(p.shape[-2:])
    p = p.reshape(batch, -1, last_dim)
    norms = torch.norm(p, dim=-1, keepdim=True)
    normed = p / (norms + 1e-12)
    
    return p, norms, normed

def learn_and_label(patches: torch.Tensor):
    vq_optim.zero_grad()
    batch, n_patches, patch_size = patches.shape
    quantized, indices, loss = vq.forward(patches)
    loss = loss + F.mse_loss(quantized.view(*patches.shape), patches)
    loss.backward()
    vq_optim.step()
    return indices.view(batch, n_patches, 1)
    

def experiment_loss(batch: torch.Tensor, fake: torch.Tensor, channels: torch.Tensor, encoded: torch.Tensor):
    # used_elements = (torch.abs(encoded).sum() / batch.shape[0]) * 1e-3
    
    real_spec = spectral(batch)
    fake_spec = spectral(fake)
    
    # smoothness_loss = F.mse_loss(fake_spec, real_spec)
    
    # channel_specs = stft(channels, 2048, 256, pad=True).view(batch.shape[0], n_events, -1)
    
    # # encourage events to be orthogonal/non-overlapping
    # # TODO: Use constant-q spectrogram with better frequency resolution
    # sim = channel_specs @ channel_specs.permute(0, 2, 1)
    # sim = torch.triu(sim)
    # difference_loss = sim.mean() * 1e-2
    
    # break up the spectrograms into overlapping patches
    rp, rn, rnorm = patches(real_spec)
    fp, fn, fnorm = patches(fake_spec)
    
    
    real_labels: torch.Tensor = learn_and_label(rnorm)
    
    total_elements = real_labels.nelement()
    counts = torch.bincount(real_labels.view(-1), minlength=n_patch_centroids)
    # this is the percentage of patches that belong to this cluster/label
    frequency = counts / total_elements
    # we want inverse frequency
    weighting = 1 / frequency
    
    # get patch losses
    diff = ((rp - fp) ** 2).mean(dim=-1, keepdim=True)
    
    # get importance weighting for each patch
    weights = weighting[real_labels.view(-1)].view(batch.shape[0], -1, 1)
    
    # weight patch losses by importance
    loss = (diff * weights).mean()
    
    # print(loss.item(), smoothness_loss.item(), used_elements.item())
    
    # loss = loss + smoothness_loss + used_elements
    
    
    return loss
    
    


def train(batch, i):
    optim.zero_grad()

    b = batch.shape[0]
    
    recon, encoded, imp = model.forward(batch)
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    
    loss = experiment_loss(batch, recon_summed, recon, encoded)
    
    loss.backward()
    optim.step()
    
    
    # with torch.no_grad():
        
    #     # switch to cpu evaluation
    #     model.to('cpu')
    #     model.eval()
        
    #     # generate context latent
    #     z = torch.zeros(1, context_dim).normal_(0, 12)
        
    #     # generate events
    #     events = torch.zeros(1, 1, 4096, 128).uniform_(0, 130)
    #     events = F.avg_pool2d(events, (7, 7), (1, 1), (3, 3))
    #     events = events.view(1, 4096, 128)
        
    #     ch, _, encoded = model.from_sparse(events, z)
    #     ch = torch.sum(ch, dim=1, keepdim=True)
        
    #     # switch back to GPU training
    #     model.to('cuda')
    #     model.train()
    
    # recon = max_norm(ch)
    
    recon = max_norm(recon_summed)
    encoded = max_norm(encoded)
    
    return loss, recon, encoded


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x[:, None, :, :]
        x = F.max_pool2d(x, (16, 8), (16, 8))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)


@readme
class PatchInfoLoss(BaseExperimentRunner):
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
            item = item.view(-1, 1, n_samples)
            l, r, e = train(item, i)

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l, i)
