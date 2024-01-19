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

        # self.atom_bias = nn.Parameter(torch.zeros(4096).uniform_(-0.01, 0.01))

        # self.apply(lambda x: exp.init_weights(x))



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

    def from_sparse(self, sparse, ctxt):
        encoded, packed, one_hot = sparsify2(sparse, n_to_keep=n_events)
        x, imp = self.generate(encoded, one_hot, packed, ctxt)
        return x, imp, encoded
    
    
    def generate(self, encoded, one_hot, packed, dense):
        

        # one hot is n_events vectors
        proj = self.from_context(dense)
        
        oh = self.embed_one_hot(one_hot)
        

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
        
        # encoded = encoded + self.atom_bias[None, :, None]
        # encoded = torch.relu(encoded)

        # note that some of the "top" channels will be zero, the eventual
        # scheduling convolution will make the event silent        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        
        final, imp = self.generate(encoded, one_hot, packed, dense)
        
        # print('ENCODED', encoded.max().item())
        # print('DENSE', dense.mean().item(), dense.std().item())
        
        
        return final, encoded, imp
    
    


model = Model().to(device)
optim = optimizer(model, lr=1e-3)



import zounds
band = zounds.FrequencyBand(40, 22050 // 2)
scale = zounds.LinearScale(band, 128)
sr = zounds.SR22050()

filter_bank = morlet_filter_bank(sr, 128, scale, 0.1, normalize=True).real
filter_bank = torch.from_numpy(filter_bank).to(device).float().view(128, 1, 128)


def compute_features(
    x: torch.Tensor, 
    kernel_size=128, 
    pif_window=256,
    pif_step=64, 
    scattering_factor=100, 
    time_residual_factor=500, 
    harm_residual_factor=250):
    
    batch_size = x.shape[0]
    
    full_spec = F.conv1d(x, filter_bank, stride=1, padding=kernel_size // 2)
    full_spec = torch.relu(full_spec)
    
    pooled = F.avg_pool1d(full_spec, kernel_size, 1, padding=kernel_size // 2)[..., :full_spec.shape[-1]]
    residual = full_spec - pooled
    windowed = residual.unfold(-1, pif_window, pif_step)
    scattering = torch.abs(torch.fft.rfft(windowed, dim=-1)) * scattering_factor
    
    pooled = F.avg_pool1d(full_spec, kernel_size, kernel_size // 2, padding=kernel_size // 2)
    
    time_pooled = F.avg_pool1d(pooled, kernel_size=25, stride=1, padding=12)
    time_residual = (pooled - time_pooled) * time_residual_factor
    
    pooled = pooled.permute(0, 2, 1)
    harm_pooled = F.avg_pool1d(pooled, kernel_size=25, stride=1, padding=12)
    harm_residual = (pooled - harm_pooled) * harm_residual_factor
    
    x = torch.cat([
        scattering.reshape(batch_size, -1), 
        time_pooled.reshape(batch_size, -1), 
        time_residual.reshape(batch_size, -1),
        harm_pooled.reshape(batch_size, -1),
        harm_residual.reshape(batch_size, -1)
    ], dim=-1)
    
    return x


def feature(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 1024)
    feature = torch.cat([compute_features(y) for y in bands.values()], dim=-1)
    return feature

def feature_loss(a: torch.Tensor, b: torch.Tensor):
    a = feature(a)
    b = feature(b)
    return F.mse_loss(a, b)

def dict_op(
        a: Dict[int, torch.Tensor],
        b: Dict[int, torch.Tensor],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[int, torch.Tensor]:
    return {k: op(v, b[k]) for k, v in a.items()}



def scattering(x: torch.Tensor):
    x = stft(x, 2048, 256, pad=True).view(-1, 128, 1025)
    
    spectral_pooled = F.avg_pool1d(x, 25, 1, padding=12)
    spectral_residual = x - spectral_pooled
    
    x = x.permute(0, 2, 1)
    
    time_pooled = F.avg_pool1d(x, 25, 1, padding=12)
    time_residual = x - time_pooled
    
    return spectral_residual, time_residual, spectral_pooled, time_pooled


def scattering_loss(a: torch.Tensor, b: torch.Tensor):
    rsr, rtr, rsp, rtp = scattering(a)    
    fsr, ftr, fsp, ftp = scattering(b)    
    
    spec_residual_loss = F.mse_loss(fsr, rsr) * 100
    time_residual_loss = F.mse_loss(ftr, rtr) * 500
    
    spec_pooled_loss = F.mse_loss(fsp, rsp)
    time_pooled_loss = F.mse_loss(ftp, rtp)
    
    
    return spec_residual_loss + time_residual_loss + spec_pooled_loss + time_pooled_loss


def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d2 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    return dict(**d1, **d2)
    


def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

    target = multiband_transform(target)

    full = torch.sum(recon, dim=1, keepdim=True)
    full = multiband_transform(full)

    residual = dict_op(target, full, lambda a, b: a - b)
    
    # we want to make sure that all the atoms bring the residual 
    # as close to 0 as possible.  The atoms should fully explain
    # the audio segment, and should also not require that the random
    # channel chosen has to make up for other channel's mistakes
    overshoot = sum([torch.norm(x) for x in residual.values()]) / len(residual)
    print('OVERSHOOT', overshoot.item())
    
    # since the residual contains elements from the network itself
    # make sure we're not optimizing those values
    residual = {k: v.clone().detach() for k, v in residual.items()}
    
    i = np.random.randint(0, n_events)
    ch = recon[:, i: i + 1, :]
    ch = multiband_transform(ch)
    t = dict_op(residual, ch, lambda a, b: a + b)
    # ask one channel to explain as much of the residual as possible
    diff = dict_op(ch, t, lambda a, b: a - b)
    loss = sum([torch.abs(y).mean() for y in diff.values()]) / len(diff)
    return loss + overshoot



def train(batch, i):
    optim.zero_grad()

    b = batch.shape[0]
    
    recon, encoded, imp = model.forward(batch)
    
    
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    
    
    # TODO: add single-channel loss, orthogonal loss and sparsity loss
    # loss = single_channel_loss(batch, recon)
    # loss = scattering_loss(batch, recon_summed)
    loss = feature_loss(recon_summed, batch)
    
        
    loss.backward()
    optim.step()
    
    
    # with torch.no_grad():
        
    #     # switch to cpu evaluation
    #     model.to('cpu')
    #     model.eval()
        
    #     # generate context latent
    #     z = torch.zeros(1, context_dim).normal_(0, 0.3)
        
    #     # generate events
    #     events = torch.zeros(1, 1, 4096, 128).uniform_(0, 4)
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
class OrthogonalEvents(BaseExperimentRunner):
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
