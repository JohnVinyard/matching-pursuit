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


exp = Experiment(
    samplerate=22050,
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512)

n_events = 1
context_dim = 16
impulse_size = 16384
resonance_size = 32768
samplerate = 22050
n_samples = 32768

class RecurrentResonance(nn.Module):
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
        
        self.register_buffer('group_delay', torch.linspace(0, np.pi, 257))
        
        self.to_resonances = ConvUpsample(
            latent_dim,
            channels,
            start_size=8,
            end_size=self.n_frames,
            mode='learned',
            out_channels=257 * 2,
            batch_norm=True,
            from_latent=True
        )
        
        
    
    def forward(self, latent, impulse):
        '''
            - take stft of impulse
            - generate a sequence of transfer functions (including phase dither)
            - each time step is current_input + (previous input * resonance)
        '''
        batch_size, n_events, _ = impulse.shape
        
        latent = latent.view(-1, self.latent_dim)
        impulse = F.pad(impulse, (0, resonance_size - impulse_size))
        impulse = impulse.view(-1, resonance_size)
        
        
        imp = stft(impulse, 512, 256, pad=True, return_complex=True)[:, :self.n_frames, :]
        imp = imp.view(-1, self.n_frames, 257)
        prev = torch.zeros(batch_size * n_events, 257, dtype=torch.complex128, device=device)
        
        resonances = self.to_resonances(latent).view(-1, 257 * 2, self.n_frames)
        mags, phases = resonances[:, :257, :], resonances[:, 257:, :]
        
        # mags = self.base_resonance + (torch.sigmoid(mags) * self.res_factor)
        # phase_dither = torch.sigmoid(phases) * torch.zeros_like(phases).uniform_(0, 1) * self.group_delay[None, :, None]
        
        out_frames = []
        
        for i in range(self.n_frames):
            current_input = imp[:, i, :]
            
            if i > 0:
                prev_frame = out_frames[i - 1].view(*prev.shape)
            else:
                prev_frame = prev
            
            current_res = mags[:, :, i]
            current_phase = phases[:, :, i]
            current = torch.complex(current_res, current_phase)
            
            # prev_mag = torch.abs(prev_frame)
            # prev_phase = torch.angle(prev_frame)
            # res_mag = prev_mag * current_res
            # res_phase = prev_phase + current_phase
            # prev = res_mag * torch.exp(1j * res_phase)
            
            # real = res_mag * torch.cos(res_phase)
            # imag = res_mag * torch.sin(res_phase)            
            # prev = torch.complex(real, imag)
            
            next_frame = current_input + (prev_frame * current)
            
            out_frames.append(next_frame[:, None, :])
        
        out_frames = torch.cat(out_frames, dim=1)
        out_frames = torch.fft.irfft(out_frames, dim=-1)
        out_frames = overlap_add(out_frames[:, None, :, :], apply_window=False)[..., :exp.n_samples]
        out_frames = out_frames.view(batch_size, n_events, resonance_size)
        return out_frames
        

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
        mx = F.avg_pool1d(mx, 9, 1, 4)
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
        # waves = make_waves(512, f0s, int(samplerate))
        # waves = waves * torch.hamming_window(512)[None, ...]
        
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
        

        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(
            self.refractory_period, power=10, device=device))

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
        # mixed = unit_norm(mixed)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps
        final = mixed

        # final = F.pad(mixed, (0, n_samples - mixed.shape[-1]))
        # up = torch.zeros(final.shape[0], n_events, n_samples, device=final.device)
        # up[:, :, ::256] = packed
        # final = fft_convolve(final, up)[..., :n_samples]
        
        
        final = self.verb.forward(dense, final)
        # final = unit_norm(final, dim=-1)

        return final, imp

    def forward(self, x):
        encoded = self.encode(x)
        
        dense = torch.mean(encoded, dim=-1)
        mean = self.to_context_mean(dense)
        std = self.to_context_std(dense)
        dense = mean + (torch.zeros_like(mean).normal_(0, 1) * std)
        
        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        

        final, imp = self.generate(encoded, one_hot, packed, dense)
        return final, encoded, imp
    
    
    def random_generation(self, exponent=2):
        with torch.no_grad():
            # generate context latent
            z = torch.zeros(1, context_dim).normal_(0, 1)
            
            # generate events
            events = torch.zeros(1, 1, 4096, 128).uniform_(0, 10) ** exponent
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


# def dict_op(
#         a: Dict[int, torch.Tensor],
#         b: Dict[int, torch.Tensor],
#         op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[int, torch.Tensor]:
#     return {k: op(v, b[k]) for k, v in a.items()}


# def multiband_transform(x: torch.Tensor):
#     bands = fft_frequency_decompose(x, 512)
#     d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
#     d2 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
#     return dict(**d1, **d2)
    

# def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

#     target = multiband_transform(target)

#     full = torch.sum(recon, dim=1, keepdim=True)
#     full = multiband_transform(full)

#     residual = dict_op(target, full, lambda a, b: a - b)
    
    
#     loss = 0
    
#     # for i in range(n_events):
#     i = np.random.randint(0, n_events)
#     ch = recon[:, i: i + 1, :]
    
#     ch = multiband_transform(ch)

#     t = dict_op(residual, ch, lambda a, b: a + b)

#     diff = dict_op(ch, t, lambda a, b: a - b)
#     loss = loss + sum([torch.abs(y).sum() for y in diff.values()])

#     return loss


n_iterations  = 16

def train(batch, i):
    
    residual = batch.clone()
    
    batch_number = i
    
    total_loss = 0
    
    recon = torch.zeros_like(batch)
    print('===================================')
    
    for i in range(n_iterations):
        optim.zero_grad()
        
        residual = residual.clone().detach()
        
        iteration_number = i
        atoms, encoded, imp = model.forward(residual)
        
        # find the best location
        fm = fft_convolve(atoms, residual)[..., :exp.n_samples]
        sparse, packed, one_hot = sparsify2(fm, n_to_keep=n_events)
        positioned = fft_convolve(packed, atoms)
        
        recon = recon + positioned
        
        start_norm = torch.norm(stft(residual, 2048, 256, pad=True), dim=(-1, -2))
        residual = (residual - positioned)
        end_norm = torch.norm(stft(residual, 2048, 256, pad=True), dim=(-1, -2))
        loss = (end_norm / (start_norm + 1e-8)).mean()
        
        total_loss = total_loss + loss.clone()
        loss.backward()
        
        print(f'{batch_number} - {iteration_number}: {loss.item()}')
        optim.step()
    
    recon = max_norm(recon)
    return total_loss, recon


# def make_conjure(experiment: BaseExperimentRunner):
#     @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
#     def encoded(x: torch.Tensor):
#         x = x[:, None, :, :]
#         x = F.max_pool2d(x, (16, 8), (16, 8))
#         x = x.view(x.shape[0], x.shape[2], x.shape[3])
#         x = x.data.cpu().numpy()[0]
#         return x

#     return (encoded,)


@readme
class IterativeDecomposition(BaseExperimentRunner):
    # encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None, load_weights=True, save_weights=False, model=model):
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            load_weights=load_weights, 
            save_weights=save_weights, 
            model=model)

    # def run(self):
    #     for i, item in enumerate(self.iter_items()):
    #         item = item.view(-1, 1, n_samples)
    #         l, r, e = train(item, i)


    #         self.real = item
    #         self.fake = r
    #         self.encoded = e
    #         print(i, l.item())
    #         self.after_training_iteration(l, i)
