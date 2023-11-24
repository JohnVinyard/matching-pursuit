from typing import Callable, Dict

import numpy as np
import torch
import zounds
from conjure import numpy_conjure, SupportedContentType
from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F

from modules.angle import windowed_audio
from config.experiment import Experiment
from modules.ddsp import NoiseModel
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.overlap_add import overlap_add
from modules.pif import fft_based_pif
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from modules.upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=256,
    kernel_size=512,
    windowed_pif=True)

n_events = 64
context_dim = 16
impulse_size = 4096
resonance_size = 32768

# base_resonance = 0.02
# resonance_selection_method = lambda x: torch.relu(x)



class ResonanceModel(nn.Module):
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
        
        # self.filters = nn.ModuleList([
        #     nn.Linear(latent_dim, self.coarse_coeffs) for _ in range(self.n_piecewise)
        # ])
        
        # self.to_filter = ConvUpsample(
        #     latent_dim,
        #     channels,
        #     start_size=8,
        #     end_size=32,
        #     mode='nearest',
        #     out_channels=self.coarse_coeffs,
        #     from_latent=True,
        #     batch_norm=True
        # )
        self.to_filter = nn.Linear(latent_dim, 257)
        
        self.to_mixture = ConvUpsample(
            latent_dim, 
            channels, 
            start_size=8, 
            end_size=32, 
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
        
        # TODO: There should be another collection for just resonances
        resonances = []
        convs = []
        
        imp = F.pad(impulse, (0, self.resonance_size - impulse.shape[-1]))
        
        
        decay = torch.sigmoid(self.decay(latent))
        decay = self.base_resonance + (decay * self.res_factor)
        decay = torch.log(1e-12 + decay)
        decay = torch.cumsum(decay, dim=-1)
        decay = torch.exp(decay)
        decay = F.interpolate(decay, size=self.resonance_size, mode='linear')
        
        filt = self.to_filter(latent)
        filt = torch.sigmoid(filt)
        filt = filt.view(-1, n_events, 1, 257)
        # filt = self.to_filter(latent).view(-1, self.coarse_coeffs, self.n_frames).permute(0, 2, 1)
        # filt = torch.sigmoid(filt)
        # filt = F.interpolate(filt, size=257, mode='linear')
        # filt = filt.view(-1, 32, 257)
        # filt = F.avg_pool1d(filt, 7, 1, 3)
        # filt = filt.view(-1, )
        
        
        for i in range(self.n_piecewise):
            
            # choose a linear combination of resonances
            sel = self.selections[i].forward(latent)
            sel = torch.softmax(sel, dim=-1)
            res = sel @ self.atoms
            res = res * decay
            
            filtered_res = res
            
            windowed = windowed_audio(filtered_res, 512, 256)
            windowed = unit_norm(windowed, dim=-1)
            windowed = torch.fft.rfft(windowed, dim=-1)
            windowed = windowed * filt
            windowed = torch.fft.irfft(windowed)
            filtered_res = overlap_add(windowed, apply_window=False)[..., :self.resonance_size]\
                .view(-1, n_events, self.resonance_size)
            
            resonances.append(filtered_res[:, None, :, :])
            
            conv = fft_convolve(filtered_res, imp)
            
            convs.append(conv[:, None, :, :])
        
        # TODO: Concatenate both the pure resonances and the convolved audio
        resonances = torch.cat(resonances, dim=1)
        convs = torch.cat(convs, dim=1)
        
        # produce a linear mixture-over time
        mx = self.to_mixture(latent)
        mx = F.interpolate(mx, size=self.resonance_size, mode='linear')
        mx = F.avg_pool1d(mx, 9, 1, 4)
        mx = torch.softmax(mx, dim=1)
        mx = mx.view(-1, n_events, self.n_piecewise, self.resonance_size).permute(0, 2, 1, 3)
        
        final_res = (mx * resonances).sum(dim=1)
        final_convs = (mx * convs).sum(dim=1)
        
        final_mx = self.final_mix(latent)
        final_mx = torch.softmax(final_mx, dim=-1)
        
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



# class RecurrentResonanceModelWithComplexWaveforms(nn.Module):
#     def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
#         super().__init__()
#         self.encoding_channels = encoding_channels
#         self.latent_dim = latent_dim
#         self.channels = channels
#         self.window_size = window_size
#         self.resonance_samples = resonance_samples
#         self.filter_coeffs = window_size // 2 + 1

#         n_atoms = 512

#         self.n_frames = resonance_samples // (window_size // 2)
#         self.res_factor = (1 - base_resonance) * 0.95

#         bank = torch.zeros(n_atoms, self.resonance_samples)
#         for i in range(n_atoms):
#             bank[i, ::(i + 1)] = 1

#         # n_f0s = n_atoms // 4
#         # f0s = np.linspace(40, 4000, n_f0s)
#         # sq, st, tri, sines = make_waves(self.resonance_samples, f0s, int(exp.samplerate))
#         #
#         # atoms = np.concatenate([sq, st, tri, sines], axis=0)
#         # bank = torch.from_numpy(atoms).float()
#         # n_atoms = bank.shape[0]

#         # TODO: should this be multi-band, or parameterized differently?
#         # What about a convolutional model to produce impulse responses?

#         # band = zounds.FrequencyBand(40, exp.samplerate.nyquist)
#         # scale = zounds.LinearScale(band, n_atoms)
#         # bank = morlet_filter_bank(
#         #     exp.samplerate, resonance_samples, scale, 0.01, normalize=True).real.astype(np.float32)
#         # bank = torch.from_numpy(bank).view(n_atoms, resonance_samples)

#         self.register_buffer('atoms', bank)

#         # we don't want the filter to dominate the spectral shape, just roll off highs, mostly
#         # self.to_filter = nn.Linear(latent_dim, 32)
#         self.to_filter = ConvUpsample(
#             latent_dim, channels, 4, end_size=self.n_frames, mode='learned', out_channels=32, from_latent=True,
#             batch_norm=True)
#         self.selection = nn.Linear(latent_dim, n_atoms)
#         self.to_momentum = nn.Linear(latent_dim, self.n_frames)

#     def forward(self, x):
#         # compute resonance/sustain
#         # computing each frame independently makes something like "damping" possible
#         mom = base_resonance + \
#               (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
#         mom = torch.log(1e-12 + mom)
#         # mom = mom.repeat(1, 1, self.n_frames)
#         mom = torch.cumsum(mom, dim=-1)
#         mom = torch.exp(mom)
#         new_mom = mom

#         # compute resonance shape/pattern
#         # sel = torch.softmax(self.selection(x), dim=-1)
#         sel = resonance_selection_method(self.selection(x))
#         atoms = self.atoms
#         res = sel @ atoms

#         # compute low-pass filter time-series
#         filt = self.to_filter(x).view(-1, 32, self.n_frames).permute(0, 2, 1)
#         filt = F.interpolate(filt, size=self.filter_coeffs, mode='linear').view(-1, n_events, self.n_frames,
#                                                                                 self.filter_coeffs)
#         filt = torch.sigmoid(filt)

#         windowed = windowed_audio(res, self.window_size, self.window_size // 2)
#         windowed = unit_norm(windowed, dim=-1)
#         windowed = windowed * new_mom[..., None]
#         windowed = torch.fft.rfft(windowed, dim=-1)
#         windowed = windowed * filt
#         windowed = torch.fft.irfft(windowed, dim=-1)
#         windowed = overlap_add(
#             windowed, apply_window=False)[..., :self.resonance_samples]

#         return windowed, new_mom


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

    def forward(self, x):
        x = self.to_frames(x)
        x = self.noise_model(x)
        return x.view(-1, n_events, self.n_samples)


class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

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

        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)

    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}

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
        waves = make_waves(resonance_size, f0s, int(exp.samplerate))
        
        self.res = ResonanceModel(
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

        self.to_context_mean = nn.Linear(4096, context_dim)
        self.to_context_std = nn.Linear(4096, context_dim)
        self.embed_memory_context = nn.Linear(4096, context_dim)

        self.from_context = nn.Linear(context_dim, 4096)

        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(
            self.refractory_period, power=10, device=device))

        self.apply(lambda x: exp.init_weights(x))

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

    def generate(self, encoded, one_hot, packed):
        ctxt = torch.sum(encoded, dim=-1)
        dense = self.embed_memory_context(ctxt)  # (batch, context_dim)

        # ctxt is a single vector
        ce = self.embed_context(ctxt)

        # one hot is n_events vectors
        oh = self.embed_one_hot(one_hot)

        embeddings = ce[:, None, :] + oh

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        # padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        mixed = self.res.forward(embeddings, imp)

        # TODO: resonance model eats everything but the application of amplitudes
        # mixes
        # mx = self.mix.forward(embeddings)
        
        # conv = fft_convolve(padded, res)[..., :resonance_size]

        # stacked = torch.cat([padded[..., None], conv[..., None], res[..., None]], dim=-1)
        # mixed = stacked @ mx.view(-1, n_events, 3, 1)
        # mixed = mixed.view(-1, n_events, resonance_size)

        amps = torch.abs(self.to_amp(embeddings))
        
        mixed = mixed * amps

        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events, exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        final = self.verb.forward(dense, final)

        return final, imp

    def forward(self, x):
        encoded = self.encode(x)

        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)
        encoded = torch.relu(encoded)
        

        final, imp = self.generate(encoded, one_hot, packed)
        return final, encoded, imp


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def dict_op(
        a: Dict[int, torch.Tensor],
        b: Dict[int, torch.Tensor],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[int, torch.Tensor]:
    return {k: op(v, b[k]) for k, v in a.items()}


def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d2 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    return dict(**d1, **d2)
    
    # pif = exp.perceptual_feature(x)
    # pif = fft_based_pif(x, 256, 64)
    # return { 'pif': pif }


# def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

#     target = multiband_transform(target)

#     full = torch.sum(recon, dim=1, keepdim=True)
#     full = multiband_transform(full)

#     residual = dict_op(target, full, lambda a, b: a - b)

#     loss = 0

#     for i in range(n_events):
#         ch = recon[:, i: i + 1, :]
#         ch = multiband_transform(ch)

#         t = dict_op(residual, ch, lambda a, b: a + b)

#         diff = dict_op(ch, t, lambda a, b: a - b)
#         loss = loss + sum([torch.abs(y).sum() for y in diff.values()])


#     return loss


def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

    target = multiband_transform(target)

    full = torch.sum(recon, dim=1, keepdim=True)
    full = multiband_transform(full)

    residual = dict_op(target, full, lambda a, b: a - b)
    
    loss = 0

    # for i in range(n_events):
    i = np.random.randint(0, n_events)
    ch = recon[:, i: i + 1, :]
    ch = multiband_transform(ch)

    t = dict_op(residual, ch, lambda a, b: a + b)

    diff = dict_op(ch, t, lambda a, b: a - b)
    loss = loss + sum([torch.abs(y).sum() for y in diff.values()])

    return loss

def train(batch, i):
    optim.zero_grad()

    # b = batch.shape[0]
    
    recon, encoded, imp = model.forward(batch)
    
    energy_loss = torch.abs(imp).sum(dim=-1).mean() * 1e-5
    print('ENERGY LOSS', energy_loss.item())
    
    
    recon_summed = torch.sum(recon, dim=1, keepdim=True)

    loss = (single_channel_loss(batch, recon) * 1e-6) + energy_loss
    

    loss.backward()
    optim.step()

    print('GEN', loss.item())

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
class GraphRepresentation4(BaseExperimentRunner):
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

            # if l is None:
            #     continue

            # if i % 1000 == 0:
            #     torch.save(model.state_dict(), 'siam.dat')

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l, i)
