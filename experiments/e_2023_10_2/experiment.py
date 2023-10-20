from types import FunctionType
from typing import Callable, Dict
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from angle import windowed_audio
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.overlap_add import overlap_add
from modules.ddsp import NoiseModel
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from scipy.signal import square, sawtooth


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_events = 32
context_dim = 16
impulse_size = 4096
resonance_size = 32768

base_resonance = 0.2
apply_group_delay_to_dither = True
use_dense_context = False


def make_memory_context(encoding: torch.Tensor, exponent: float):
    batch, channels, time = encoding.shape
    memory = torch.linspace(0, 1, time, device=encoding.device) ** exponent
    encoding_spec = torch.fft.rfft(encoding, dim=-1)
    memory_spec = torch.fft.rfft(memory[None, None, :], dim=-1)
    spec = encoding_spec * memory_spec
    context = torch.fft.irfft(spec)
    return context


class STFTRecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.step = window_size // 2
        self.n_coeffs = window_size // 2 + 1
        self.resonance_samples = resonance_samples
        self.n_frames = resonance_samples // self.step
        self.encoding_channels = encoding_channels


        self.register_buffer('group_delay', torch.linspace(0, np.pi, self.n_coeffs))

        self.resonance_factor = 0.99 - base_resonance


        # TODO: Does this work better with just simple linear projections?
        self.to_initial = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_resonance = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        self.to_phase_dither = LinearOutputStack(
            channels, layers=3, out_channels=self.n_coeffs, in_channels=latent_dim, norm=nn.LayerNorm((channels,)))
        
    
    def forward(self, latents):
        batch_size = latents.shape[0]

        initial = self.to_initial(latents)
        # initial = torch.relu(initial)

        res = base_resonance + (torch.sigmoid(self.to_resonance(latents)) * self.resonance_factor)
        dither = torch.sigmoid(self.to_phase_dither(latents)) 

        if apply_group_delay_to_dither:
            dither = dither * self.group_delay[None, None, :]


        # print(res.shape, dither.shape, initial.shape)

        # first_frame = initial[:, None, :]

        # Non-recurrent

        full_res = res[..., None].repeat(1, 1, 1, self.n_frames)
        full_res = torch.log(1e-12 + full_res)
        full_res = torch.cumsum(full_res, dim=-1)
        full_res = torch.exp(full_res)
        # full_res = self.to_env(latents.view(-1, self.latent_dim)).view(-1, n_events, self.n_frames, self.n_coeffs).permute(0, 1, 3, 2)
        # envelope should be in [0 - 1]
        # full_res = torch.sigmoid(full_res)

        mags = full_res * initial[..., None]
        mags = mags.permute(0, 1, 3, 2)

        full_delay = self.group_delay[None, None, :, None].repeat(1, 1, 1, self.n_frames)
        full_delay = torch.cumsum(full_delay, dim=-1)

        if apply_group_delay_to_dither:
            full_delay = full_delay + (torch.zeros_like(full_delay).uniform_(-np.pi, np.pi) * dither[:, :, :, None])
        
        phases = full_delay
        phases = phases.permute(0, 1, 3, 2)
        # end non-recurrent
        frames = torch.complex(
            mags * torch.cos(phases),
            mags * torch.sin(phases)
        )

        windowed = torch.fft.irfft(frames, dim=-1, norm='ortho')
        
        
        samples = overlap_add(windowed, apply_window=True)[..., :self.resonance_samples]
        # samples = max_norm(samples, dim=-1)

        assert samples.shape == (batch_size, self.encoding_channels, self.resonance_samples)
        return samples, full_res


def make_waves(n_samples, f0s, samplerate):
    sawtooths = []
    squares = []
    triangles = []

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
    
    sawtooths = np.concatenate(sawtooths, axis=0)
    squares = np.concatenate(squares, axis=0)
    triangles = np.concatenate(triangles, axis=0)
    return sawtooths, squares, triangles


class RecurrentResonanceModelWithComplexWaveforms(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.resonance_samples = resonance_samples
        self.filter_coeffs = window_size // 2 + 1

        n_atoms = 1024
        self.n_frames = resonance_samples // (window_size // 2)
        self.res_factor = (1 - base_resonance) * 0.9

        n_f0s = n_atoms // 3
        f0s = np.linspace(40, 4000, n_f0s)
        sq, st, tri = make_waves(self.resonance_samples, f0s, int(exp.samplerate))


        atoms = np.concatenate([sq, st, tri], axis=0)
        bank = torch.from_numpy(atoms).float()
        n_atoms = bank.shape[0]

        # TODO: should this be multi-band, or parameterized differently?
        # What about a convolutional model to produce impulse responses?

        # band = zounds.FrequencyBand(40, exp.samplerate.nyquist)
        # scale = zounds.LinearScale(band, n_atoms)
        # bank = morlet_filter_bank(
        #     exp.samplerate, resonance_samples, scale, 0.01, normalize=True).real.astype(np.float32)
        # bank = torch.from_numpy(bank).view(n_atoms, resonance_samples)


        self.register_buffer('atoms', bank)

        # we don't want the filter to dominate the spectral shape, just roll off highs, mostly
        self.to_filter = nn.Linear(latent_dim, 32)
        self.selection = nn.Linear(latent_dim, n_atoms)
        self.to_momentum = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):

        # compute resonance/sustain
        # computing each frame independently makes something like "damping" possible
        mom = base_resonance + \
            (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
        mom = torch.log(1e-12 + mom)
        # mom = mom.repeat(1, 1, self.n_frames)
        mom = torch.cumsum(mom, dim=-1)
        mom = torch.exp(mom)
        new_mom = mom

        # compute resonance shape/pattern
        sel = torch.softmax(self.selection(x), dim=-1)
        atoms = self.atoms
        res = sel @ atoms

        # compute low-pass filter
        filt = self.to_filter(x)
        filt = F.interpolate(filt, size=self.filter_coeffs, mode='linear')
        filt = torch.sigmoid(filt)

        windowed = windowed_audio(res, self.window_size, self.window_size // 2)
        windowed = unit_norm(windowed, dim=-1)
        windowed = windowed * new_mom[..., None]
        windowed = torch.fft.rfft(windowed, dim=-1)
        windowed = windowed * filt[:, :, None, :]
        windowed = torch.fft.irfft(windowed, dim=-1)
        windowed = overlap_add(
            windowed, apply_window=False)[..., :self.resonance_samples]

        return windowed, new_mom


class RecurrentResonanceModel(nn.Module):
    def __init__(self, encoding_channels, latent_dim, channels, window_size, resonance_samples):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.latent_dim = latent_dim
        self.channels = channels
        self.window_size = window_size
        self.resonance_samples = resonance_samples

        n_atoms = 1024
        self.n_frames = resonance_samples // (window_size // 2)
        self.res_factor = (1 - base_resonance) * 0.9

        # TODO: should this be multi-band, or parameterized differently?
        # What about a convolutional model to produce impulse responses?

        band = zounds.FrequencyBand(40, exp.samplerate.nyquist)
        scale = zounds.LinearScale(band, n_atoms)
        bank = morlet_filter_bank(
            exp.samplerate, resonance_samples, scale, 0.01, normalize=True).real.astype(np.float32)
        bank = torch.from_numpy(bank).view(n_atoms, resonance_samples)

        # bank = torch.zeros_like(bank).uniform_(-1, 1)
        # self.atoms = nn.ParameterDict({str(k): v for k, v in fft_frequency_decompose(bank.view(1, n_atoms, resonance_samples), 512).items()})

        self.atoms = nn.Parameter(bank)
        # self.register_buffer('atoms', bank)

        # self.to_res = ConvUpsample(
        #     latent_dim, channels, 4, end_size=resonance_samples, mode='nearest', out_channels=1, from_latent=True, batch_norm=True)

        self.selection = nn.Linear(latent_dim, n_atoms)
        self.to_momentum = nn.Linear(latent_dim, self.n_frames)

    def forward(self, x):

        # compute resonance/sustain
        mom = base_resonance + \
            (torch.sigmoid(self.to_momentum(x)) * self.res_factor)
        mom = torch.log(1e-12 + mom)
        # mom = mom.repeat(1, 1, self.n_frames)
        mom = torch.cumsum(mom, dim=-1)
        mom = torch.exp(mom)
        new_mom = mom

        # compute resonance shape/pattern
        sel = torch.softmax(self.selection(x), dim=-1)
        atoms = self.atoms
        res = sel @ atoms

        windowed = windowed_audio(res, self.window_size, self.window_size // 2)
        windowed = unit_norm(windowed, dim=-1)
        windowed = windowed * new_mom[..., None]
        windowed = overlap_add(
            windowed, apply_window=False)[..., :self.resonance_samples]

        return windowed, new_mom


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
            self.n_frames * 32,
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


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_period = nn.Linear(257, 8)

        self.embed_cond = nn.Conv1d(4096, 256, 1, 1, 0)
        self.embed_spec = nn.Conv1d(1024, 256, 1, 1, 0)
        self.embed_context = nn.Linear(context_dim, 256)

        self.net = nn.Sequential(

            # 128
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 128
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 1, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 128
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 1, 3),
                nn.LeakyReLU(0.2),
            ),


            nn.Conv1d(256, 1, 1, 1, 0)
        )

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, cond, audio):
        batch_size = cond.shape[0]


        spec = stft(audio, 2048, 256, pad=True).view(
            batch_size, 128, 1025)[..., :1024].permute(0, 2, 1)

        spec = self.embed_spec(spec)
        cond = self.embed_cond(cond)
        x = cond + spec
        j = self.net(x)
        return j


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

        self.res = RecurrentResonanceModel(
            n_events, 256, 64, 1024, resonance_samples=resonance_size)

        # self.res = STFTRecurrentResonanceModel(
        #     n_events, 256, 64, 1024, resonance_samples=resonance_size)

        # self.res = RecurrentResonanceModelWithComplexWaveforms(
        #     n_events, 256, 64, 1024, resonance_samples=resonance_size
        # )

        self.mix = GenerateMix(256, 128, n_events)
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

    def generate(self, encoded, one_hot, packed, dense):
        if use_dense_context:
            ctxt = self.from_context(dense)
        else:
            # ctxt = make_memory_context(encoded, 4) # (batch, encoding_channels, time)
            # print(ctxt.shape)
            ctxt = torch.sum(encoded, dim=-1)
            dense = self.embed_memory_context(ctxt) # (batch, context_dim)

        # ctxt is a single vector
        ce = self.embed_context(ctxt)
        
        # one hot is n_events vectors
        oh = self.embed_one_hot(one_hot)

        embeddings = ce[:, None, :] + oh
        

        # generate...

        # impulses
        imp = self.imp.forward(embeddings)
        padded = F.pad(imp, (0, resonance_size - impulse_size))

        # resonances
        res, env = self.res.forward(embeddings)

        # mixes
        mx = self.mix.forward(embeddings)

        conv = fft_convolve(padded, res)[..., :resonance_size]

        stacked = torch.cat([padded[..., None], conv[..., None]], dim=-1)
        mixed = stacked @ mx.view(-1, n_events, 2, 1)
        mixed = mixed.view(-1, n_events, resonance_size)
        # mixed = unit_norm(mixed, dim=-1)

        amps = torch.abs(self.to_amp(embeddings))
        mixed = mixed * amps

        final = F.pad(mixed, (0, exp.n_samples - mixed.shape[-1]))
        up = torch.zeros(final.shape[0], n_events,
                         exp.n_samples, device=final.device)
        up[:, :, ::256] = packed

        final = fft_convolve(final, up)[..., :exp.n_samples]

        # final = torch.sum(final, dim=1, keepdim=True)

        final = self.verb.forward(dense, final)


        return final, env

    def forward(self, x):
        encoded = self.encode(x)

        # TODO: This should be a mean/std instead
        # mean + (std * N(0, 1))
        non_sparse = torch.mean(encoded, dim=-1)

        non_sparse_mean = self.to_context_mean(non_sparse)
        non_sparse_std = self.to_context_std(non_sparse)
        non_sparse = non_sparse_mean + \
            (torch.zeros_like(non_sparse_mean).normal_(0, 1) * non_sparse_std)

        encoded, packed, one_hot = sparsify2(encoded, n_to_keep=n_events)

        final, env = self.generate(encoded, one_hot, packed, dense=non_sparse)
        return final, encoded, env, non_sparse


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)

try:
    model.load_state_dict(torch.load('siam.dat'))
    print('LOADED SIAM WEIGHTS')
except IOError:
    print('No weights saved')


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
    


def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):

    # target = stft(target, ws, ss, pad=True)
    target = multiband_transform(target)

    full = torch.sum(recon, dim=1, keepdim=True)
    # full = stft(full, ws, ss, pad=True)
    full = multiband_transform(full)

    # residual = target - full
    residual = dict_op(target, full, lambda a, b: a - b)

    loss = 0

    for i in range(n_events):
        ch = recon[:, i: i + 1, :]
        # ch = stft(ch, ws, ss, pad=True)
        ch = multiband_transform(ch)

        # t = residual + ch
        t = dict_op(residual, ch, lambda a, b: a + b)

        # loss = loss + F.mse_loss(ch, t.clone().detach())
        diff = dict_op(ch, t, lambda a, b: a - b)
        loss = loss + sum([torch.abs(y).sum() for y in diff.values()])

        # loss = loss + torch.abs(ch - t.clone().detach()).sum()

    return loss


def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()

    # with torch.no_grad():
    #     feat = exp.perceptual_feature(batch)

    if True or i % 2 == 0:
        recon, encoded, env, context = model.forward(batch)

        recon_summed = torch.sum(recon, dim=1, keepdim=True)

        # compute spec loss
        # fake_spec = stft(recon_summed, 2048, 256, pad=True)
        # real_spec = stft(batch, 2048, 256, pad=True)
        # spec_loss = F.mse_loss(fake_spec, real_spec)

        loss = single_channel_loss(batch, recon)

        # make sure random encodings also sound reasonable
        with torch.no_grad():
            random_encoding = torch.zeros_like(encoded).uniform_(-1, 1)
            e, p, oh = sparsify2(random_encoding, n_to_keep=n_events)
            dense = torch.zeros(e.shape[0], context_dim, device=e.device).normal_(0, 1)
            fake, _ = model.generate(e, oh, p, dense=dense)
            fake_summed = torch.sum(fake, dim=1, keepdim=True)
        
        # if np.random.rand() > 0.5:
        #     print('PREVIEWING FAKE')
        #     recon_summed = fake_summed
        #     encoded = e
        
        # fake = fake.sum(dim=1, keepdim=True)
        # j = disc.forward(encoded.clone().detach(), recon_summed)[..., None]
        # fj = disc.forward(e.clone().detach(), fake)[..., None]
        # j = torch.cat([j, fj], dim=-1)

        # adv_loss = (torch.abs(1 - j).mean() * 1)

        # loss = loss + adv_loss

        loss.backward()
        optim.step()

        print('GEN', loss.item())

        recon = max_norm(recon_summed)
        encoded = max_norm(encoded)
        return loss, recon, encoded
    else:
        with torch.no_grad():
            recon, encoded, env, context = model.forward(batch)
            recon_summed = torch.sum(recon, dim=1, keepdim=True)

        rj = disc.forward(encoded, batch)
        fj = disc.forward(encoded, recon_summed)
        loss = (torch.abs(1 - rj) + torch.abs(0 - fj)).mean()
        loss.backward()
        disc_optim.step()
        print('DISC', loss.item())
        return None, None, None


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
class GraphRepresentation(BaseExperimentRunner):

    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)

            if l is None:
                continue

            if i % 1000 == 0:
                torch.save(model.state_dict(), 'siam.dat')

            self.real = item
            self.fake = r
            self.encoded = e
            print(i, l.item())
            self.after_training_iteration(l)
