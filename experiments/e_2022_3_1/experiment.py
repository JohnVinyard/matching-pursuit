from config.dotenv import Config
from itertools import chain
import torch
import numpy as np
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank
from torch.nn import functional as F
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.latent_loss import latent_loss
from modules.linear import LinearOutputStack
from modules.mixer import Mixer
from modules.psychoacoustic import PsychoacousticFeature
from upsample import ConvUpsample
from util import device
from train import gan_cycle
from random import choice

from modules.multiresolution import BandEncoder
from util.readmedocs import readme
from util.weight_init import make_initializer

network_channels = 64
n_samples = 2**14

feature = PsychoacousticFeature(
    kernel_sizes=[128] * 6
).to(device)

init_func = make_initializer(0.1)


def compute_feature_dict(x):
    x = {k: v.contiguous() for k, v in x.items()}
    return feature.compute_feature_dict(
        x,
        constant_window_size=128
    )


def long_sample_stream(batch_size, overfit=False, normalize=False):
    stream = batch_stream(
        Config.audio_path(),
        '*.wav',
        batch_size,
        n_samples * 2,
        overfit=overfit)

    for batch in stream:

        if normalize:
            batch /= (batch.max(axis=-1, keepdims=True) + 1e-12)
        
        first, second = batch[..., :n_samples], batch[..., n_samples:]

        first = torch.from_numpy(first).to(device).view(-1, 1, n_samples)
        first_bands = fft_frequency_decompose(first, 512)
        first_feat = compute_feature_dict(first_bands)

        second = torch.from_numpy(second).to(device).view(-1, 1, n_samples)
        second_bands = fft_frequency_decompose(second, 512)
        second_feat = compute_feature_dict(second_bands)

        yield first_bands, first_feat, second_bands, second_feat


def sample_stream(batch_size, overfit=False, normalize=False):

    stream = batch_stream(
        Config.audio_path(),
        '*.wav',
        batch_size,
        n_samples,
        overfit=overfit)

    for batch in stream:

        if normalize:
            batch /= (batch.max(axis=-1, keepdims=True) + 1e-12)

        batch = torch.from_numpy(batch).to(device).view(-1, 1, n_samples)
        bands = fft_frequency_decompose(batch, 512)
        feat = compute_feature_dict(bands)
        yield bands, feat


class BranchEncoder(nn.Module):
    def __init__(self, periodicity):
        super().__init__()
        self.periodicity = periodicity
        self.p = BandEncoder(network_channels, periodicity)
        self.r = LinearOutputStack(network_channels, 2, in_channels=512)
        self.t = Mixer(network_channels, 32, 3)

    def forward(self, x):
        x = x.view(-1, network_channels, 32, self.periodicity)
        x = self.p(x).permute(0, 2, 1)  # (-1, 32, 512)
        x = self.r(x)
        x = self.t(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {str(k): BranchEncoder(65) for k in feature.band_sizes})
        self.r = LinearOutputStack(network_channels, 2, in_channels=384)
        self.t = Mixer(network_channels, 32, 3)
        self.e = LinearOutputStack(network_channels, 2)
        self.apply(init_func)

    def forward(self, x):
        d = {k: self.encoders[str(k)](x[k]) for k in x.keys()}
        x = torch.cat(list(d.values()), dim=-1)
        x = self.r(x)
        x = self.t(x)
        x = self.e(x)

        x = x[:, -1, :]

        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict(
            {str(k): BandUpsample(k) for k in feature.band_sizes})
        self.up = ConvUpsample(
            network_channels, network_channels, 4, 32, 'learned', network_channels)
        self.apply(init_func)

    def forward(self, x):
        x = self.up(x)
        x = x.permute(0, 2, 1)
        return {int(k): self.decoders[str(k)](x) for k in self.decoders.keys()}


class BandUpsample(nn.Module):
    """
    Expand and finalize an individual frequency band
    """

    def __init__(self, band_size):
        super().__init__()
        self.band_size = band_size
        self.channels = network_channels
        self.t = Mixer(network_channels, 32, 3)

        self.t = nn.Sequential(
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
        )

        self.osc = OscillatorBank(
            self.channels,
            32,
            self.band_size,
            activation=torch.sigmoid,
            # encourage sparser activations while remaining linear
            amp_activation=torch.abs,
            return_params=False,
            constrain=True,
            # linear frequency seems to be integral
            log_frequency=False,
            lowest_freq=0.05 if band_size == 512 else 0.01,
            sharpen=False,
            compete=False)

        self.noise = NoiseModel(
            input_channels=self.channels,
            input_size=32,
            # don't give noise sufficient frequency resolution to
            # approximate oscillators, but give it enough to potentially
            # guide oscillators (if that's a thing?)
            n_noise_frames=64,
            n_audio_samples=self.band_size,
            channels=self.channels,
            activation=lambda x: x,
            squared=False,
            # don't mess with the DC frequency for the lowest band
            mask_after=1 if band_size == 512 else None)

    def forward(self, x):
        # x = self.t(x)
        x = x.permute(0, 2, 1)
        x = self.t(x)
        harm = self.osc(x)
        noise = self.noise(x)
        return harm, noise

embedder = Encoder().to(device)
embedder_optim = Adam(embedder.parameters(), lr=1e-4, betas=(0, 0.9))

generator = Decoder().to(device)
generator_optim = Adam(generator.parameters(), lr=1e-4, betas=(0, 0.9))



def train_gen(batch):
    generator_optim.zero_grad()

    with torch.no_grad():
        embedded = embedder(batch)

    decoded = generator(embedded)
    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)
    re_embedded = embedder(fake_feat)
    loss = F.mse_loss(re_embedded, embedded)
    print('G', loss.item())
    loss.backward()
    generator_optim.step()
    return decoded, re_embedded


def train_disc(a, b):
    embedder_optim.zero_grad()
    a = embedder(a)
    b = embedder(b)

    # embeddings from adjacent segments should be near one another
    embedding_loss = F.mse_loss(a, b)

    # embeddings should have feature-wise zero-mean and unit variance
    # and be uncorrelated
    ll = latent_loss(a)

    loss = embedding_loss + ll
    loss.backward()
    embedder_optim.step()
    print('D', loss.item())
    return a


@readme
class MultiresolutionAutoencoderWithActivationRefinements17(object):
    def __init__(self, batch_size=2, overfit=False):
        super().__init__()
        self.feature = feature
        self.n_samples = 2**14
        self.batch_size = batch_size
        self.overfit = overfit
        self.samplerate = zounds.SR22050()

        self.bands = None
        self.decoded = None
        self.encoded = None

        self.freq = None
        self.noise = None

    def real(self):
        with torch.no_grad():
            single = {k: v[0].view(1, 1, -1) for k, v in self.bands.items()}
            audio = fft_frequency_recompose(single, self.n_samples)
            return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()

    def real_spec(self):
        return np.log(0.01 + np.abs(zounds.spectral.stft(self.real())))

    def fake(self, bands=None, noise=True, harm=True):

        with torch.no_grad():
            decoded = {}

            bands = set(bands or [512, 1024, 2048, 4096, 8192, 16384])

            for k, v in self.decoded.items():
                if k not in bands:
                    decoded[k] = torch.zeros_like(v[0])
                    continue

                if noise and harm:
                    decoded[k] = sum(v)
                elif harm:
                    decoded[k] = v[0]
                else:
                    decoded[k] = v[1]

            single = {k: v[:1].view(1, 1, -1) for k, v in decoded.items()}

            audio = fft_frequency_recompose(single, self.n_samples)
            return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()

    def fake_spec(self, bands=None, noise=True, harm=True):
        return np.log(0.01 + np.abs(zounds.spectral.stft(self.fake(bands, noise, harm))))

    def latent(self):
        return self.encoded.data.cpu().numpy().squeeze()

    def run(self):

        stream = sample_stream(
            self.batch_size, overfit=self.overfit, normalize=True)
        
        long_stream = long_sample_stream(
            self.batch_size, overfit=self.overfit, normalize=True)
        
        while True:
            bands, feat = next(stream)
            self.bands = bands
            decoded, _ = train_gen(feat)
            self.decoded = decoded

            _, a_feat, _, b_feat = next(long_stream)
            self.encoded = train_disc(a_feat, b_feat)