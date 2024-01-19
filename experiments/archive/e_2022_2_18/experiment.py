from config.dotenv import Config
from itertools import chain
import torch
import numpy as np
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss, squared_gan_loss
from modules.ddsp import NoiseModel, OscillatorBank
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.psychoacoustic import PsychoacousticFeature
from upsample import FFTUpsampleBlock
from util import device
from torch.nn import functional as F
from train import gan_cycle

from modules.multiresolution import BandEncoder, DecoderShell, EncoderShell
from util.readmedocs import readme

network_channels = 64
n_samples = 2**14

feature = PsychoacousticFeature(
    kernel_sizes=[128] * 6
).to(device)


def compute_feature_dict(x):
    x = {k: v.contiguous() for k, v in x.items()}
    return feature.compute_feature_dict(
        x,
        constant_window_size=128
    )


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


# Discriminator ===================================================================


class EncoderBranch(nn.Module):
    """
    Encode individual frequency band
    """

    def __init__(self, band_size, periodicity_feature_size, is_disc=False):
        super().__init__()

        self.encoder = BandEncoder(
            network_channels, periodicity_feature_size, band_size)
        self.linear = nn.Linear(512, network_channels)
        self.context = nn.Conv1d(network_channels, network_channels, 3, 1, 1)
        self.periodicity_feature_size = periodicity_feature_size
        self.band_size = band_size
        self.is_disc = is_disc

        if is_disc:
            self.disc_context = nn.Conv1d(
                network_channels, network_channels, 3, 1, 1)

    def forward(self, x, conditioning=None):
        
        x = x.view(-1, 64, 32, self.periodicity_feature_size)

        if self.is_disc:
            spec = torch.norm(conditioning, dim=-1)
            spec = self.disc_context(spec)
            spec_context = spec.permute(0, 2, 1)
        else:
            spec_context = 0

        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        
        x = x + spec_context
        x = x.permute(0, 2, 1)
        x = self.context(x)
        x = F.leaky_relu(x, 0.2)
        x = x.permute(0, 2, 1)
        return x


class Summarizer(nn.Module):
    """
    Combine and summarize all frequency bands
    """

    def __init__(self, is_disc=False):
        super().__init__()

        self.is_disc = is_disc

        self.reducer = nn.Linear(
            feature.n_bands * network_channels, network_channels)

        if self.is_disc:
            self.summary = nn.Sequential(
                nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            )
        else:
            self.summary = nn.Sequential(
                nn.Conv1d(network_channels, network_channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 2, 2, 0),
                nn.LeakyReLU(0.2),
                nn.Conv1d(network_channels, network_channels, 1, 1, 0)
            )

    def forward(self, x):
        features = []
        x = self.reducer(x)
        features.append(x)

        x = x.permute(0, 2, 1)

        # x = self.summary(x)

        for layer in self.summary:
            x = layer(x)
            features.append(x)

        x = x.permute(0, 2, 1)

        return x, features


class Encoder(nn.Module):
    def __init__(self, is_disc=False):
        super().__init__()
        self.shell = EncoderShell(
            network_channels,
            make_make_band_encoder(is_disc),
            lambda: Summarizer(is_disc),
            feature,
            is_disc=is_disc,
            return_features=False)

        self.is_disc = is_disc
        if is_disc:
            self.judge = LinearOutputStack(network_channels, 3, out_channels=1)

    def forward(self, x, conditioning=None):
        sizes = set([len(v.shape) for v in x.values()])

        if 3 in sizes:
            feat = compute_feature_dict(x)
        else:
            feat = x

        if self.is_disc:
            x, features = self.shell(feat, conditioning)
        else:
            x, features = self.shell(feat)

        x = x.reshape(x.shape[0], -1, network_channels)

        if self.is_disc:
            r = self.judge(x)
            return r, features

        return x

# Generator ============================================================================


class Expander(nn.Module):
    """
    Expand a representation of all bands together
    """

    def __init__(self, twod=False):
        super().__init__()
        self.initial = nn.Linear(network_channels, network_channels * 4)
        self.twod = twod

        # (batch, channels, 4)

        if self.twod:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(network_channels, 32,
                                   (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),  # (32, 4, 4)

                nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),  # (16, 8, 8)

                nn.ConvTranspose2d(16, 8, (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),  # (8, 16, 16)

                nn.ConvTranspose2d(8, 4, (4, 4), (2, 2), (1, 1)),
                nn.LeakyReLU(0.2),  # (4, 32, 32)

                nn.ConvTranspose2d(4, 1, (4, 1), (2, 1), (1, 0)),
                # (1, 64, 32)
            )
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose1d(
                    network_channels, network_channels, 4, 2, 1),
                # FFTUpsampleBlock(network_channels, network_channels, 4, infer=True),
                # nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose1d(
                    network_channels, network_channels, 4, 2, 1),
                # FFTUpsampleBlock(network_channels, network_channels, 8, infer=True),
                # nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),

                nn.ConvTranspose1d(
                    network_channels, network_channels, 4, 2, 1),
                # FFTUpsampleBlock(network_channels, network_channels, 16, infer=True),
                # nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2),

                nn.Conv1d(network_channels, network_channels, 1, 1, 0)
            )

    def forward(self, x):
        x = x.reshape(-1, network_channels)
        x = self.initial(x)

        if self.twod:
            x = x.view(-1, network_channels, 2, 2)
        else:
            x = x.view(-1, network_channels, 4)

        x = self.net(x)
        x = x.view(-1, network_channels, 32)

        return x


class BandUpsample(nn.Module):
    """
    Expand and finalize an individual frequency band
    """

    def __init__(self, band_size):
        super().__init__()
        self.band_size = band_size
        n_layers = 4
        self.channels = network_channels

        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2)
            ) for _ in range(n_layers)
        ])

        self.final = nn.Conv1d(network_channels, network_channels, 3, 1, 1)

        density = {
            512: 32,
            1024: 32,
            2048: 32,
            4096: 32,
            8192: 32,
            16384: 32
        }

        noise_frames = {
            512: 64,
            1024: 64,
            2048: 64,
            4096: 64,
            8192: 64,
            16384: 64
        }

        linear_freqs = set([512, 1024])

        self.osc = OscillatorBank(
            self.channels,
            density[band_size],
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
            n_noise_frames=band_size // 8,
            n_audio_samples=self.band_size,
            channels=self.channels,
            activation=lambda x: x,
            squared=False,
            # don't mess with the DC frequency for the lowest band
            mask_after=1 if band_size == 512 else None)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.final(x)
        harm = self.osc(x)
        noise = self.noise(x)
        return harm, noise


def make_decoder(band_size):
    return BandUpsample(band_size)


def make_make_band_encoder(is_disc=False):

    def make_band_encoder(periodicity, band_size):
        return EncoderBranch(band_size, periodicity, is_disc)
    
    return make_band_encoder


decoder = DecoderShell(
    network_channels,
    make_decoder,
    Expander,
    feature).to(device)
encoder = Encoder().to(device)
ae_optim = Adam(
    chain(decoder.parameters(), encoder.parameters()),
    lr=1e-4,
    betas=(0, 0.9))


disc = Encoder(is_disc=True).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


def train_gen(batch):

    ae_optim.zero_grad()

    encoded = encoder(batch)
    decoded = decoder(encoded)

    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)

    rj, r_features = disc(batch, batch)
    fj, f_features = disc(fake_feat, batch)

    # minimize distance in feature space from real
    loss = 0
    for i in range(len(r_features)):
        loss = loss + torch.abs(r_features[i] - f_features[i]).sum()

    loss.backward()
    ae_optim.step()
    print('G', loss.item())
    return decoded, encoded


def train_disc(batch):
    disc_optim.zero_grad()

    encoded = encoder(batch)
    decoded = decoder(encoded)

    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)

    # get disc loss
    rj, r_features = disc(batch, batch)
    fj, f_features = disc(fake_feat, batch)

    loss = least_squares_disc_loss(rj, fj)

    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class MultiresolutionAutoencoderWithActivationRefinements10(object):
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

        for bands, feat in stream:

            step = next(gan_cycle)

            if step == 'gen':
                self.bands = bands
                decoded, encoded = train_gen(feat)
                self.decoded = decoded
                self.encoded = encoded
            else:
                train_disc(feat)
