import torch
import numpy as np
from data import feature, sample_stream, compute_feature_dict
from modules.ddsp import DDSP, NoiseModel, OscillatorBank
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_recompose
from util import device
from itertools import chain
from torch.nn import functional as F

from modules.multiresolution import BandEncoder, DecoderShell, EncoderShell
from util.readmedocs import readme

network_channels = 64
multiply = True
learnable_pos = True


class EncoderBranch(nn.Module):
    """
    Encode an individual resolution into a sequence
    with 32 time steps
    """

    def __init__(self, band_size, periodicity_feature_size):
        super().__init__()
        self.encoder = BandEncoder(
            network_channels, periodicity_feature_size, band_size)
        self.linear = nn.Linear(512, network_channels)
        self.context = nn.Conv1d(network_channels, network_channels, 3, 1, 1)
        self.periodicity_feature_size = periodicity_feature_size
        self.band_size = band_size

    def forward(self, x):
        x = x.view(-1, 64, 32, self.periodicity_feature_size)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = x.permute(0, 2, 1)
        x = self.context(x)
        x = F.leaky_relu(x, 0.2)
        x = x.permute(0, 2, 1)
        return x


class Summarizer(nn.Module):
    """
    Encode a 32-step time sequence into a single latent vector
    """

    def __init__(self):
        super().__init__()
        self.reducer = nn.Linear(
            feature.n_bands * network_channels, network_channels)
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
        x = self.reducer(x)
        x = x.permute(0, 2, 1)
        x = self.summary(x)
        return x


class Expander(nn.Module):
    """
    Decode a latent vector into a 32-step time sequence
    """

    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(network_channels, network_channels * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 1, 1, 0)
        )

    def forward(self, x):
        x = x.view(-1, network_channels)
        x = self.initial(x).view(-1, network_channels, 4)
        x = self.net(x)
        return x


class BandUpsample(nn.Module):
    """
    Upsample an individual resolution
    """

    def __init__(self, band_size):
        super().__init__()
        self.band_size = band_size

        self.net = nn.Sequential(
            nn.Conv1d(network_channels, network_channels, 3, 1, 1),
            nn.LeakyReLU(0.2)
        )

        self.osc_bank = OscillatorBank(
            network_channels,
            network_channels,
            band_size,
            constrain=True,
            log_frequency=False,
            log_amplitude=False)

        self.noise = NoiseModel(
            input_channels=network_channels,
            input_size=32,
            n_noise_frames=band_size // 4,
            n_audio_samples=band_size,
            channels=network_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)

        osc = self.osc_bank(x)
        noise = self.noise(x)
        x = osc + noise

        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shell = EncoderShell(
            network_channels, make_band_encoder, Summarizer, feature)

    def forward(self, x):
        sizes = set([len(v.shape) for v in x.values()])

        if 3 in sizes:
            feat = compute_feature_dict(x)
        else:
            feat = x

        x = self.shell(feat)
        return x


def make_decoder(band_size):
    return BandUpsample(band_size)


def make_band_encoder(periodicity, band_size):
    return EncoderBranch(band_size, periodicity)


def make_summarizer():
    return Summarizer()


decoder = DecoderShell(
    network_channels,
    make_decoder,
    Expander,
    feature).to(device)
encoder = Encoder().to(device)
optim = Adam(
    chain(decoder.parameters(), encoder.parameters()), lr=1e-4, betas=(0, 0.9))


@readme
class MultiresolutionAutoencoder2(object):
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

    def real(self):
        with torch.no_grad():
            single = {k: v[0].view(1, 1, -1) for k, v in self.bands.items()}
            audio = fft_frequency_recompose(single, self.n_samples)
            return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()

    def real_spec(self):
        return np.log(0.01 + np.abs(zounds.spectral.stft(self.real())))

    def fake(self):
        with torch.no_grad():
            single = {k: v[0].view(1, 1, -1) for k, v in self.decoded.items()}
            audio = fft_frequency_recompose(single, self.n_samples)
            return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()

    def fake_spec(self):
        return np.log(0.01 + np.abs(zounds.spectral.stft(self.fake())))

    def latent(self):
        return self.encoded.data.cpu().numpy().squeeze()

    def run(self):
        stream = sample_stream(self.batch_size, overfit=self.overfit)
        for bands, feat in stream:
            self.bands = bands
            optim.zero_grad()
            encoded = encoder(feat)
            self.encoded = encoded
            decoded = decoder(encoded)
            self.decoded = decoded
            recon_feat = compute_feature_dict(decoded)

            loss = 0
            for k, v in feat.items():
                loss = loss + F.mse_loss(recon_feat[k], v)

            loss.backward()
            optim.step()
            print('AE', loss.item())
