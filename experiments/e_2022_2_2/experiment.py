import torch
import numpy as np
from data import feature, sample_stream, compute_feature_dict
from modules.ddsp import NoiseModel, OscillatorBank
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_recompose
from util import device
from torch.nn import functional as F

from modules.multiresolution import BandEncoder, DecoderShell, EncoderShell
from modules.transformer import Transformer
from train import train_disc, train_gen, get_latent, gan_cycle
from util.readmedocs import readme

network_channels = 64


# Discriminator ===================================================================

class EncoderBranch(nn.Module):
    """
    Encode individual frequency band
    """

    def __init__(self, band_size, periodicity_feature_size):
        super().__init__()
        self.encoder = BandEncoder(
            network_channels, periodicity_feature_size, band_size)
        self.linear = nn.Linear(512, network_channels)
        self.context = nn.Conv1d(network_channels, network_channels, 3, 1, 1)
        self.periodicity_feature_size = periodicity_feature_size
        self.band_size = band_size

        self.factor = nn.Parameter(torch.FloatTensor(1).fill_(1))

    def forward(self, x):
        x = x * self.factor
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
    Combine and summarize all frequency bands
    """

    def __init__(self):
        super().__init__()
        self.reducer = nn.Linear(
            feature.n_bands * network_channels, network_channels)
        self.transformer = Transformer(network_channels, 4)
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
        # print('DISC STD', x.std().item())
        x = self.reducer(x)
        x = x.permute(0, 2, 1)
        x = self.summary(x)
        # x = self.transformer(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shell = EncoderShell(
            network_channels, make_band_encoder, Summarizer, feature)

        self.judge = nn.Linear(network_channels, 1)

    def forward(self, x):
        sizes = set([len(v.shape) for v in x.values()])

        if 3 in sizes:
            feat = compute_feature_dict(x)
        else:
            feat = x

        x = self.shell(feat)
        x = x.reshape(-1, network_channels)
        x = self.judge(x)
        return x

# Generator ============================================================================


class Expander(nn.Module):
    """
    Expand a representation of all bands together
    """
    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(network_channels, network_channels * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose1d(network_channels, network_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(network_channels, network_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(network_channels, network_channels, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 1, 1, 0)
        )

    def forward(self, x):
        x = x.view(-1, network_channels)
        x = self.initial(x).view(-1, network_channels, 4)
        x = self.net(x)
        # print('GEN STD', x.std().item())
        return x


class BandUpsample(nn.Module):
    """
    Expand and finalize an individual frequency band
    """
    def __init__(self, band_size):
        super().__init__()
        self.band_size = band_size
        # n_layers = int(np.log2(band_size) - np.log2(32))
        n_layers = 4
        self.channels = network_channels

        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(network_channels, network_channels, 3, 1, 1),
                nn.LeakyReLU(0.2)
            ) for _ in range(n_layers)
        ])

        self.final = nn.Conv1d(network_channels, network_channels, 3, 1, 1)

        self.osc = OscillatorBank(
            self.channels,
            32,
            self.band_size,
            activation=lambda x: torch.abs(torch.clamp(x, -1, 1)),
            return_params=False,
            constrain=True,
            log_frequency=True)

        self.noise = NoiseModel(
            input_channels=self.channels,
            input_size=32,
            n_noise_frames=self.band_size // 4,
            n_audio_samples=self.band_size,
            channels=self.channels)
        
        self.factor = nn.Parameter(torch.FloatTensor(1).fill_(1))
        self.harm_factor = nn.Parameter(torch.FloatTensor(1).fill_(1))
        self.noise_factor = nn.Parameter(torch.FloatTensor(1).fill_(1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.final(x)
        # print('BAND', x.std().item())
        harm = self.osc(x) * self.harm_factor
        noise = self.noise(x) * self.noise_factor
        return (harm + noise) * self.factor


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
gen_optim = Adam(decoder.parameters(), lr=1e-4, betas=(0, 0.9))

encoder = Encoder().to(device)
disc_optim = Adam(
    encoder.parameters(), lr=1e-4, betas=(0, 0.9))


@readme
class MultiresolutionGan(object):
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

    def fake(self):
        with torch.no_grad():
            single = {k: v[0].view(1, 1, -1) for k, v in self.decoded.items()}
            audio = fft_frequency_recompose(single, self.n_samples)
            return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()

    def fake_spec(self):
        return np.log(0.01 + np.abs(zounds.spectral.stft(self.fake())))

    def run(self):
        stream = sample_stream(self.batch_size, overfit=self.overfit)

        def make_latent():
            return get_latent(self.batch_size, network_channels)

        for bands, feat in stream:
            step = next(gan_cycle)
            self.bands = bands

            if step == 'gen':
                decoded = train_gen(feat, decoder, encoder,
                                    gen_optim, make_latent)
                self.decoded = decoded
            else:
                train_disc(feat, encoder, decoder, disc_optim, make_latent)
