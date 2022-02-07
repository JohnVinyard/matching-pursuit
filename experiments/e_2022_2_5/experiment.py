from config.dotenv import Config
from itertools import chain
import torch
import numpy as np
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss
from modules.ddsp import NoiseModel, OscillatorBank
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.psychoacoustic import PsychoacousticFeature
from util import device
from torch.nn import functional as F

from modules.multiresolution import BandEncoder, DecoderShell, EncoderShell
from train import gan_cycle
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

# class Periodicity(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = nn.Sequential(

#             # (32, 64, 65)
#             nn.Sequential(
#                 nn.Conv1d(32, 32, (4, 4), (4, 4)),
#                 nn.LeakyReLU(0.2)
#             ),

#             # (32, 16, 16),
#             nn.Sequential(
#                 nn.Conv1d(32, 32, (2, 2), (2, 2)),
#                 nn.LeakyReLU(0.2)
#             ),
            
#         )

#         self.final = nn.Linear(64, network_channels)
    
#     def forward(self, x):
#         batch, channels, frames, period = x.shape
#         x = x.view(batch, channels, frames, period).permute(0, 2, 1, 3)
#         x = self.net(x)
#         x = x.view(batch, frames, 8, 8).view(batch, frames, 64)
#         x = self.final(x)        
#         return x

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
    Combine and summarize all frequency bands
    """

    def __init__(self, is_disc=False):
        super().__init__()
        self.is_disc = is_disc

        self.reducer = nn.Linear(
            feature.n_bands * network_channels, network_channels)

        # if is_disc:
        #     self.summary = nn.Sequential(*[nn.Sequential(
        #         nn.Conv1d(network_channels, network_channels, 3, 1, 1),
        #         nn.LeakyReLU(0.2)
        #     ) for _ in range(5)])
        # else:
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
            network_channels, make_band_encoder, make_summarizer(is_disc), feature)

        self.is_disc = is_disc
        if is_disc:
            self.judge = nn.Linear(network_channels, 1)

    def forward(self, x):
        sizes = set([len(v.shape) for v in x.values()])

        if 3 in sizes:
            feat = compute_feature_dict(x)
        else:
            feat = x

        x, features = self.shell(feat)
        x = x.reshape(x.shape[0], -1, network_channels)

        if self.is_disc:
            x = self.judge(x)
            return x, features

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
        x = x.reshape(-1, network_channels)
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
            activation=torch.sigmoid,
            amp_activation=torch.abs,
            return_params=False,
            constrain=True,
            log_frequency=False)

        self.noise = NoiseModel(
            input_channels=self.channels,
            input_size=32,
            n_noise_frames=self.band_size // 4,
            n_audio_samples=self.band_size,
            channels=self.channels,
            activation=lambda x: x,
            squared=True)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = self.final(x)
        harm = self.osc(x)
        noise = self.noise(x)
        return (harm + noise)


def make_decoder(band_size):
    return BandUpsample(band_size)


def make_band_encoder(periodicity, band_size):
    return EncoderBranch(band_size, periodicity)


def make_summarizer(is_disc=False):
    def f():
        return Summarizer(is_disc)
    return f


decoder = DecoderShell(
    network_channels,
    make_decoder,
    Expander,
    feature).to(device)
encoder = Encoder().to(device)
ae_optim = Adam(chain(decoder.parameters(), encoder.parameters()), lr=1e-3, betas=(0, 0.9))


disc = Encoder(is_disc=True).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


def loss_func(fake_feat, real_feat):
    loss = 0
    for k, v in real_feat.items():
        loss = loss + torch.abs(fake_feat[k] - v).sum()
    return loss


def train_gen(batch):
    ae_optim.zero_grad()

    r, r_features = disc(batch)

    encoded = encoder(batch)
    decoded = decoder(encoded)

    fake_feat = compute_feature_dict(decoded)

    recon_loss = 0
    for k, v in batch.items():
        recon_loss = recon_loss + torch.abs(fake_feat[k] - v).sum()
    recon_loss = recon_loss * 0.01


    j, features = disc(fake_feat)
    loss = 0
    for i, f in enumerate(r_features):
        loss = loss + torch.abs(r_features[i] - features[i]).sum()

    total = recon_loss + loss


    total.backward()
    ae_optim.step()
    print('G', total.item())
    return decoded, encoded


def train_disc(batch):
    disc_optim.zero_grad()

    encoded = encoder(batch)
    decoded = decoder(encoded)
    fake_feat = compute_feature_dict(decoded)

    rj, _ = disc(batch)
    fj, _ = disc(fake_feat)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class MultiresolutionAdversarialAutoencoder(object):
    def __init__(self, batch_size=2, overfit=False, train_disc=True):
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
        self.train_disc = train_disc

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

        stream = sample_stream(
            self.batch_size, overfit=self.overfit, normalize=True)

        for bands, feat in stream:
            step = next(gan_cycle)
            self.bands = bands

            if step == 'gen':
                decoded, encoded = train_gen(feat)
                self.decoded = decoded
                self.encoded = encoded
            elif self.train_disc:
                train_disc(feat)