import torch
import numpy as np
from data import feature, sample_stream, compute_feature_dict
from modules.ddsp import DDSP
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_recompose
from modules.linear import LinearOutputStack
from util import device
from itertools import chain
from torch.nn import functional as F

from modules.multiresolution import BandEncoder, DecoderShell, EncoderShell
from modules.pos_encode import ExpandUsingPosEncodings
from modules.transformer import Transformer
from util.readmedocs import readme

network_channels = 64
multiply = True
learnable_pos = True

class EncoderBranch(nn.Module):
    def __init__(self, band_size, periodicity_feature_size):
        super().__init__()
        self.encoder = BandEncoder(
            network_channels, periodicity_feature_size, band_size)
        self.linear = nn.Linear(512, network_channels)
        self.transformer = Transformer(network_channels, 4)
        self.periodicity_feature_size = periodicity_feature_size
        self.band_size = band_size

    def forward(self, x):
        x = x.view(-1, 64, 32, self.periodicity_feature_size)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        x = self.transformer(x)
        return x


class Expander(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = ExpandUsingPosEncodings(
            network_channels, 
            32, 
            16, 
            network_channels, 
            multiply=multiply, 
            learnable_encodings=learnable_pos)
        # self.transformer = Transformer(network_channels, 3)
        self.transformer = LinearOutputStack(network_channels, 3)

    def forward(self, x):
        x = self.expand(x)
        x = self.transformer(x)
        return x


class Summarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.reducer = nn.Linear(
            feature.n_bands * network_channels, network_channels)
        self.transformer = Transformer(network_channels, 4)
        self.to_latent = nn.Linear(network_channels, network_channels)

    def forward(self, x):
        x = self.reducer(x)
        x = self.transformer(x)
        x = x[:, -1:, :]
        x = self.to_latent(x)
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


class PosEncodedUpsample(nn.Module):
    def __init__(self, band_size):
        super().__init__()
        self.band_size = band_size
        self.expander = ExpandUsingPosEncodings(
            network_channels, 
            band_size, 
            16, 
            network_channels, 
            multiply=multiply, 
            learnable_encodings=learnable_pos)
        # self.transformer = Transformer(network_channels, 4)
        self.transformer = LinearOutputStack(network_channels, 4)
        self.ddsp = DDSP(network_channels, band_size, constrain=False)

    def forward(self, x):
        batch_size, time, channels = x.shape
        x = self.expander.forward(x)
        x = self.transformer.forward(x)

        # time last dim convention
        x = x.permute(0, 2, 1)
        x = self.ddsp(x)
        return x


def make_decoder(band_size):
    return PosEncodedUpsample(band_size)


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
    chain(decoder.parameters(), encoder.parameters()), lr=1e-3, betas=(0, 0.9))


@readme
class FNetAutoencoder(object):
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
    
    def encoding(self):
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
