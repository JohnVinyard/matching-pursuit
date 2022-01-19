import torch
from torch import nn
from torch.optim.adam import Adam
import zounds
from datastore import batch_stream
from decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.multiresolution import BandEncoder, ConvBandDecoder, DecoderShell, EncoderShell
import numpy as np
from modules.pos_encode import ExpandUsingPosEncodings
from modules.psychoacoustic import PsychoacousticFeature
from modules.transformer import Transformer
from train import gan_cycle, train_gen, train_disc
from train.gan import get_latent

path = '/hdd/musicnet/train_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

sr = zounds.SR22050()
batch_size = 2
min_band_size = 512
n_samples = 2**14
network_channels = 64
latent_dim = network_channels

feature = PsychoacousticFeature().to(device)


def process_batch(s):
    s = s.reshape(-1, 1, n_samples)
    s = torch.from_numpy(s).to(device).float()
    bands = fft_frequency_decompose(s, min_band_size)
    return bands


def build_compute_feature_dict():
    stream = batch_stream(path, '*.wav', 16, n_samples)
    s = next(stream)

    bands = process_batch(s)
    feat = feature.compute_feature_dict(bands)

    print('Computing stats')
    means = {k: v.mean() for k, v in feat.items()}
    stds = {k: v.std() for k, v in feat.items()}

    def f(bands):
        x = feature.compute_feature_dict(bands)
        x = {k: v - means[k] for k, v in x.items()}
        x = {k: v / stds[k] for k, v in x.items()}
        return x

    return f


compute_feature_dict = build_compute_feature_dict()


def sample_stream():
    stream = batch_stream(path, '*.wav', batch_size, n_samples)
    for s in stream:
        bands = process_batch(s)
        feat = compute_feature_dict(bands)
        yield bands, feat


def real():
    with torch.no_grad():
        single = {k: v[0].view(1, 1, -1) for k, v in bands.items()}
        audio = fft_frequency_recompose(single, n_samples)
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()


def real_spec():
    return np.log(0.01 + np.abs(zounds.spectral.stft(real())))


def fake():
    with torch.no_grad():
        single = {k: v[0].view(1, 1, -1) for k, v in decoded.items()}
        audio = fft_frequency_recompose(single, n_samples)
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()


def fake_spec():
    return np.log(0.01 + np.abs(zounds.spectral.stft(fake())))


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
            network_channels, 32, 16, network_channels)
        self.transformer = Transformer(network_channels, 3)
    
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
        self.judge = nn.Linear(network_channels, 1)

    def forward(self, x):
        x = self.reducer(x)
        x = self.transformer(x)
        x = x[:, -1:, :]
        x = self.judge(x)
        x = torch.sigmoid(x)
        return x


class Discriminator(nn.Module):
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
    return ConvBandDecoder(
        network_channels, band_size, nn.LeakyReLU(0.2), feature, use_ddsp=True)


def make_band_encoder(periodicity, band_size):
    return EncoderBranch(band_size, periodicity)


def make_summarizer():
    return Summarizer()


gen = DecoderShell(
    network_channels,
    make_decoder,
    Expander,
    feature).to(device)

gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


def make_latent():
    return get_latent(batch_size, latent_dim)


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    stream = sample_stream()

    for iteration, step in enumerate(gan_cycle):
        bands, feat = next(stream)

        if step == 'gen':
            decoded = train_gen(feat, gen, disc, gen_optim, make_latent)
        else:
            train_disc(feat, disc, gen, disc_optim, make_latent)
