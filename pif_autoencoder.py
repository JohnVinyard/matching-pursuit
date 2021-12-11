import torch
from torch import nn
from torch.nn.modules.conv import Conv1d
from torch.optim.adam import Adam
import zounds
from all_in_one import covariance
from datastore import batch_stream
from decompose import fft_frequency_decompose, fft_frequency_recompose
from modules import pos_encode_feature
from modules3 import LinearOutputStack
import numpy as np
from torch.nn import functional as F
from modules2 import DilatedBlock

from test_optisynth import PsychoacousticFeature
from itertools import chain

path = '/hdd/musicnet/train_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

sr = zounds.SR22050()
overfit = False
batch_size = 1 if overfit else 4
min_band_size = 512
n_samples = 2**14
network_channels = 64
gen_uses_sine_activation = True
compact_disc = True
init_value = 0.125

n_bands = int(np.log2(n_samples) - np.log2(min_band_size))

feature = PsychoacousticFeature().to(device)


def latent(batch_size=batch_size):
    return torch.FloatTensor(batch_size, network_channels).normal_(0, 1).to(device)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if gen_uses_sine_activation:
            return torch.sin(x)
        else:
            return F.leaky_relu(x, 0.2)


def init_weights(p):

    with torch.no_grad():
        try:
            p.weight.uniform_(-init_value, init_value)
        except AttributeError:
            pass


class PosEncodedEncoder(nn.Module):
    def __init__(self, channels, periodicity_feature_size, return_features=False):
        super().__init__()
        self.channels = channels
        self.periodicity_feature_size = periodicity_feature_size
        self.return_features = return_features

        self.periodicity_feature_size = periodicity_feature_size
        self.period = LinearOutputStack(
            channels, 3, in_channels=periodicity_feature_size, out_channels=8)

        self.embed_periodicity = nn.Conv1d(512, channels, 1, 1, 0)
        self.embed_pos = nn.Conv1d(33, channels, 1, 1, 0)

        n_layers = int(np.log2(32))

        self.net = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, 1, 0),
            nn.LeakyReLU(0.2),
            *[
                nn.Sequential(
                    nn.Conv1d(channels, channels, 2, 2, 0),
                    nn.LeakyReLU(0.2)
                )
                for _ in range(n_layers)],
            nn.Conv1d(channels, channels, 1, 1, 0)
        )

    def forward(self, x):

        time_dim = 32

        pos = pos_encode_feature(torch.linspace(-1, 1, time_dim).view(-1, 1), 1, time_dim, 16)\
            .view(1, time_dim, 33)\
            .repeat(batch_size, 1, 1)\
            .permute(0, 2, 1).to(device)
        pos = self.embed_pos(pos)

        x = x.view(batch_size, 64, time_dim, self.periodicity_feature_size)
        x = self.period(x)
        x = x.permute(0, 1, 3, 2).reshape(batch_size, 8 * 64, time_dim)
        x = self.embed_periodicity(x)

        x = torch.cat([pos, x], dim=1)

        if self.return_features:
            raise NotImplementedError(
                'TODO: implement return features if used in discriminator')
        else:
            x = self.net(x)
            x = x.view(batch_size, self.channels)
            return x


class DilatedDiscEncoder(nn.Module):
    def __init__(self, channels, periodicity_feature_size, return_features=False):
        super().__init__()
        self.channels = channels
        self.periodicity_feature_size = periodicity_feature_size
        self.period = LinearOutputStack(
            channels, 3, in_channels=periodicity_feature_size, out_channels=8)
        self.return_features = return_features

        self.net = nn.Sequential(
            nn.Conv1d(512, channels, 1, 1, 0),
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 2),
            DilatedBlock(channels, 4),
            DilatedBlock(channels, 8),
            DilatedBlock(channels, 1),
        )

    def forward(self, x):
        # (batch, 64, 32, N)
        x = x.view(batch_size, 64, 32, self.periodicity_feature_size)
        x = self.period(x)
        # (batch, 64, 32, 8)
        x = x.permute(0, 3, 1, 2)
        # (batch, 8, 64, 32)
        x = x.reshape(batch_size, 512, 32)
        if self.return_features:
            features = []
            for layer in self.net:
                x = layer(x)
                features.append(x.view(batch_size, -1))

            x = x.permute(0, 2, 1)
            features = torch.cat(features, dim=-1)
            return features, x
        else:
            x = self.net(x)
            x = x.permute(0, 2, 1)
            return x


class BandEncoder(nn.Module):
    def __init__(self, channels, periodicity_feature_size, return_features=False):
        super().__init__()
        self.channels = channels
        self.periodicity_feature_size = periodicity_feature_size
        self.period = LinearOutputStack(
            channels, 3, in_channels=periodicity_feature_size, out_channels=8)
        self.return_features = return_features

        self.collapse = nn.Sequential(
            nn.Conv2d(8, 16, (3, 3), (2, 2), (1, 1)),  # 32, 16
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1)),  # 16, 8
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)),  # 8, 4
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, self.channels, (3, 3), (2, 2), (1, 1)),  # 4, 2
            nn.LeakyReLU(0.2),
            nn.Conv2d(self.channels, self.channels,
                      (4, 2), (4, 2), (0, 0)),  # 4, 2
        )

    def forward(self, x):
        # (batch, 64, 32, N)
        x = x.view(batch_size, 64, 32, self.periodicity_feature_size)
        x = self.period(x)
        # (batch, 64, 32, 8)
        x = x.permute(0, 3, 1, 2)
        # (batch, 8, 64, 32)
        if self.return_features:
            features = []
            for layer in self.collapse:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    features.append(x.reshape(batch_size, -1))
            # (batch, 128, 1, 1)
            x = x.view(batch_size, self.channels)
            features = torch.cat(features, dim=-1)
            return features, x
        else:
            x = self.collapse(x)
            # (batch, 128, 1, 1)
            x = x.view(batch_size, self.channels)
            return x


class MFCC(nn.Module):
    def __init__(self, n_coeffs=12):
        super().__init__()
        self.n_coeffs = n_coeffs
    
    def forward(self, x):
        x = x.view(batch_size, 64, 32, -1)
        x = torch.norm(x, dim=-1)
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        x = torch.abs(x)
        x = x[:, 1:self.n_coeffs + 1, :]
        return x

class Encoder(nn.Module):
    def __init__(self, channels, return_features=False, compact=True, use_pos_encoding=False):
        super().__init__()
        self.channels = channels
        self.compact = compact
        self.return_features = return_features
        self.use_pos_encoding = use_pos_encoding

        bands = {str(k): self._make_encoder(v, use_pos_encoding)
                 for k, v in feature.kernel_sizes.items()}
        self.bands = nn.ModuleDict(bands)
        self.encoder = LinearOutputStack(
            channels, 5, in_channels=channels * len(self.bands))

        self.apply(init_weights)

    def _make_encoder(self, periodicity_size, use_pos_encoding):
        if use_pos_encoding:
            return PosEncodedEncoder(self.channels, periodicity_size, self.return_features)
        elif self.compact:
            return BandEncoder(self.channels, periodicity_size, self.return_features)
        else:
            return DilatedDiscEncoder(self.channels, periodicity_size, self.return_features)

    def forward(self, x):
        if self.return_features:
            features, encodings = zip(
                *[self.bands[str(k)](v) for k, v in x.items()])
            encodings = torch.cat(encodings, dim=-1)
            x = self.encoder(encodings)
            features = torch.cat(features, dim=-1)
            return features, x
        else:
            encodings = [self.bands[str(k)](v) for k, v in x.items()]
            encodings = torch.cat(encodings, dim=-1)
            x = self.encoder(encodings)
            return x


class ConvBandDecoder(nn.Module):
    def __init__(self, channels, band_size, use_filters=False, use_transposed_conv=False, use_ddsp=False):
        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.band_specific = LinearOutputStack(channels, 3)
        self.n_layers = int(np.log2(band_size) - np.log2(4))
        self.expand = LinearOutputStack(channels, 3, out_channels=channels * 4)
        self.use_filters = use_filters
        self.use_transposed_conv = use_transposed_conv
        self.use_ddsp = use_ddsp

        if self.use_transposed_conv:
            self.upsample = nn.Sequential(*[
                nn.Sequential(
                    nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                    Activation()
                )
                for _ in range(self.n_layers)])
        else:
            self.upsample = nn.Sequential(*[
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv1d(channels, channels, 7, 1, 3),
                    Activation()
                )
                for _ in range(self.n_layers)])
        self.to_samples = nn.Conv1d(
            channels, 64 if use_filters else 1, 7, 1, 3)

        if self.use_ddsp:
            self.amp = nn.Conv1d(64, 64, 1, 1, 0)
            self.freq = nn.Conv1d(64, 64, 1, 1, 0)
            self.bands = torch.from_numpy(np.geomspace(
                0.01, 1, 64) * np.pi).float().to(device)

    def forward(self, x):
        x = x.view(batch_size, self.channels)
        x = self.band_specific(x)

        x = self.expand(x).view(batch_size, self.channels, 4)
        x = self.upsample(x)
        x = self.to_samples(x)

        if self.use_ddsp:
            amp = torch.sigmoid(self.amp(x))
            freq = torch.sigmoid(self.freq(x))
            amp = F.avg_pool1d(amp, 64, 1, 32)[..., :-1]
            freq = F.avg_pool1d(freq, 64, 1, 32)[..., :-1]
            freq = freq * self.bands[None, :, None]
            freq = torch.sin(torch.cumsum(freq, dim=-1)) * amp
            x = torch.mean(x, dim=1, keepdim=True)
            return x

        if not self.use_filters:
            return x

        x = F.pad(x, (0, 1))
        x = feature.banks[self.band_size][0].transposed_convolve(x) * 0.1
        return x


class PosEncodedDecoder(nn.Module):
    def __init__(
            self,
            channels,
            band_size,
            use_filter=True,
            learned_encoding=False,
            use_mlp=True,
            use_ddsp=False):

        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.use_filter = use_filter
        self.learned_encoding = learned_encoding
        self.use_mlp = use_mlp

        self.use_ddsp = use_ddsp

        self.band_specific = LinearOutputStack(channels, 3)

        if self.learned_encoding:
            self.pos = nn.Parameter(torch.FloatTensor(
                1, 33, self.band_size).normal_(0, 1))

        if self.use_mlp:
            self.to_channels = LinearOutputStack(
                channels, 7, in_channels=33 + channels, out_channels=64 if use_filter else 1)
        else:
            self.net = nn.Sequential(
                nn.Conv1d(33 + channels, channels, 7, 1, 3),
                Activation(),
                nn.Conv1d(channels, channels, 7, 1, 3),
                Activation(),
                nn.Conv1d(channels, channels, 7, 1, 3),
                Activation(),
                nn.Conv1d(channels, channels, 7, 1, 3),
                Activation(),
                nn.Conv1d(channels, channels, 7, 1, 3),
                Activation(),
                nn.Conv1d(channels, 64 if use_filter else 1, 7, 1, 3),
            )

        if self.use_ddsp:
            self.amp = nn.Conv1d(64, 64, 1, 1, 0)
            self.freq = nn.Conv1d(64, 64, 1, 1, 0)
            self.bands = torch.from_numpy(np.geomspace(
                0.01, 1, 64) * np.pi).float().to(device)

    def forward(self, x):
        x = x.view(batch_size, self.channels)
        x = self.band_specific(x)

        if not self.learned_encoding:
            pos = pos_encode_feature(torch.linspace(-1, 1, self.band_size).view(-1, 1), 1, self.band_size, 16)\
                .view(1, self.band_size, 33).repeat(batch_size, 1, 1).permute(0, 2, 1).to(device)
        else:
            pos = self.pos.repeat(batch_size, 1, 1)

        latent = x.view(batch_size, self.channels,
                        1).repeat(1, 1, self.band_size)
        x = torch.cat([pos, latent], dim=1)

        if self.use_mlp:
            x = x.permute(0, 2, 1)
            x = self.to_channels(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)
        

        if self.use_ddsp:
            amp = torch.sigmoid(self.amp(x))
            freq = torch.sigmoid(self.freq(x))
            amp = F.avg_pool1d(amp, 64, 1, 32)[..., :-1]
            freq = F.avg_pool1d(freq, 64, 1, 32)[..., :-1]
            freq = freq * self.bands[None, :, None]
            freq = torch.sin(torch.cumsum(freq, dim=-1)) * amp
            x = torch.mean(x, dim=1, keepdim=True)
            return x

        if not self.use_filter:
            return x

        
        x = F.pad(x, (0, 1))
        x = feature.banks[self.band_size][0].transposed_convolve(x) * 0.1
        return x


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        bands = {str(k): self._make_decoder(k)
                 for k, v in zip(feature.band_sizes, feature.kernel_sizes)}

        self.bands = nn.ModuleDict(bands)
        self.apply(init_weights)

    def _make_decoder(self, band_size):
        return ConvBandDecoder(
            self.channels,
            band_size,
            use_filters=True,
            use_transposed_conv=False,
            use_ddsp=False)
        # return PosEncodedDecoder(
        #     self.channels,
        #     band_size,
        #     use_filter=True,
        #     learned_encoding=True,
        #     use_mlp=True,
        #     use_ddsp=True)

    def forward(self, x):
        return {int(k): decoder(x) for k, decoder in self.bands.items()}


def process_batch(s):
    s = s.reshape(-1, 1, n_samples)
    s /= (s.max(axis=-1, keepdims=True) + 1e-12)
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


class Judge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = LinearOutputStack(channels, 5, out_channels=1)

    def forward(self, x):
        x = x.view(batch_size, -1, self.channels)
        return self.net(x)


encoder = Encoder(
    network_channels,
    use_pos_encoding=False).to(device)
decoder = Decoder(network_channels).to(device)
gen_optim = Adam(
    chain(encoder.parameters(), decoder.parameters()),
    lr=1e-4,
    betas=(0, 0.9))


disc_encoder = Encoder(
    network_channels,
    return_features=True,
    compact=compact_disc).to(device)
judge = Judge(network_channels).to(device)
disc_optim = Adam(
    chain(disc_encoder.parameters(), judge.parameters()),
    lr=1e-4,
    betas=(0, 0.9))


def real():
    with torch.no_grad():
        single = {k: v[0].view(1, 1, -1) for k, v in bands.items()}
        audio = fft_frequency_recompose(single, n_samples)
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()


def real_spec():
    return np.abs(zounds.spectral.stft(real()))


def fake():
    with torch.no_grad():
        single = {k: v[0].view(1, 1, -1) for k, v in decoded.items()}
        audio = fft_frequency_recompose(single, n_samples)
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()


def fake_spec():
    return np.abs(zounds.spectral.stft(fake()))


def train_gen(feat):
    gen_optim.zero_grad()

    # get real disc encoder features
    rf, _ = disc_encoder(feat)

    # encode, decode, and compute PIF features
    enc = encoder(feat)
    fake = decoder(enc)
    fake_feat = compute_feature_dict(fake)

    # judge the reconstruction and return intermediate features
    ff, e = disc_encoder(fake_feat)
    j = judge(e)

    # ensure that each feature has zero mean, unit variance,
    # and that features are as independent as possible
    # enc = enc.view(batch_size, network_channels)
    # mean_loss = torch.abs(0 - enc.mean(dim=0)).mean()
    # std_loss = torch.abs(1 - enc.std(dim=0)).mean()
    # cov = covariance(enc)
    # d = torch.sqrt(torch.diag(cov))
    # cov = cov / d[None, :]
    # cov = cov / d[:, None]
    # cov = torch.abs(cov)
    # cov = cov.mean()

    feature_loss = torch.abs(ff - rf).sum()
    # + mean_loss + std_loss + cov
    loss = torch.abs(1 - j).mean() + feature_loss
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake, enc


def train_disc(feat):
    disc_optim.zero_grad()

    # encode, decode and compute PIF features
    enc = encoder(feat)
    fake = decoder(enc)
    fake_feat = compute_feature_dict(fake)

    # judge the real input
    _, renc = disc_encoder(feat)
    rj = judge(renc)

    # judge the fake input
    _, fenc = disc_encoder(fake_feat)
    fj = judge(fenc)

    loss = (torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()) * 0.5
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    stream = sample_stream()
    bands, feat = next(stream)

    iterations = 0

    while True:

        if not overfit:
            bands, feat = next(stream)

        decoded, encoded = train_gen(feat)
        e = encoded.data.cpu().numpy().squeeze()

        if not overfit:
            bands, feat = next(stream)

        train_disc(feat)
