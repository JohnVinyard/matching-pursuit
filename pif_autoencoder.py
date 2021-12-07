import torch
from torch import nn
from torch.optim.adam import Adam
import zounds
from datastore import batch_stream
from decompose import fft_frequency_decompose, fft_frequency_recompose
from modules import pos_encode_feature
from modules3 import LinearOutputStack
import numpy as np
from torch.nn import functional as F

from test_optisynth import PsychoacousticFeature
from itertools import chain

path = '/hdd/musicnet/train_data'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sr = zounds.SR22050()
overfit = False
batch_size = 1 if overfit else 4
min_band_size = 512
n_samples = 2**14
network_channels = 64

n_bands = int(np.log2(n_samples) - np.log2(min_band_size))

feature = PsychoacousticFeature().to(device)


def latent(batch_size=batch_size):
    return torch.FloatTensor(batch_size, network_channels).normal_(0, 1).to(device)

class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.leaky_relu(x, 0.2)


def init_weights(p):

    with torch.no_grad():
        try:
            p.weight.uniform_(-0.125, 0.125)
        except AttributeError:
            pass


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
                if isinstance(layer, nn.LeakyReLU):
                    features.append(x.view(batch_size, -1))
            # (batch, 128, 1, 1)
            x = x.view(batch_size, self.channels)
            features = torch.cat(features, dim=-1)
            return features, x
        else:
            x = self.collapse(x)
            # (batch, 128, 1, 1)
            x = x.view(batch_size, self.channels)
            return x


class Encoder(nn.Module):
    def __init__(self, channels, return_features=False):
        super().__init__()
        self.channels = channels

        bands = {str(k): BandEncoder(channels, v, return_features)
                 for k, v in feature.kernel_sizes.items()}
        self.bands = nn.ModuleDict(bands)
        self.encoder = LinearOutputStack(
            channels, 5, in_channels=channels * len(self.bands))
        self.return_features = return_features
        self.apply(init_weights)

    def forward(self, x):
        if self.return_features:
            features, encodings = zip(*[self.bands[str(k)](v) for k, v in x.items()])
            encodings = torch.cat(encodings, dim=1)
            x = self.encoder(encodings)
            features = torch.cat(features, dim=-1)
            return features, x
        else:
            encodings = [self.bands[str(k)](v) for k, v in x.items()]
            encodings = torch.cat(encodings, dim=1)
            x = self.encoder(encodings)
            return x


class ConvBandDecoder(nn.Module):
    def __init__(self, channels, band_size, use_filters=False, use_transposed_conv=False):
        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.band_specific = LinearOutputStack(channels, 3)
        self.n_layers = int(np.log2(band_size) - np.log2(4))
        self.expand = LinearOutputStack(channels, 3, out_channels=channels * 4)
        self.use_filters = use_filters
        self.use_transposed_conv = use_transposed_conv

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

    def forward(self, x):
        x = x.view(batch_size, self.channels)
        x = self.band_specific(x)

        x = self.expand(x).view(batch_size, self.channels, 4)
        x = self.upsample(x)
        x = self.to_samples(x)

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
            use_mlp=True):

        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.use_filter = use_filter
        self.learned_encoding = learned_encoding
        self.use_mlp = use_mlp

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
        
    
    def forward(self, x):
        x = x.view(batch_size, self.channels)
        x = self.band_specific(x)

        if not self.learned_encoding:
            pos = pos_encode_feature(torch.linspace(-1, 1, self.band_size).view(-1, 1), 1, self.band_size, 16)\
                .view(1, self.band_size, 33).repeat(batch_size, 1, 1).permute(0, 2, 1).to(device)
        else:
            pos = self.pos.repeat(batch_size, 1, 1)
        
        latent = x.view(batch_size, self.channels, 1).repeat(1, 1, self.band_size)
        x = torch.cat([pos, latent], dim=1)

        if self.use_mlp:
            x = x.permute(0, 2, 1)
            x = self.to_channels(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)
        
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
            use_transposed_conv=True)
        # return PosEncodedDecoder(
        #     self.channels, 
        #     band_size, 
        #     use_filter=True, 
        #     learned_encoding=False, 
        #     use_mlp=True)

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
        x = x.view(batch_size, self.channels)
        return torch.sigmoid(self.net(x))


encoder = Encoder(network_channels).to(device)
decoder = Decoder(network_channels).to(device)
gen_optim = Adam(
    chain(encoder.parameters(), decoder.parameters()), 
    lr=1e-4, 
    betas=(0, 0.9))

disc_encoder = Encoder(network_channels, return_features=True).to(device)
judge = Judge(network_channels).to(device)
disc_optim = Adam(chain(disc_encoder.parameters(), judge.parameters()), lr=1e-4, betas=(0, 0.9))

def real():
    with torch.no_grad():
        single = {k: v[0].view(1, 1, -1) for k, v in bands.items()}
        audio = fft_frequency_recompose(single, n_samples)
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()


def fake():
    with torch.no_grad():
        single = {k: v[0].view(1, 1, -1) for k, v in decoded.items()}
        audio = fft_frequency_recompose(single, n_samples)
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()

def train_gen(feat):
    gen_optim.zero_grad()

    # get real disc encoder features
    rf, _ = disc_encoder(feat)


    # encode, decode, and compute fake disc features
    enc = encoder(feat)
    fake = decoder(enc)
    fake_feat = compute_feature_dict(fake)
    ff, e = disc_encoder(fake_feat)
    j = judge(e)

    feature_loss = F.mse_loss(ff, rf) * 100
    loss = torch.abs(1 - j).mean() + feature_loss
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake, enc

def train_disc(feat):
    disc_optim.zero_grad()

    enc = encoder(feat)
    fake = decoder(enc)
    fake_feat = compute_feature_dict(fake)

    renc = encoder(feat)
    rj = judge(renc)

    fenc = encoder(fake_feat)
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
        

        # encoded = encoder(feat)
        # decoded = decoder(encoded)

        # real_features = torch.cat([v.view(batch_size, -1)
        #                           for v in feat.values()], dim=1)
        # fake_features, _ = feature(decoded)

        # loss = F.mse_loss(fake_features, real_features)
        # loss.backward()
        # optim.step()
        # print(iterations + 1, loss.item())

        # e = encoded.data.cpu().numpy()
        # iterations += 1
