from config.dotenv import Config
from itertools import chain
import torch
import numpy as np
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss
from modules.ddsp import NoiseModel, OscillatorBank
from torch.nn import functional as F
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.mixer import Mixer
from modules.psychoacoustic import PsychoacousticFeature
from upsample import ConvUpsample
from util import device
from train import gan_cycle

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
        self.t = Mixer(network_channels, 32, 3, return_features=True)
    
    def forward(self, x):
        x = x.view(-1, network_channels, 32, self.periodicity)
        x = self.p(x).permute(0, 2, 1) # (-1, 32, 512)
        x = self.r(x)
        x, features = self.t(x)
        return x, features

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict({str(k): BranchEncoder(65) for k in feature.band_sizes})
        self.r = LinearOutputStack(network_channels, 2, in_channels=384)
        self.t = Mixer(network_channels, 32, 3, return_features=True)
        self.e = LinearOutputStack(network_channels, 2)
        self.apply(init_func)
    
    def forward(self, x):
        d = {k: self.encoders[str(k)](x[k]) for k in x.keys()}

        x = [v[0] for v in d.values()]
        feats = list(chain(*[v[1] for v in d.values()]))

        x = torch.cat(x, dim=-1)
        x = self.r(x)
        x, f = self.t(x)
        x = self.e(x)
        x = x[:, -1, :]
        return x, [*feats, *f]

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.judge = LinearOutputStack(network_channels, 3, out_channels=1)
    
    def forward(self, x, conditioning=None):
        x, feats = self.encoder(x)
        c, _ = self.encoder(conditioning)
        j = self.judge(x + c)
        return j, feats


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict({str(k): BandUpsample(k) for k in feature.band_sizes})
        self.up = ConvUpsample(
            network_channels, network_channels, 4, 32, 'fft_learned', network_channels)
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
        x = self.t(x)
        x = x.permute(0, 2, 1)
        harm = self.osc(x)
        noise = self.noise(x)
        return harm, noise

encoder = Encoder().to(device)
decoder = Decoder().to(device)
ae_optim = Adam(chain(encoder.parameters(), decoder.parameters()), lr=1e-4, betas=(0, 0.9))

disc = Discriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


def train_gen(batch):
    ae_optim.zero_grad()

    encoded, _ = encoder(batch)
    decoded = decoder(encoded)

    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)


    # TODO: Factor this out
    loss = 0
    for k, v in batch.items():
        loss = loss + torch.abs(v - fake_feat[k]).sum()
    
    print('G', loss.item())
    loss.backward()
    ae_optim.step()
    return decoded, encoded


def train_disc(batch):

    disc_optim.zero_grad()

    encoded, _ = encoder(batch)
    decoded = decoder(encoded)

    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)

    # get disc loss
    rj, _ = disc(batch, batch)
    fj, _ = disc(fake_feat, batch)

    loss = least_squares_disc_loss(rj, fj)

    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class MultiresolutionAutoencoderWithActivationRefinements13(object):
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
                # train_disc(feat)
                pass
