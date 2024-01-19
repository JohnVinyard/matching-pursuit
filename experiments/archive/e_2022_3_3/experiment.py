from itertools import chain
from config.dotenv import Config
import torch
import numpy as np
from data.datastore import batch_stream
from modules.atoms import unit_norm
from modules.ddsp import NoiseModel, OscillatorBank
from torch.nn import functional as F
import zounds
from torch import nn
from torch.optim import Adam
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.pos_encode import ExpandUsingPosEncodings, LearnedPosEncodings
from modules.psychoacoustic import PsychoacousticFeature
from modules.transformer import Transformer
from train import gan_cycle
from train.gan import get_latent, least_squares_disc_loss, least_squares_generator_loss
from upsample import ConvUpsample
from util import device
from random import choice

from modules.multiresolution import BandEncoder
from util.readmedocs import readme
from util.weight_init import make_initializer
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

network_channels = 64
n_samples = 2**14

feature = PsychoacousticFeature(
    kernel_sizes=[128] * 6
).to(device)

init_func = make_initializer(0.1)

adjust = {
    512: 0.05,
    1024: 0.03,
    2048: 0.05,
    4096: 0.25,
    8192: 1,
    16384: 20
}

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
    def __init__(self, periodicity, band_size):
        super().__init__()
        self.band_size = band_size
        self.periodicity = periodicity
        self.p = BandEncoder(network_channels, periodicity)
        self.r = LinearOutputStack(network_channels, 2, in_channels=512)
        # self.t = Mixer(network_channels, 32, 3)

        self.down = nn.Sequential(
            nn.Conv3d(1, 8, (2, 2, 2), (2, 2, 2), (0, 0, 0)), # (32, 16, 32)
            nn.LeakyReLU(0.2),
            nn.Conv3d(8, 16, (2, 2, 2), (2, 2, 2), (0, 0, 0)), # (16, 8, 16)
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, (2, 2, 2), (2, 2, 2), (0, 0, 0)), # (8, 4, 8)
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, network_channels, (2, 2, 2), (2, 2, 2), (0, 0, 0)), # (4, 2, 4)
            nn.LeakyReLU(0.2),
            nn.Conv3d(network_channels, network_channels, (2, 2, 2), (2, 2, 2), (0, 0, 0)), # (2, 1, 2)
            nn.LeakyReLU(0.2),
            nn.Conv3d(network_channels, network_channels, (2, 1, 2), (2, 1, 2), (0, 0, 0)), # (2, 1, 2)
        )

    def forward(self, x):
        # x = x * adjust[self.band_size]

        x = x.view(-1, 1, network_channels, 32, self.periodicity)
        x = self.down(x).reshape(-1, network_channels)

        # noise = torch.zeros_like(x).normal_(0, 0.1)
        # x = x + noise
        # x = self.p(x).permute(0, 2, 1)  # (-1, 32, 512)
        # x = self.r(x)
        # x = self.t(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {str(k): BranchEncoder(65, k) for k in feature.band_sizes})
        self.r = LinearOutputStack(network_channels, 2, in_channels=384)
        # self.t = Mixer(network_channels, 32, 3)
        # self.e = LinearOutputStack(network_channels, 2)

        self.t = Transformer(network_channels, 3)
        self.pos = LearnedPosEncodings(16, network_channels)

        stride = 1

        self.reduce = nn.Sequential(
            nn.Conv1d(network_channels, network_channels, 3, stride, 1),
            # nn.AvgPool1d(3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(network_channels, network_channels, 3, stride, 1),
            # nn.AvgPool1d(3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(network_channels, network_channels, 3, stride, 1),
            # nn.AvgPool1d(3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(network_channels, network_channels, 3, stride, 1),
            # nn.AvgPool1d(3, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(network_channels, network_channels, 3, stride, 1),
            # nn.AvgPool1d(3, 2, 1),
        )

        self.final = nn.Linear(384, network_channels)


        self.apply(init_func)

    def forward(self, x):
        d = {k: self.encoders[str(k)](x[k]) for k in x.keys()}
        x = torch.cat(list(d.values()), dim=-1)
        x = self.final(x)
        return x
        # x = torch.cat(list(d.values()), dim=-1)
        # x = self.r(x)
        # x = self.pos(x)

        # # batch_size = x.shape[0]
        # # x = x.permute(0, 2, 1)
        # # x = self.reduce(x).view(batch_size, -1, network_channels)

        # x = self.t(x)
        # x = x[:, -1, :]
        # return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = LinearOutputStack(network_channels, 4, out_channels=1)
    
    def forward(self, x):
        return self.disc(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoders = nn.ModuleDict(
            {str(k): BandUpsample(k) for k in feature.band_sizes})
        
        self.up = ConvUpsample(
            network_channels, network_channels, 4, 32, 'fft', network_channels)

        # self.ln = nn.Linear(network_channels, network_channels * 4)
        # self.up = nn.Sequential(
        #     nn.ConvTranspose2d(network_channels, 32, (4, 4), (2, 2), (1, 1)),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(16, 8, (4, 4), (2, 2), (1, 1)),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(8, 4, (4, 4), (2, 2), (1, 1)),
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(4, 1, (4, 1), (2, 1), (1, 0)),
        # )

        self.apply(init_func)

    def forward(self, x):
        x = self.up(x)

        # x = self.ln(x).reshape(-1, network_channels, 2, 2)
        # x = self.up(x).reshape(-1, network_channels, 32)

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

        self.pos = LearnedPosEncodings(16, network_channels)
        self.t = nn.Sequential(
            nn.Conv1d(network_channels, network_channels, 1, 1, 0),
            nn.Conv1d(network_channels, network_channels, 1, 1, 0),
            nn.Conv1d(network_channels, network_channels, 1, 1, 0),
            nn.Conv1d(network_channels, network_channels, 1, 1, 0),
        )

        self.initial = LinearOutputStack(network_channels, 3)

        self.final = nn.Conv1d(network_channels, network_channels, 1, 1, 0)

        self.expand = ExpandUsingPosEncodings(
            network_channels, 32, 16, network_channels, multiply=True, learnable_encodings=True)

        # self.t = Transformer(network_channels, 2)

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

        # x = self.pos(x)
        # x = self.initial(x)
        # x = self.expand(x.reshape(-1, 1, network_channels))
        # x = self.t(x)

        x = x.permute(0, 2, 1)
        
        for layer in self.t:
            orig = x
            x = layer(x)
            x = F.leaky_relu(x, 0.2)
        
        x = self.final(x)
        harm = self.osc(x) #/ adjust[self.band_size]
        noise = self.noise(x) #/ adjust[self.band_size]
        return harm, noise


decoder = Decoder().to(device)
gen_optim = Adam(decoder.parameters(), lr=1e-3, betas=(0, 0.9))

disc_encoder = Encoder().to(device)
disc = Discriminator().to(device)
disc_optim = Adam(chain(disc.parameters(), disc_encoder.parameters()), lr=1e-3, betas=(0, 0.9))


def latent_stream(batch_size, overfit=False):
    current = get_latent(batch_size, network_channels)

    while True:
        yield current
        if not overfit:
            current = get_latent(batch_size, network_channels)

def train_gen(batch, z):
    gen_optim.zero_grad()

    # sample from prior
    # z = get_latent(batch[512].shape[0], network_channels)
    # generate
    decoded = decoder(z)
    
    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)

    std_loss = 0
    for k, v in fake_feat.items():
        std_loss = std_loss + torch.abs(v.std() - batch[k].std()).mean()

    # Does the encoding look real?
    j = disc(disc_encoder(fake_feat))

    adv_loss = least_squares_generator_loss(j)

    loss = adv_loss + (std_loss * 0.01)
    loss.backward()
    # clip_grad_norm_(decoder.parameters(), 1)
    gen_optim.step()
    print('G', loss.item())
    return decoded, z


def train_disc(batch, z):
    disc_optim.zero_grad()
    # z = get_latent(batch[512].shape[0], network_channels)
    decoded = decoder(z)

    # add together harmonics and noise
    decoded_sum = {k: sum(v) for k, v in decoded.items()}
    fake_feat = compute_feature_dict(decoded_sum)

    
    fj = disc(disc_encoder(fake_feat))
    rj = disc(disc_encoder(batch))

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()

    # clip_grad_norm_(disc_encoder.parameters(), 1)
    # clip_grad_norm_(disc.parameters(), 1)

    disc_optim.step()
    print('D', loss.item())


@readme
class MultiresolutionAutoencoderWithActivationRefinements19(object):
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
        
        z_stream = latent_stream(self.batch_size, self.overfit)
        
        while True:
            step = next(gan_cycle)
            z = next(z_stream)
            bands, feat = next(stream)

            self.bands = bands

            if step == 'gen':
                decoded, encoded = train_gen(feat, z)
                self.decoded = decoded
                self.encoded = encoded
            else:
                train_disc(feat, z)