import torch
import numpy as np
from config.dotenv import Config
from data.datastore import batch_stream
import zounds
from modules.linear import LinearOutputStack
from modules.transformer import Transformer
from upsample import ConvUpsample
from util import device
from torch.nn import functional as F

from modules import AudioCodec, MelScale
from util.readmedocs import readme
from torch import nn
from torch.optim import Adam
from itertools import chain
from util.weight_init import make_initializer
from torch.distributions import Normal

init_weights = make_initializer(0.1)


class NormalDist(nn.Module):
    def __init__(self, channels, force_positive=False):
        super().__init__()
        self.ln = nn.Linear(channels, channels)
        self.mean = nn.Linear(channels, channels)
        self.std = nn.Linear(channels, channels)
        self.force_positive = force_positive

    def forward(self, x):
        x = self.ln(x)
        mean = self.mean(x)

        if self.force_positive:
            mean = torch.exp(mean) + 1e-12

        std = torch.exp(self.std(x)) + 1e-12
        return mean, std


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_mag = LinearOutputStack(
            128, 2, out_channels=64, in_channels=256)
        self.embed_phase = LinearOutputStack(
            128, 2, out_channels=64, in_channels=256)
        # self.t = Transformer(128, 4)

        self.net = nn.Sequential(
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128, 128, 3, 2, 1),
            nn.LeakyReLU(0.2,),
            nn.Conv1d(128, 128, 2, 2, 0),
        )
        self.final = LinearOutputStack(128, 2)

        self.apply(init_weights)

    def forward(self, x):
        batch, _, time, channels = x.shape
        mag = x[:, :, :, 0]
        phase = x[:, :, :, 1]
        mag = self.embed_mag(mag)
        phase = self.embed_phase(phase)
        x = torch.cat([mag, phase], dim=-1)
        # x = self.t(x)
        # x = x[:, -1, :]

        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.reshape(batch, -1)

        x = self.final(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = ConvUpsample(
            128, 128, 4, 64, mode='learned', out_channels=128)

        self.mag = LinearOutputStack(128, 4, out_channels=256)

        self.phase = LinearOutputStack(128, 4, out_channels=256)
        self.phase_dist = NormalDist(256)

        self.apply(init_weights)

    def forward(self, x):
        x = self.up(x)
        x = x.permute(0, 2, 1)

        mag = self.mag(x) ** 2

        phase = self.phase(x)
        phase_mean, phase_std = self.phase_dist(phase)

        # x = torch.cat([mag[..., None], phase[..., None]], dim=-1)  # (batch, time, channels, 2)
        # return x
        return mag, phase_mean, phase_std


encoder = Encoder().to(device)
gen = Decoder().to(device)
optim = Adam(chain(encoder.parameters(), gen.parameters()), lr=1e-3, betas=(0, 0.9))


def train_ae(batch):
    optim.zero_grad()
    encoded = encoder(batch)
    mag, phase_mean, phase_std = gen(encoded)

    phase_dist = Normal(phase_mean, phase_std)

    phase = phase_dist.rsample()

    decoded = torch.cat([mag[..., None], phase[..., None]], dim=-1)

    phase_prob = -(phase_dist.log_prob(batch[..., 1]).sum()) * 1e-5

    loss = F.mse_loss(decoded, batch) + phase_prob

    loss.backward()
    optim.step()
    print('AE', loss.item())
    return encoded, decoded


codec = AudioCodec(MelScale())


def to_spectrogram(audio_batch, window_size, step_Size, samplerate):
    return codec.to_frequency_domain(audio_batch)


def from_spectrogram(spec, window_size, step_size, samplerate):
    return codec.to_time_domain(spec)


@readme
class InstaneousFreqExperiment3(object):
    def __init__(self, overfit, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.overfit = overfit
        self.n_samples = 2 ** 14
        self.samplerate = zounds.SR22050()

        self.window_size = 512
        self.step_size = 256

        self.orig = None
        self.decoded = None
        self.spec = None

        self.phase_variance = None

    def real(self):
        return zounds.AudioSamples(self.orig[0].squeeze(), self.samplerate).pad_with_silence()

    def fake(self):
        audio = from_spectrogram(self.encoded[:1, ...].data.cpu(
        ), self.window_size, self.step_size, int(self.samplerate))
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()

    @property
    def mag(self):
        return self.encoded[0, :, :, 0].data.cpu().numpy().squeeze()

    @property
    def phase(self):
        return self.encoded[0, :, :, 1].data.cpu().numpy().squeeze()

    @property
    def real_mag(self):
        return self.spec[0, :, :, 0].data.cpu().numpy().squeeze()

    @property
    def real_phase(self):
        return self.spec[0, :, :, 1].data.cpu().numpy().squeeze()

    def run(self):
        stream = batch_stream(
            Config.audio_path(),
            '*.wav',
            self.batch_size,
            self.n_samples,
            overfit=self.overfit)

        for batch in stream:
            batch = batch.reshape(-1, self.n_samples)
            batch /= (np.abs(batch).max(axis=-1, keepdims=True) + 1e-12)

            self.orig = batch
            batch = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                encoded = to_spectrogram(batch, self.window_size, self.step_size, int(
                    self.samplerate)).to(device).float()
                self.spec = encoded

            e, d = train_ae(encoded)
            self.encoded = d
            self.latent = e.data.cpu().numpy()
