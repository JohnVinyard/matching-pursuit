import torch
import numpy as np
from config.dotenv import Config
from data.datastore import batch_stream
import zounds
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.linear import LinearOutputStack
from modules.transformer import Transformer
from train.gan import get_latent
from upsample import ConvUpsample, Linear
from util import device

from modules.ddsp import overlap_add
from modules import AudioCodec, MelScale
from util.readmedocs import readme
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from itertools import chain
from train import gan_cycle
from util.weight_init import make_initializer


init_weights = make_initializer(0.02)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_mag = LinearOutputStack(128, 2, out_channels=64, in_channels=256)
        self.embed_phase = LinearOutputStack(128, 2, out_channels=64, in_channels=256)
        self.t = Transformer(128, 4)
        self.apply(init_weights)
    
    def forward(self, x):
        batch, _, time, channels = x.shape
        mag = x[:, 0, :, :]
        phase = x[:, 1, :, :]
        mag = self.embed_mag(mag)
        phase = self.embed_phase(phase)
        x = torch.cat([mag, phase], dim=-1)
        x = self.t(x)
        x = x[:, -1, :]
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = ConvUpsample(128, 128, 4, 64, mode='nearest', out_channels=128)
        # self.t = Transformer(128, 3)
        self.mag = LinearOutputStack(128, 2, out_channels=256)
        self.phase = LinearOutputStack(128, 2, out_channels=256)
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.up(x)
        x = x.permute(0, 2, 1)
        # x = self.t(x)
        mag = self.mag(x) ** 2
        phase = self.phase(x)
        x = torch.cat([mag[..., None], phase[..., None]], dim=-1) # (batch, time, channels, 2)
        x = x.permute(0, 3, 1, 2)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = LinearOutputStack(128, 3, out_channels=1)
        self.apply(init_weights)
    
    def forward(self, x):
        return self.net(x)



encoder = Encoder().to(device)
disc = Discriminator().to(device)
disc_optim = Adam(chain(encoder.parameters(), disc.parameters()), lr=1e-3, betas=(0, 0.9))

gen = Decoder().to(device)
gen_optim = Adam(gen.parameters(), lr=1e-3, betas=(0, 0.9))


def train_gen(batch):
    gen_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    decoded = gen(z)

    encoded = encoder(decoded)
    j = disc(encoded)
    loss = least_squares_generator_loss(j)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return decoded


def train_disc(batch):
    disc_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    decoded = gen(z)

    fe = encoder(decoded)
    fj = disc(fe)

    re = encoder(batch.permute(0, 3, 1, 2))
    rj = disc(re)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


codec = AudioCodec(MelScale())

def to_spectrogram(audio_batch, window_size, step_Size, samplerate):
    return codec.to_frequency_domain(audio_batch)

def from_spectrogram(spec, window_size, step_size, samplerate):
    return codec.to_time_domain(spec)


def fake_stream():
    synth = zounds.SineSynthesizer(zounds.SR22050())
    samples = synth.synthesize(zounds.SR22050().frequency  * (2**14), [220, 440, 880])
    yield np.array(samples).reshape((1, 1, -1))
    input()

@readme
class InstaneousFreqExperiment2(object):
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
    
    def real(self):
        return zounds.AudioSamples(self.orig[0].squeeze(), self.samplerate).pad_with_silence()
    
    def fake(self):
        audio = from_spectrogram(self.encoded[:1, ...].permute(0, 2, 3, 1).data.cpu(), self.window_size, self.step_size, int(self.samplerate))
        return zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), self.samplerate).pad_with_silence()
    
    @property
    def mag(self):
        return self.encoded[0, 0, :, :].data.cpu().numpy().squeeze()
    
    @property
    def phase(self):
        return self.encoded[0, 1, :, :].data.cpu().numpy().squeeze()
    
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
                encoded = to_spectrogram(batch, self.window_size, self.step_size, int(self.samplerate)).to(device).float()
                self.spec = encoded

            step = next(gan_cycle)

            if step == 'gen':
                self.encoded = train_gen(encoded)
            else:
                train_disc(encoded)

