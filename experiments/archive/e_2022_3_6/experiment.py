import torch
import numpy as np
from config.dotenv import Config
from data.datastore import batch_stream
import zounds
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.linear import LinearOutputStack
from train.gan import get_latent
from util import device

from modules.ddsp import overlap_add
from util.readmedocs import readme
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from itertools import chain
from train import gan_cycle
from util.weight_init import make_initializer


init_weights = make_initializer(0.1)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.mag = nn.Conv2d(1, 8, (3, 3), (1, 1), (1, 1))
        self.phase = nn.Conv2d(1, 8, (3, 3), (1, 1), (1, 1))

        self.down = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), (2, 2), (1, 1)), # (32, 128)
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1)), # (16, 64)
            nn.LeakyReLU(0.2),

            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)), # (8, 32)
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)), # (4, 16)
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1)), # (2, 8)
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, (2, 3), (2, 2), (0, 1)), # (1, 4)
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 128, (1, 4), (1, 4), (0, 0)), # (1, 1)
        )

        self.apply(init_weights)
    
    def forward(self, x):
        mag = x[:, :1, :, :]
        phase = x[:, 1:, :, :]
        mag = self.mag(mag)
        phase = self.phase(phase)
        x = torch.cat([mag, phase], dim=1)
        x = self.down(x)
        x = x.view(-1, 128)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Linear(128, 1024)
        self.up = nn.Sequential(

            nn.ConvTranspose2d(256, 128, (4, 4), (2, 2), (1, 1)), # (4, 4)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)), # (8, 8)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1)), # (16, 16)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)), # (32, 32)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 8, (4, 4), (2, 2), (1, 1)), # (64, 64)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(8, 4, (3, 4), (1, 2), (1, 1)), # (64, 128)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(4, 2, (3, 4), (1, 2), (1, 1)), # (64, 256)
        )

        self.apply(init_weights)
    
    def forward(self, x):
        x = self.initial(x).reshape(-1, 256, 2, 2)
        x = self.up(x)
        mag = x[:, :1, :, :] ** 2
        phase = torch.sin(x[:, 1:, :, :]) * (np.pi * 2)
        x = torch.cat([mag, phase], dim=1)
        x = F.pad(x, (0, 1))
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


def to_spectrogram(audio_batch, window_size, step_size, samplerate):
    batch_size = audio_batch.shape[0]

    audio_batch = F.pad(audio_batch, (0, step_size))
    windowed = audio_batch.unfold(-1, window_size, step_size)
    window = torch.hann_window(window_size).to(audio_batch.device)
    spec = torch.fft.rfft(windowed * window, dim=-1, norm='ortho')
    n_coeffs = (window_size // 2) + 1
    spec = spec.reshape(batch_size, -1, n_coeffs)

    mag = torch.abs(spec) + 1e-12
    phase = torch.angle(spec)
    phase = torch.diff(
        phase, 
        dim=1, 
        prepend=torch.zeros(batch_size, 1, n_coeffs).to(audio_batch.device))
    
    return torch.cat([mag[..., None], phase[..., None]], dim=-1)


def from_spectrogram(spec, window_size, step_size, samplerate):
    print(spec.shape)
    batch_size, time, n_coeffs, _ = spec.shape
    mag = spec[..., 0]
    phase = spec[..., 1]

    real = mag
    imag = torch.cumsum(phase, dim=1)
    imag = (imag + np.pi) % (2 * np.pi) - np.pi

    # spec = torch.complex(real, imag)
    spec = real * torch.exp(1j * imag)
    windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
    signal = overlap_add(windowed[:, None, :, :], apply_window=False)
    return signal


def fake_stream():
    synth = zounds.SineSynthesizer(zounds.SR22050())
    samples = synth.synthesize(zounds.SR22050().frequency  * (2**14), [220, 440, 880])
    yield np.array(samples).reshape((1, 1, -1))
    input()

@readme
class InstaneousFreqExperiment(object):
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
        audio = from_spectrogram(self.encoded[:1, ...].permute(0, 2, 3, 1), self.window_size, self.step_size, int(self.samplerate))
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
            batch = batch.reshape(-1, 1, self.n_samples)
            batch /= (np.abs(batch).max(axis=-1, keepdims=True) + 1e-12)

            self.orig = batch
            batch = torch.from_numpy(batch).to(device)
            with torch.no_grad():
                encoded = to_spectrogram(batch, self.window_size, self.step_size, int(self.samplerate))
                self.spec = encoded

            step = next(gan_cycle)

            if step == 'gen':
                self.encoded = train_gen(encoded)
            else:
                train_disc(encoded)

