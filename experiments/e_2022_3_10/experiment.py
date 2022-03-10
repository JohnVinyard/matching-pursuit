from config.dotenv import Config
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from upsample import ConvUpsample
from util import readme, device
import zounds
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
from itertools import chain
from train import gan_cycle, get_latent

n_samples = 2 ** 14
samplerate = zounds.SR22050()
scale = zounds.MelScale(zounds.FrequencyBand(20, samplerate.nyquist), 512)
fb = zounds.learn.FilterBank(
    samplerate, 512, scale, 0.1, normalize_filters=True).to(device)
aim_feature = AuditoryImage(512, 32).to(device)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 8, (4, 2, 4), (4, 2, 4)),  # (128, 16, 64)
            nn.LeakyReLU(0.2),
            nn.Conv3d(8, 16, (4, 2, 4), (4, 2, 4)),  # (32, 8, 16)
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 32, (4, 2, 4), (4, 2, 4)),  # (8, 4, 4)
            nn.LeakyReLU(0.2),
            nn.Conv3d(32, 64, (2, 2, 2), (2, 2, 2)),  # (4, 2, 2)
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, (2, 2, 2), (2, 2, 2)),  # (2, 1, 1)
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, (2, 1, 1), (2, 1, 1)),  # (2, 1, 1)
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = x[:, None, :, :, :]
        x = self.net(x).reshape(x.shape[0], 128)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = LinearOutputStack(128, 3, out_channels=1)

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = ConvUpsample(
            channels, channels, 4, 32, 'fft_learned', channels)

        self.osc = OscillatorBank(
            input_channels=channels,
            n_osc=128,
            n_audio_samples=n_samples,
            activation=torch.sigmoid,
            amp_activation=torch.abs,
            return_params=False,
            constrain=True,
            log_frequency=False,
            lowest_freq=0.05,
            sharpen=False,
            compete=False)

        self.noise = NoiseModel(
            input_channels=channels,
            input_size=32,
            n_noise_frames=64,
            n_audio_samples=n_samples,
            channels=channels,
            activation=lambda x: x,
            squared=False,
            mask_after=1)

    def forward(self, x):
        x = self.up(x)
        harm = self.osc(x)
        noise = self.noise(x)
        return harm + noise


gen = Generator(128).to(device)
optim = Adam(gen.parameters(), lr=1e-3, betas=(0, 0.9))


encoder = Encoder().to(device)
disc = Discriminator().to(device)
disc_optim = Adam(chain(encoder.parameters(),
                        disc.parameters()), lr=1e-3, betas=(0, 0.9))


def compute_feature(audio):
    spec = fb.forward(audio, normalize=False)
    pif = aim_feature(spec)
    return pif


def train_gen(batch):
    optim.zero_grad()

    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    fake_feat = compute_feature(fake)

    encoded = encoder(fake_feat)
    fj = disc(encoded)

    loss = least_squares_generator_loss(fj)
    loss.backward()
    optim.step()
    print('G', loss.item())
    return fake


def train_disc(batch):
    disc_optim.zero_grad()

    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    fake_feat = compute_feature(fake)

    real_encoded = encoder(batch)
    rj = disc(real_encoded)

    fake_encoded = encoder(fake_feat)
    fj = disc(fake_encoded)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class SingleResolutionPif(object):
    def __init__(self, batch_size, overfit):
        super().__init__()
        self.batch_size = batch_size
        self.overfit = overfit
        self.actual = None
        self.estimate = None

    def real(self):
        return zounds.AudioSamples(
            self.actual[0].data.cpu().numpy().squeeze(), samplerate).pad_with_silence()

    def fake(self):
        return zounds.AudioSamples(
            self.estimate[0].data.cpu().numpy().squeeze(), samplerate).pad_with_silence()

    def run(self):
        stream = batch_stream(
            Config.audio_path(),
            '*.wav',
            self.batch_size,
            n_samples,
            overfit=self.overfit,
            normalize=True)

        for samples in stream:

            actual = torch.from_numpy(samples).to(device).float()
            self.actual = actual
            batch = compute_feature(actual)

            step = next(gan_cycle)
            if step == 'gen':
                self.estimate = train_gen(batch)
            else:
                train_disc(batch)
