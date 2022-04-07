import torch
import zounds
from torch import nn
from modules.ddsp import NoiseModel, OscillatorBank
from train.gan import get_latent, gan_cycle
from train.optim import optimizer
from util import device, playable
from modules import LinearOutputStack

from data.audiostream import audio_stream
from modules.phase import AudioCodec, MelScale
from modules.psychoacoustic import PsychoacousticFeature
from util.readmedocs import readme
from util.weight_init import make_initializer
from loss import least_squares_generator_loss, least_squares_disc_loss
from torch.nn import functional as F

n_samples = 2 ** 14
samplerate = zounds.SR22050()
basis = MelScale()
codec = AudioCodec(basis)

psych_feat = PsychoacousticFeature(kernel_sizes=[128] * 6).to(device)


gen_init = make_initializer(0.12)
disc_init = make_initializer(0.05)


class Discriminator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(390, 128, (3, 3), (2, 2), (1, 1)),  # (32, 16)
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),  # (16, 8)
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),  # (8, 4)
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)),  # (4, 2)
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, (4, 2), (4, 2), (0, 0)),  # (4, 2)

        )
        self.final = nn.Linear(128, 1)

        self.apply(disc_init)

    def forward(self, x):
        feat = psych_feat.compute_feature_dict(x, constant_window_size=128)
        x = torch.cat(list(feat.values()), dim=-1)
        x = x.permute(0, 3, 1, 2)
        x = self.down(x)
        x = x.view(-1, 128)
        x = self.final(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.ln = nn.Linear(128, 128 * 4)

        def upsample(): return nn.UpsamplingNearest2d(scale_factor=2)

        self.up = nn.Sequential(
            upsample(),
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),  # (4, 4)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),

            upsample(),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),  # (8, 8)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(32),

            upsample(),
            nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1)),  # (16, 16)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(16),

            upsample(),
            nn.Conv2d(16, 8, (3, 3), (1, 1), (1, 1)),  # (32, 32)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(8),

            upsample(),
            nn.Conv2d(8, 4, (3, 3), (1, 1), (1, 1)),  # (64, 64)
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(4),

            upsample(),
            nn.Conv2d(4, 2, (3, 3), (1, 1), (1, 1)),  # (128, 128)
        )

        self.n_samples = n_samples

        self.osc = OscillatorBank(
            input_channels=128,
            n_osc=128,
            n_audio_samples=n_samples,
            activation=torch.sigmoid,
            amp_activation=torch.abs,
            return_params=False,
            constrain=True,
            log_frequency=False,
            lowest_freq=20 / samplerate.nyquist,
            sharpen=False,
            compete=False)

        self.noise = NoiseModel(
            input_channels=128,
            input_size=128,
            n_noise_frames=128,
            n_audio_samples=n_samples,
            channels=128,
            activation=lambda x: x,
            squared=False,
            mask_after=1)

        self.apply(gen_init)

    def forward(self, x):
        x = self.ln(x).reshape(-1, 128, 2, 2)
        x = self.up(x)
        h = x[:, 0, :, :]
        n = x[:, 1, :, :]
        h = self.osc(h)
        n = self.noise(n)
        return h + n


gen = Generator(n_samples).to(device)
gen_optim = optimizer(gen)

disc = Discriminator(n_samples).to(device)
disc_optim = optimizer(disc)


def train_gen(batch):
    gen_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    j = disc(fake)
    loss = least_squares_generator_loss(j)
    loss.backward()
    print('G', loss.item())
    gen_optim.step()
    return fake


def train_disc(batch):
    disc_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    fj = disc(fake)
    rj = disc(batch)
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class SingleBandGenerator(object):

    def __init__(self, overfit=False, batch_size=4):
        super().__init__()
        self.overfit = overfit
        self.batch_size = batch_size

        self.generated = None
        self.orig = None

    def real(self):
        return playable(self.orig, samplerate)

    def real_spec(self):
        spec = codec.to_frequency_domain(self.orig.view(-1, n_samples))
        return spec[..., 0].data.cpu().numpy().squeeze()[0]

    def fake(self):
        return playable(self.generated, samplerate)

    def fake_spec(self):
        spec = codec.to_frequency_domain(self.generated.view(-1, n_samples))
        return spec[..., 0].data.cpu().numpy()[0]

    def run(self):
        stream = audio_stream(
            self.batch_size, n_samples, self.overfit, normalize=False, as_torch=True)

        for item in stream:
            self.orig = item

            step = next(gan_cycle)
            if step == 'gen':
                self.generated = train_gen(item)
            else:
                train_disc(item)
