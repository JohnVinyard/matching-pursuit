
from locale import normalize
from config.dotenv import Config
from data import audio_stream
import zounds
import torch
import numpy as np
from data.datastore import batch_stream
from modules.ddsp import NoiseModel, OscillatorBank
from modules.overfitraw import OverfitRawAudio
from torch.optim import Adam
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature
from train.gan import get_latent
from upsample import ConvUpsample
from util import playable

from modules.phase import AudioCodec, MelScale
from torch.nn import functional as F
from torch import nn
from torch.optim import SGD, Adadelta, Adagrad, AdamW
from torch.distributions import Normal

from util.weight_init import make_initializer

n_samples = 2 ** 15
samplerate = zounds.SR22050()
window_size = 512
step_size = window_size // 2


basis = MelScale()
transformer = AudioCodec(basis)

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(samplerate, 512, scale, 0.1, normalize_filters=True)
aim = AuditoryImage(512, 128, do_windowing=True)

psych_feat = PsychoacousticFeature(kernel_sizes=[128] * 6)


def compute_feature(x):
    # x = aim(fb(x, normalize=False))
    # x, _ = psych_feat.forward(x)
    x = psych_feat.compute_feature_dict(x, constant_window_size=128)
    return x


def compute_loss(a, b):
    a = compute_feature(a)
    b = compute_feature(b)

    loss = 0
    for k, v in a.items():
        print(v.shape)
        loss = loss + F.mse_loss(v, b[k])
    return loss


init_weights = make_initializer(0.05)


class TwoDGenerator(nn.Module):
    def __init__(self, probabilistic=False):
        super().__init__()

        self.probabilistic = probabilistic
        self.ln = nn.Linear(128, 128 * 8)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),  # (4, 8)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1)),  # (8, 16)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)),  # (16, 32)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 8, (4, 4), (2, 2), (1, 1)),  # (32, 64)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(8, 4, (4, 4), (2, 2), (1, 1)),  # (64, 128)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(4, 3 if self.probabilistic else 2,
                               (4, 4), (2, 2), (1, 1)),  # (128, 256)
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.ln(x).reshape(-1, 128, 2, 4)
        x = self.up(x)

        if self.probabilistic:
            mag = x[:, 0, :, :] ** 2

            phase_mean = torch.sin(x[:, 1, :, :]) * 2 * np.pi
            phase_std = (x[:, 2, :, :] ** 2) + 1e-8

            phase = Normal(phase_mean, phase_std)
            return mag, phase
        else:
            mag = x[:, 0, :, :]
            phase = x[:, 1, :, :]

            mag = mag ** 2
            phase = torch.sin(phase) * np.pi * 2

            freqs = torch.from_numpy(basis.center_frequencies)
            freqs = freqs * 2 * np.pi
            # subtract the expected value
            phase = phase - freqs[None, None, :]

            x = torch.cat([mag[:, :, :, None], phase[:, :, :, None]], dim=-1)
            return x


class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.ln = nn.Linear(128, 128 * 4)

        upsample = lambda: nn.UpsamplingNearest2d(scale_factor=2)

        self.up = nn.Sequential(
            upsample(),
            nn.Conv2d(128, 64, (3, 3), (1, 1), (1, 1)),  # (4, 4)
            nn.LeakyReLU(0.2),

            upsample(),
            nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1)),  # (8, 8)
            nn.LeakyReLU(0.2),

            upsample(), 
            nn.Conv2d(32, 16, (3, 3), (1, 1), (1, 1)),  # (16, 16)
            nn.LeakyReLU(0.2),

            upsample(),
            nn.Conv2d(16, 8, (3, 3), (1, 1), (1, 1)),  # (32, 32)
            nn.LeakyReLU(0.2),

            upsample(),
            nn.Conv2d(8, 4, (3, 3), (1, 1), (1, 1)),  # (64, 64)
            nn.LeakyReLU(0.2),

            upsample(),
            nn.Conv2d(4, 2, (3, 3), (1, 1), (1, 1)),  # (128, 128)
        )

        # self.up = ConvUpsample(
            # 128, 128, 4, 128, mode='learned', out_channels=128)

        # self.to_harm = nn.Conv1d(128, 128, 1, 1, 0)
        # self.to_noise = nn.Conv1d(128, 128, 1, 1, 0)

        self.n_samples = n_samples

        self.osc = OscillatorBank(
            input_channels=128,
            n_osc=128,
            n_audio_samples=n_samples,
            activation=torch.sigmoid,
            amp_activation=torch.relu,
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
            activation=torch.relu,
            squared=False,
            mask_after=1)

        self.apply(init_weights)

    def forward(self, x):

        x = self.ln(x).reshape(-1, 128, 2, 2)
        x = self.up(x)

        h = x[:, 0, :, :]
        n = x[:, 1, :, :]

        # x = self.up(x)
        # h = self.to_harm(h)
        # n = self.to_noise(n)
        h = self.osc(h)
        n = self.noise(n)
        return h + n


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    # signal = OverfitRawAudio((1, n_samples,), std=0.01)
    # optim = Adam(signal.parameters(), lr=1e-4, betas=(0, 0.9))

    gen = Generator(n_samples)
    gen_optim = Adam(gen.parameters(), lr=1e-3, betas=(0, 0.9))
    latent = get_latent(1, 128)

    stream = batch_stream(
        Config.audio_path(), '*.wav', 1, n_samples, overfit=True)

    while True:
        gen_optim.zero_grad()

        batch = next(stream)

        o = zounds.AudioSamples(
            batch.squeeze(), samplerate).pad_with_silence()
        batch = torch.from_numpy(batch).float()
        fake = gen(latent)

        # real_feat = compute_feature(batch)
        # fake_feat = compute_feature(fake)

        # loss = F.mse_loss(fake_feat, real_feat)

        loss = compute_loss(fake, batch)

        
        # real_spec = transformer.to_frequency_domain(batch)

        # fake = gen(latent).view(1, n_samples)
        # fake_spec = transformer.to_frequency_domain(fake)

        # real_repr = real_spec[..., 0] * real_spec[..., 1]
        # fake_repr = fake_spec[..., 0] * fake_spec[..., 1]

        # mag_loss = F.mse_loss(fake_spec[..., 0], real_spec[..., 0])
        # phase_loss = F.mse_loss(fake_repr, real_repr)

        # loss = mag_loss + phase_loss

        # mag_loss = F.mse_loss(fake_spec[..., 0], real_spec[..., 0]) * 100
        # phase_loss = F.mse_loss(fake_spec[..., 1], real_spec[..., 1])

        # print('M', mag_loss.item(), 'P', phase_loss.item())

        # loss = mag_loss + phase_loss
        loss.backward()
        gen_optim.step()
        print(loss.item())

        # mag = real_spec[..., 0].data.cpu().numpy().squeeze()
        # phase = real_spec[..., 1].data.cpu().numpy().squeeze()

        # recon = transformer.to_time_domain(real_spec)
        # r = playable(recon, samplerate)

        # fake_mag = fake_spec[..., 0].data.cpu().numpy().squeeze()
        # fake_phase = fake_spec[..., 1].data.cpu().numpy().squeeze()

        # fake_recon = transformer.to_time_domain(fake_spec)
        f = playable(fake, samplerate)
