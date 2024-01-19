from pyclbr import readmodule_ex
from re import M

import numpy as np
from modules import stft
from modules.atoms import AudioEvent
from modules.linear import LinearOutputStack
from modules.metaformer import AttnMixer, MetaFormer
from modules.pif import AuditoryImage
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
from train import gan_cycle, get_latent
from loss import least_squares_disc_loss, least_squares_generator_loss
from train.optim import optimizer
from upsample import ConvUpsample, Linear
from util import device, playable

from util.readmedocs import readme
import torch
import zounds
from torch import nn
from torch.nn import functional as F

from util.weight_init import make_initializer


init_weights = make_initializer(0.1)

latent_dim = 128

n_events = 8
sequence_length = 64
n_harmonics = 64
log_amplitude = True

n_samples = 2 ** 14
sr = zounds.SR22050()


def compute_feature(x):
    x = stft(x, pad=True, log_amplitude=log_amplitude).view(-1, 64, 257)
    return x


class DownsamplingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = LinearOutputStack(128, 2, in_channels=257)
        self.net = nn.Sequential(
            nn.Conv1d(128, 128, 7, 4, 3),  # 16
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 7, 4, 3),  # 4
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 1, 4, 4, 0),
        )

    def forward(self, x):
        # x = stft(x, pad=True, log_amplitude=log_amplitude).view(-1, 64, 257)
        x = compute_feature(x)
        x = self.embedding(x).permute(0, 2, 1)
        x = self.net(x)
        x = x.view(-1, 1)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = LinearOutputStack(128, 2, in_channels=257)

        self.embed_pos = nn.Linear(33, 128)

        self.net = MetaFormer(
            128,
            5,
            lambda channels: AttnMixer(channels),
            lambda channels: nn.LayerNorm((64, channels)),
            return_features=False)

        # self.net = Transformer(64, 6)
        self.up = nn.Linear(128, 1)

    def forward(self, x):
        pos = pos_encoded(x.shape[0], 64, 16, x.device)
        pos = self.embed_pos(pos)

        x = compute_feature(x)
        x = self.embedding(x)

        x = pos + x

        x = self.net(x)
        x = x[:, -1, :]
        x = self.up(x)
        return x


class AtomGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = ConvUpsample(
            128, 128, 4, 64, mode='nearest', out_channels=70)

    def forward(self, x):
        params = self.up(x)
        return params


class Generator(nn.Module):
    def __init__(self, n_atoms=n_events):
        super().__init__()
        self.atoms = AudioEvent(
            sequence_length=sequence_length,
            n_samples=n_samples,
            n_events=n_events,
            min_f0=50,
            max_f0=8000,
            n_harmonics=n_harmonics,
            sr=sr,
            noise_ws=512,
            noise_step=256)
        self.atom_gen = AtomGenerator()
        self.n_atoms = n_atoms
        self.ln = LinearOutputStack(128, 3, out_channels=128 * n_atoms)
        # self.baseline = LinearOutputStack(128, 3, out_channels=n_events)
        n_rooms = 8
        self.to_room = LinearOutputStack(128, 3, out_channels=n_rooms)
        self.to_mix = LinearOutputStack(128, 2, out_channels=1)
        self.verb = NeuralReverb(n_samples, n_rooms)

    def forward(self, x, inject_noise=False):

        x = x.view(-1, latent_dim)

        verb_mix = self.to_room(x)
        mix = torch.sigmoid(self.to_mix(x))
        # b = self.baseline(x).view(-1, n_events, 1)

        x = self.ln(x)
        x = x.view(-1, 128)
        atoms = self.atom_gen(x)

        x = atoms.view(-1, self.n_atoms, 70, 64)
        # baselines = b

        if inject_noise:
            baselines = baselines + \
                torch.zeros_like(baselines).normal_(0, 0.01)
            x = x + torch.zeros_like(x).normal_(0, 0.01)

        # scale and shift
        # baselines = (baselines + 1) / 2
        # x = (x + 1) / 2

        # baselines = torch.clamp(baselines, -1, 1)
        # x = torch.clamp(x, -1, 1)

        # x = torch.sigmoid(x)
        # baselines = torch.sigmoid(baselines)


        f0 = x[:, :, 0, :]
        osc_env = x[:, :, 1, :]
        noise_env = x[:, :, 2, :]
        overall_env = x[:, :, 3, :]
        noise_std = x[:, :, 4, :]
        harm_env = x[:, :, 5:-1, :]

        signal = self.atoms.forward(
            f0,
            overall_env,
            osc_env,
            noise_env,
            harm_env,
            noise_std,
            # baselines
        )
        dry = signal.mean(dim=1, keepdim=True)
        wet = self.verb.forward(dry, verb_mix).mean(dim=1, keepdim=True)

        signal = ((1 - mix)[:, None, :] * dry) + (wet * mix[:, None, :])
        return signal


# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.encoder = Encoder()
#         self.encoder = DownsamplingEncoder()
#         self.decoder = Generator()
#         self.apply(init_weights)

#     def forward(self, x, inject_noise=False):
#         e = self.encoder(x)
#         d, baselines, params = self.decoder(e, inject_noise=inject_noise)
#         return e, d, baselines, params



gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-4)

disc = Encoder().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(batch):
    gen_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    j = disc(fake)
    loss = least_squares_generator_loss(j)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
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
class SynthParamsDecoderGan(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.orig = None
        self.recon = None

    def real(self):
        return playable(self.orig, sr)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.real()))

    def fake(self):
        return playable(self.recon, sr)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.fake()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.orig = item

            step = next(gan_cycle)
            if step == 'gen':
                self.recon = train_gen(item)
            else:
                train_disc(item)

