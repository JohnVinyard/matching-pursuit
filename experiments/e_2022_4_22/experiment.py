from pyclbr import readmodule_ex

import numpy as np
from modules import stft
from modules.atoms import AudioEvent
from modules.linear import LinearOutputStack
from modules.metaformer import AttnMixer, MetaFormer
from modules.pif import AuditoryImage
from modules.transformer import Transformer
from train.optim import optimizer
from util import device, playable

from util.readmedocs import readme
import torch
import zounds
from torch import nn
from torch.nn import functional as F

from util.weight_init import make_initializer


init_weights = make_initializer(0.02)

latent_dim = 128

n_events = 16
sequence_length = 64
n_harmonics = 64

n_samples = 2 ** 14
sr = zounds.SR22050()
band = zounds.FrequencyBand(20, sr.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    sr,
    512,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)
aim = AuditoryImage(512, 64, do_windowing=True, check_cola=True).to(device)


def compute_feature(x):
    x = fb(x, normalize=False)
    x = aim(x)
    return x


def feature_loss(inp, target):
    f = compute_feature(inp)
    t = compute_feature(target)
    return F.mse_loss(f, t)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = LinearOutputStack(128, 2, in_channels=257)

        self.net = MetaFormer(
            128,
            6,
            lambda channels: AttnMixer(channels),
            lambda channels: lambda x: x,
            return_features=False)

        # self.net = Transformer(64, 6)
        self.up = nn.Linear(128, 128)

    def forward(self, x):
        x = stft(x, pad=True, log_amplitude=True).view(-1, 64, 257)
        x = self.embedding(x)
        x = self.net(x)
        x = x[:, -1, :]
        x = self.up(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_atoms=n_events):
        super().__init__()
        self.atoms = AudioEvent(
            sequence_length=sequence_length,
            n_samples=n_samples,
            n_events=n_events,
            min_f0=20,
            max_f0=1000,
            n_harmonics=n_harmonics,
            sr=sr,
            noise_ws=512,
            noise_step=256)
        self.n_atoms = n_atoms
        self.ln = LinearOutputStack(128, 3, out_channels=128 * n_atoms)
        self.net = LinearOutputStack(128, 5, out_channels=70 * 64)
        self.baselines = LinearOutputStack(128, 3, out_channels=1)

    def forward(self, x):

        x = x.view(-1, latent_dim)
        x = self.ln(x)
        x = x.view(-1, self.n_atoms, 128)
        baselines = self.baselines(x).view(-1, self.n_atoms, 1)
        x = self.net(x)
        x = x.view(-1, self.n_atoms, 70, 64)

        # scale and shift
        baselines = baselines * 0.5
        x = (x + 1) / 2

        f0 = x[:, :, 0, :] ** 2
        osc_env = x[:, :, 1, :] ** 2
        noise_env = x[:, :, 2, :] ** 2
        overall_env = x[:, :, 3, :]
        noise_std = x[:, :, 4, :]
        harm_env = x[:, :, 5:-1, :]

        x = self.atoms.forward(
            f0,
            overall_env,
            osc_env,
            noise_env,
            harm_env,
            noise_std,
            baselines)
        x = x.sum(dim=1, keepdim=True)
        return x * 0.1


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Generator()
        self.apply(init_weights)

    def forward(self, x):
        e = self.encoder(x)
        d = self.decoder(e)
        return e, d


model = AutoEncoder().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()

    e, d = model(batch)

    # real_spec = stft(batch, log_amplitude=True)
    # fake_spec = stft(d, log_amplitude=True)

    # loss = F.mse_loss(fake_spec, real_spec)

    loss = feature_loss(d, batch)
    loss.backward()
    optim.step()
    print(loss.item())
    return e, d


@readme
class SynthParamsDecoder(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.orig = None
        self.latent = None
        self.recon = None

    def real(self):
        return playable(self.orig, sr)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.real()))

    def fake(self):
        return playable(self.recon, sr)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.fake()))

    def z(self):
        return self.latent.data.cpu().numpy().squeeze()

    def run(self):
        for item in self.stream:
            item = item.view(-1, 1, n_samples)
            self.orig = item
            e, d = train_model(item)
            self.latent = e
            self.recon = d
