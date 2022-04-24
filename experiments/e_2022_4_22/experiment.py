from pyclbr import readmodule_ex

import numpy as np
from modules import stft
from modules.atoms import AudioEvent
from modules.linear import LinearOutputStack
from modules.metaformer import AttnMixer, MetaFormer
from modules.pif import AuditoryImage
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
    loss = torch.abs(f - t).mean(dim=(1, 2, 3))
    return loss, f, t


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


class SyntheticLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.baselines = LinearOutputStack(128, 3, in_channels=n_events)

        self.features = MetaFormer(
            128,
            4,
            lambda channels: AttnMixer(channels),
            lambda channels: lambda x: x,
            return_features=False)

        self.down = nn.Linear(70 * n_events, 128)
        self.params = MetaFormer(
            128,
            4,
            lambda channels: AttnMixer(channels),
            lambda channels: lambda x: x,
            return_features=False)

        self.to_loss = LinearOutputStack(128, 3, out_channels=1)

        self.apply(init_weights)

    def forward(self, audio_feature, baselines, synth_params):
        batch = audio_feature.shape[0]

        audio_feature = audio_feature.view(batch, 128, 64, 257)
        baselines = baselines.view(batch, n_events)
        synth_params = synth_params.view(
            batch, n_events * 70, 64).permute(0, 2, 1)
        synth_params = self.down(synth_params)
        synth_params = self.params(synth_params)[:, -1, :]

        baselines = self.baselines(baselines)

        audio_feature = torch.norm(audio_feature, dim=-1)
        audio_feature = audio_feature.permute(0, 2, 1)
        audio_feature = self.features(audio_feature)[:, -1, :]

        x = baselines + audio_feature + synth_params
        x = self.to_loss(x)

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

        signal = self.atoms.forward(
            f0,
            overall_env,
            osc_env,
            noise_env,
            harm_env,
            noise_std,
            baselines)
        signal = signal.sum(dim=1, keepdim=True) * 0.1
        return signal, baselines, x


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Generator()
        self.apply(init_weights)

    def forward(self, x):
        e = self.encoder(x)
        d, baselines, params = self.decoder(e)
        return e, d, baselines, params


model = AutoEncoder().to(device)
optim = optimizer(model, lr=1e-4)

synth_loss = SyntheticLoss()
loss_optim = optimizer(synth_loss)


def train_loss(batch):
    loss_optim.zero_grad()
    e, d, baselines, params = model(batch)
    loss, recon_feat, real_feat = feature_loss(d, batch)
    sl = synth_loss.forward(recon_feat, baselines, params)

    diff = torch.abs(loss - sl).mean()
    diff.backward()
    loss_optim.step()
    print('SL', diff.item())


def train_model(batch):
    optim.zero_grad()
    e, d, baselines, params = model(batch)
    # loss, recon_feat, real_feat = feature_loss(d, batch)
    recon_feat = compute_feature(d)
    loss = synth_loss.forward(recon_feat, baselines, params).mean()
    loss.backward()
    optim.step()
    print('AE', loss.item())
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
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)

            if i % 2 == 0:
                train_loss(item)
            else:
                self.orig = item
                e, d = train_model(item)
                self.latent = e
                self.recon = d
