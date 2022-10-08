from pathlib import Path
import torch
from torch import nn
import zounds
from config.experiment import Experiment
from experiments.e_2022_9_12 import autoencoder
from modules.dilated import DilatedStack
from scalar_scheduling import init_weights
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample

from util import device, playable

from util.readmedocs import readme
import numpy as np

init = init_weights(0.05)

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(256, 128)
        self.net = DilatedStack(128, [1, 3, 9, 1])
        self.judge = nn.Linear(128, 1)
        self.apply(init_weights)

    def forward(self, time, transfer):
        # (batch, channels, n_events)
        time = time.view(-1, 16, 128)
        transfer = transfer.view(-1, 16, 128)
        x = torch.cat([time, transfer], dim=-1)
        x = self.embed(x)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=1)
        x = self.judge(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = ConvUpsample(128, 128, 4, 16, mode='learned', out_channels=128)
        self.to_time = nn.Linear(128, 128)
        self.to_transfer = nn.Linear(128, 128)
        self.apply(init_weights)

    def forward(self, x):
        # (batch, channels, n_events)
        x = self.up(x).permute(0, 2, 1)
        # x = self.net(x)
        time = self.to_time(x)
        tf = self.to_transfer(x)
        return time, tf


critic = Critic().to(device)
critic_optim = optimizer(critic, lr=1e-3)


gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-3)


def get_latent(batch_size):
    return torch.zeros(batch_size, 128, device=device).normal_(0, 1)


def train_critic(batch):
    real_time, real_transfer = batch

    critic_optim.zero_grad()
    latent = get_latent(real_time.shape[0])
    time, transfer = gen.forward(latent)
    fj = critic.forward(time, transfer)

    rj = critic.forward(real_time, real_transfer)

    loss = torch.abs(1 - rj.mean()) + torch.abs(0 - fj.mean())
    loss.backward()

    critic_optim.step()

    for p in critic.parameters():
        p.data.clamp_(-0.01, 0.01)
    
    return loss


def train_generator(batch):
    real_time, real_transfer = batch
    gen_optim.zero_grad()

    latent = get_latent(real_time.shape[0])
    time, transfer = gen.forward(latent)
    fj = critic.forward(time, transfer)

    loss = torch.abs(1 - fj.mean())
    loss.backward()

    gen_optim.step()
    return loss, (time, transfer)


@readme
class SequeceGan(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.autoencoder = autoencoder
        self.real = None
        self.fake = None

    def orig(self):
        real_time, real_transfer = self.real
        real_time = real_time[:16]
        real_transfer = real_transfer[:16]
        with torch.no_grad():
            output = self.autoencoder.decode(real_time, real_transfer)
        return playable(output, exp.samplerate)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        fake_time, fake_transfer = self.fake
        with torch.no_grad():
            output = self.autoencoder.decode(fake_time[0], fake_transfer[0])
        return playable(output, exp.samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            with torch.no_grad():
                self.real = self.autoencoder.encode(item)

            if i % 2 == 0:
                loss = train_critic(self.real)
                print('D', loss.item())
            else:
                loss, self.fake = train_generator(self.real)
                print('G', loss.item())
