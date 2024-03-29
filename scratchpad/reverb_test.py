from pyrsistent import m
import torch
import numpy as np
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.pos_encode import ExpandUsingPosEncodings
from modules.self_similarity import SelfSimNetwork, self_sim
import zounds
from data import audio_stream
from torch import nn
from train.gan import get_latent
from loss import least_squares_disc_loss, least_squares_generator_loss
from train.optim import optimizer
from util import device, playable
from util.weight_init import make_initializer
from torch.nn import functional as F

init_weights = make_initializer(0.1)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.expand = nn.Linear(128, 512)

        self.net = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(128, 128, 3, 1, 1),
        )

        self.osc = OscillatorBank(
            128, 128, 2**14, constrain=True, lowest_freq=0.001, complex_valued=True)
        self.noise = NoiseModel(128, 64, 512, 2**14, 128)
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.expand(x).view(-1, 128, 4)

        x = self.net(x)

        osc = self.osc(x)
        noise = self.noise(x)
        x = osc + noise
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.sim = SelfSimNetwork(128, 5)
        self.final = nn.Linear(128, 1)
        self.frames = LinearOutputStack(128, 4, out_channels=1, in_channels=512)
        self.apply(init_weights)
    
    def forward(self, x):
        x, y = self.sim(x)
        y = self.frames(y)
        y = y.mean(dim=-2)
        x = self.final(x)
        return torch.cat([x.view(-1), y.view(-1)])

gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-4)

disc = Discriminator().to(device)
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

    rj = disc(batch)
    fj = disc(fake)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())
    
if __name__ == '__main__':

    # signal = torch.zeros(4, 1, 2**14)
    model = SelfSimNetwork(128, 4).to(device)
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)
    stream = audio_stream(4, 2**14, overfit=False, normalize=True)

    sr = zounds.SR22050()

    def listen():
        return playable(recon, sr)

    for i, sample in enumerate(stream):
        sample = sample.view(-1, 1, 2**14)
        if i % 2 == 0:
            recon = train_gen(sample)
        else:
            train_disc(sample)
