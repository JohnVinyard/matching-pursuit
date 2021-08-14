from torch import nn
import torch
import numpy as np
from modules import PositionalEncoding, ResidualStack


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, x):
        x = x.view(-1, self.channels)
        l = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.T)
        attn = attn / np.sqrt(attn.numel())
        attn = torch.softmax(attn.view(-1), dim=0).view(l, l)
        x = torch.matmul(attn, v)
        return x


class LinearOutputStack(nn.Module):
    def __init__(self, channels, layers, out_channels=None, in_channels=None):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.out_channels = out_channels or channels

        core = [
            ResidualStack(channels, layers),
            nn.Linear(channels, self.out_channels)
        ]

        inp = [] if in_channels is None else [nn.Linear(in_channels, channels)]

        self.net = nn.Sequential(*[
            *inp,
            *core
        ])

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.seq = nn.Sequential(
            LinearOutputStack(channels, 3, in_channels=10),
            Attention(channels),
            LinearOutputStack(channels, 3)
        )
        self.length = LinearOutputStack(channels, 2, in_channels=1)
        self.judge = LinearOutputStack(channels * 2, 1, 1)

    def forward(self, x, l):
        n = x.shape[0]
        l = self.length(l.view(1, 1)).repeat(n, 1)
        x = self.seq(x)
        x = torch.cat([l, x], dim=1)
        x = self.judge(x)
        return x


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.max_atoms = 768
        self.length = LinearOutputStack(channels, 2, 1)
        self.pos = PositionalEncoding(1, 16384, 8, channels)
        self.input = LinearOutputStack(channels, 3, in_channels=128 * 3)
        self.attn = Attention(channels)
        self.output = LinearOutputStack(channels, 3, out_channels=10)

    def forward(self, x):
        x = x.view(1, self.channels)
        l = torch.clamp(self.length(x), 0, 1)
        count = int(l * self.max_atoms) + 1

        latent = x.repeat(count, 1)
        noise = torch.zeros_like(latent).normal_(0, 1)
        _, pos = self.pos(torch.linspace(0, 0.9999, count).to(x.device))

        x = torch.cat([latent, noise, pos], dim=1)
        x = self.input(x)
        x = self.attn(x)
        x = self.output(x)
        return x, l


if __name__ == '__main__':
    disc = Discriminator(128)
    gen = Generator(128)
    latent = torch.FloatTensor(128).normal_(0, 1)

    fake, l = gen(latent)
    j = disc(fake, l)

    print(fake.shape, l, j.shape)
