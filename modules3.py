from modules2 import Dilated
from torch.nn.modules.container import Sequential
from gan_modules import RecursiveExpander
from torch import nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import PositionalEncoding, ResidualStack, get_best_matches, init_weights
from torch.nn import functional as F


def unit_norm(x):
    n = torch.norm(x, dim=-1, keepdim=True)
    x = x / (n + 1e-12)
    return x


class Attention(nn.Module):
    def __init__(self, channels, reduce=False):
        super().__init__()
        self.channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

        self.reduce = reduce
        # self.norm = nn.BatchNorm1d(self.channels)

    def forward(self, x):
        batch, time, channels = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = torch.bmm(q, k.permute(0, 2, 1))
        attn = attn / np.sqrt(attn.numel())
        # attn = torch.sigmoid(attn)
        attn = torch.softmax(attn.view(batch, -1), dim=-1).view(batch, time, time)
        x = torch.bmm(attn, v)


        # x = unit_norm(x) * 3.2

        # x = x.permute(0, 2, 1)
        # x = self.norm(x)
        # x = x.permute(0, 2, 1)

                
        if self.reduce:
            v = v.sum(dim=1, keepdim=True)
        return v


class LinearOutputStack(nn.Module):
    def __init__(
            self, channels,
            layers,
            out_channels=None,
            in_channels=None,
            activation=lambda x: F.leaky_relu(x, 0.2),
            bias=True,
            shortcut=True):

        super().__init__()
        self.channels = channels
        self.layers = layers
        self.out_channels = out_channels or channels

        core = [
            ResidualStack(channels, layers, activation=activation, bias=bias, shortcut=shortcut),
            nn.Linear(channels, self.out_channels, bias=self.out_channels > 1)
        ]

        inp = [] if in_channels is None else [
            nn.Linear(in_channels, channels, bias=bias)]

        self.net = nn.Sequential(*[
            *inp,
            *core
        ])

    def forward(self, x):
        x = self.net(x)
        return x


class ToThreeD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(1, 0)[None, ...]


class ToTwoD(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        t = x.shape[-1]
        x = x.permute(0, 2, 1).reshape(t, -1)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels, embedding_weights):
        super().__init__()
        self.channels = channels

        self.atom_embedding = nn.Embedding(512 * 6, 8)

        with torch.no_grad():
            self.atom_embedding.weight.data = torch.from_numpy(
                embedding_weights)

        self.atom_embedding.requires_grad = False

        self.seq = nn.Sequential(
            LinearOutputStack(channels, 3, in_channels=10),
            Attention(channels),
            LinearOutputStack(channels, 3)
        )
        self.length = LinearOutputStack(channels, 2, in_channels=1)
        self.judge = LinearOutputStack(channels * 2, 1, 1)

        self.apply(init_weights)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        nw = self.atom_embedding.weight
        return get_best_matches(nw, embeddings)

    def get_embeddings(self, x):
        atom, time, mag = x
        ae = self.atom_embedding(atom).view(-1, 8)
        pe = time.view(-1, 1)
        me = mag.view(-1, 1)
        return torch.cat([ae, pe, me], dim=-1)

    def forward(self, x, l):

        if isinstance(x, list):
            x = self.get_embeddings(x)

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
        self.input = LinearOutputStack(channels, 3, in_channels=channels + 3)
        self.attn = Attention(channels)
        self.output = LinearOutputStack(channels, 3)

        self.positions = LinearOutputStack(
            channels, 3, out_channels=self.max_atoms + 1)

        self.expand = RecursiveExpander(channels)

        self.to_atom = nn.Sequential(
            ResidualStack(channels, layers=1),
            nn.Linear(128, 8, bias=False)
        )

        self.to_pos = nn.Sequential(
            ResidualStack(channels, layers=1),
            nn.Linear(128, 1, bias=False)
        )

        self.to_magnitude = nn.Sequential(
            ResidualStack(channels, layers=1),
            nn.Linear(128, 1, bias=False)
        )

        self.long_shot = LinearOutputStack(
            channels, 3, out_channels=(self.max_atoms + 1) * 3)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(1, self.channels)
        l = torch.clamp(torch.abs(self.length(x)), 0, 1)
        count = int(l * self.max_atoms) + 1

        z = self.long_shot(x).reshape(self.max_atoms + 1, 3)
        z = z[:count, :]

        # exp = self.expand(x, count)
        latent = x.repeat(count, 1)

        x = torch.cat([latent, z], dim=1)

        x = self.input(x)
        x = self.attn(x)
        encodings = x = self.output(x)

        atoms = self.to_atom(encodings)
        pos = self.to_pos(encodings)
        mags = self.to_magnitude(encodings)
        recon = torch.cat([atoms, pos, mags], dim=-1)
        return recon, l


if __name__ == '__main__':
    disc = Discriminator(128)
    gen = Generator(128)
    latent = torch.FloatTensor(128).normal_(0, 1)
    fake, l = gen(latent)
    j = disc(fake, l)
    print(fake.shape, l, j.shape)
