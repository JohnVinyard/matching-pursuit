from torch import nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import PositionalEncoding, ResidualStack, get_best_matches, init_weights


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

        self.atom_j = LinearOutputStack(channels, 3, out_channels=1, in_channels=10)

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

        aj = self.atom_j(x)

        x = self.seq(x)
        x = torch.cat([l, x], dim=1)
        x = self.judge(x)
        return torch.cat([x, aj], dim=1)


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.max_atoms = 768
        self.length = LinearOutputStack(channels, 2, 1)
        self.pos = PositionalEncoding(1, 16384, 8, channels)
        self.input = LinearOutputStack(channels, 3, in_channels=128 * 3)
        self.attn = Attention(channels)
        self.output = LinearOutputStack(channels, 3)

        self.positions = LinearOutputStack(channels, 3, out_channels=self.max_atoms + 1)

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

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(1, self.channels)
        l = torch.clamp(torch.abs(self.length(x)), 0, 1)
        count = int(l * self.max_atoms) + 1

        p = torch.clamp(self.positions(x).view(-1)[:count], 0, 0.9999)

        latent = x.repeat(count, 1)
        noise = torch.zeros_like(latent).normal_(0, 1)

        
        _, pos = self.pos(p)
        x = torch.cat([latent, noise, pos], dim=1)
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
