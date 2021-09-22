from modules4 import unit_norm
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import Sequential
from modules2 import Cluster, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack, ToThreeD, ToTwoD
from torch import device, nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualStack, get_best_matches
from torch.nn import functional as F


class RasterizedDiscriminator(nn.Module):
    
    def __init__(self):
        super().__init__()
        # starting size is (256, 8)

        self.embedding_size = 17

        self.atom_embedding = nn.Embedding(
            512 * 6, self.embedding_size, scale_grad_by_freq=True)
        
        self.init_atoms = set()
        
        self.net = nn.Sequential(
            nn.Conv2d(18, 32, (3, 3), (2, 2), (1, 1)), # (128, 4)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)), # (64, 2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 1), (2, 1), (1, 0)), # (32, 2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 1), (2, 1), (1, 0)), # (16, 2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, (3, 1), (2, 1), (1, 0)), # (8, 2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1024, (3, 1), (2, 1), (1, 0)), # (4, 2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1024, (4, 2), (4, 2), (0, 0)), # (1, 1)
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, (1, 1), (1, 1), (0, 0))
        )
        self.apply(init_weights)
    
    def extract_time(self, embeddings):
        return embeddings[:, -2:-1]

    def extract_mag(self, embeddings):
        return embeddings[:, -1:]

    def extract_atom(self, embeddings):
        return embeddings[:, :17]

    def get_times(self, embeddings):
        return embeddings.view(-1)

    def get_mags(self, embeddings):
        return embeddings.view(-1)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        nw = self.atom_embedding.weight
        return get_best_matches(unit_norm(nw), unit_norm(embeddings))

    def _init_atoms(self, atom):
        indices = set([int(a) for a in atom.view(-1)])
        to_init = indices - self.init_atoms
        self.init_atoms.update(to_init)
        for ti in to_init:
            with torch.no_grad():
                self.atom_embedding.weight[ti] = torch.FloatTensor(
                    17).uniform_(-0.1, 0.1).to(atom.device)

    def get_embeddings(self, x):
        atom, time, mag = x
        self._init_atoms(atom)
        ae = self.atom_embedding.weight[atom.view(-1)].view(-1, self.embedding_size)
        
        pe = time.view(-1, 1)
        me = mag.view(-1, 1)
        return torch.cat([ae, pe, me], dim=-1)
    
    def forward(self, x):
        batch, embedding, slots, depth = x.shape
        x = self.net(x)
        return x

class RasterizedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(128, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 128, (4, 4), (2, 2), (1, 1)), # (8, 8)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, (4, 1), (2, 1), (1, 0)), # (16, 8)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, (4, 1), (2, 1), (1, 0)), # (32, 8)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, (4, 1), (2, 1), (1, 0)), # (64, 8)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, (4, 1), (2, 1), (1, 0)), # (128, 8)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 18, (4, 1), (2, 1), (1, 0)), # (256, 8)
        )
        self.apply(init_weights)
    
    def forward(self, x):
        x = x.view(-1, 128)
        x = self.expand(x).view(-1, 128, 4, 4)
        x = self.net(x)

        atom = torch.clamp(x[:, :17, :, :], -10, 10)
        pos = torch.clamp(x[:, -1:, :, :], 0, 1)

        x = torch.cat([atom, pos], dim=1)
        return x


if __name__ == '__main__':
    z = torch.FloatTensor(8, 128).normal_(0, 1)
    gen = RasterizedGenerator()
    x = gen(z)
    print(x.shape)
    disc = RasterizedDiscriminator()
    j = disc(x)
    print(j.shape)