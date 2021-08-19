from modules2 import Cluster, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack
from torch import nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualStack, get_best_matches


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, heads, layer_norm=True):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.attn = nn.Sequential(*[Attention(channels, layer_norm)
                                  for _ in range(self.heads)])
        self.fc = nn.Linear(channels, channels)
        self.ln = nn.LayerNorm(self.channels)
        self.layer_norm = layer_norm

    def forward(self, x):
        orig = x

        results = None
        for attn in self.attn:
            z = attn(x)
            if results is None:
                results = z
            else:
                results = results + z

        # residual
        results = results + orig
        x = self.fc(results)

        if self.layer_norm:
            x = self.ln(x)
        
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

        self.dense = nn.Sequential(
            LinearOutputStack(channels, 2, in_channels=10),

            MultiHeadAttention(channels, 4),
            LinearOutputStack(channels, 3),

            MultiHeadAttention(channels, 4),
            LinearOutputStack(channels, 3),

            MultiHeadAttention(channels, 4),
            LinearOutputStack(channels, 3),
        )

        # self.net = nn.Sequential(
        #     LinearOutputStack(channels, 2, in_channels=10),

        #     MultiHeadAttention(channels, 2),
        #     Cluster(channels, 64),
        #     LinearOutputStack(channels, 3),

        #     MultiHeadAttention(channels, 2),
        #     Cluster(channels, 16),
        #     LinearOutputStack(channels, 3),

        #     MultiHeadAttention(channels, 2),
        #     Cluster(channels, 4),
        #     LinearOutputStack(channels, 3),

        #     MultiHeadAttention(channels, 2),
        #     Cluster(channels, 1),
        #     LinearOutputStack(channels, 3)
        # )

        self.length = LinearOutputStack(channels, 2, in_channels=1)
        self.final = nn.Linear(channels * 2, 1)

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

        l = self.length(l).repeat(x.shape[0], 1)

        # x = self.net(x)
        x = self.dense(x)

        x = torch.cat([
            x.view(-1, self.channels),
            l.view(-1, self.channels)], dim=1)
        x = self.final(x)
        return x


class VariableExpander(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.max_atoms = 12
        self.channels = channels
        self.to_length = LinearOutputStack(channels, 2, out_channels=1)
        self.var = LinearOutputStack(channels, 3, out_channels=self.max_atoms * self.channels)

    def forward(self, x):
        x = x.view(1, self.channels)
        l = torch.clamp(torch.abs(self.to_length(x)), 0, 1)
        c = int(self.max_atoms * l.item()) + 1

        z = self.var(x).reshape(-1, self.channels)[:c]
        return z, l


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # self.initial = LinearOutputStack(channels, 3, out_channels=16 * self.channels)
        # self.net = nn.Sequential(

        #     MultiHeadAttention(channels, 2),
        #     LinearOutputStack(channels, 2),
        #     Expander(channels, 2), # 32

        #     MultiHeadAttention(channels, 2),
        #     LinearOutputStack(channels, 2),
        #     Expander(channels, 2), # 64

        #     MultiHeadAttention(channels, 2),
        #     LinearOutputStack(channels, 2)
        # )

        # self.variable = VariableExpander(channels)

        self.max_atoms = 768

        self.length = LinearOutputStack(channels, 3, out_channels=1)

        self.pos = PositionalEncoding(1, self.max_atoms + 1, 8, channels)

        self.reduce = LinearOutputStack(
            channels, 3, in_channels=self.pos.freq_channels + channels)

        self.net = nn.Sequential(
            MultiHeadAttention(channels, 4, layer_norm=False),
            LinearOutputStack(channels, 3),

            MultiHeadAttention(channels, 4, layer_norm=False),
            LinearOutputStack(channels, 3),

            MultiHeadAttention(channels, 4, layer_norm=False),
            LinearOutputStack(channels, 3),

            MultiHeadAttention(channels, 4, layer_norm=False),
            LinearOutputStack(channels, 3),
        )

        
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

        # continuous/differentiable length ratio
        length = l = torch.clamp(torch.abs(self.length(x)), 0, 0.9999)

        # discrete length for indexing
        c = int(l * self.max_atoms) + 1


        p1, p2 = self.pos(torch.linspace(0, float(l.item()), c))
        encodings = torch.cat([x.repeat(c, 1), p1], dim=1)
        encodings = self.reduce(encodings)
        encodings = self.net(encodings)


        # x = self.initial(x)
        # x = x.view(16, self.channels)
        # x = self.net(x)

        # length = []
        # atoms = []
        # for cluster in x:
        #     expanded, l = self.variable(cluster)
        #     length.append(l.view(-1) / 64)
        #     atoms.append(expanded)
        
        # encodings = torch.cat(atoms, dim=0)
        # length = torch.sum(torch.cat(length)).view(1, 1)

        atoms = self.to_atom(encodings)
        pos = self.to_pos(encodings)
        mags = self.to_magnitude(encodings)

        recon = torch.cat([atoms, pos, mags], dim=-1)
        return recon, length


if __name__ == '__main__':
    # l = torch.FloatTensor([0.5])
    # x = torch.FloatTensor(700, 128).normal_(0, 1)

    latent = torch.FloatTensor(1, 128).normal_(0, 1)
    g = Generator(128)
    x, l = g(latent)
    print(x.shape, l.shape)

    d = Discriminator(128, np.random.normal(0, 1, (3072, 8)))
    x = d(x, l)
    print(x.shape)
