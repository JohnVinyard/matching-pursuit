from modules2 import Cluster, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack
from torch import nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualStack, get_best_matches
from torch.nn import functional as F


def activation(x):
    return F.leaky_relu(x, 0.2)


def sine_one(x):
    return (torch.sin(x) + 1) * 0.5


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

        self.atom_judge = LinearOutputStack(
            channels, 3, in_channels=10)

        self.dense = nn.Sequential(
            LinearOutputStack(channels, 2, in_channels=10,
                              activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),
        )

        self.length = LinearOutputStack(
            channels, 2, in_channels=1, activation=activation)
        self.final = nn.Linear(channels * 2, channels)

        self.reducer = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(channels, channels, 4, 4, 0),
            nn.Conv1d(channels, 1, 1, 1, 0)
        )

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

        aj = self.atom_judge(x).view(-1, 1)

        l = self.length(l).repeat(x.shape[0], 1)
        x = self.dense(x)

        x = torch.cat([
            x.view(-1, self.channels),
            l.view(-1, self.channels)], dim=1)
        x = self.final(x).view(-1, 1)
        x = torch.cat([x, aj], dim=1)

        x = x.permute(1, 0).reshape(1, self.channels * 2, -1)
        diff = 1024 - x.shape[-1]
        x = F.pad(x, (0, diff))
        x = self.reducer(x)
        x = x.permute(0, 2, 1).reshape(1, 1)
        return x


# class VariableExpander(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.max_atoms = 12
#         self.channels = channels
#         self.to_length = LinearOutputStack(channels, 2, out_channels=1)
#         self.var = LinearOutputStack(
#             channels, 3, out_channels=self.max_atoms * self.channels)

#     def forward(self, x):
#         x = x.view(1, self.channels)
#         l = torch.clamp(torch.abs(self.to_length(x)), 0, 1)
#         c = int(self.max_atoms * l.item()) + 1

#         z = self.var(x).reshape(-1, self.channels)[:c]
#         return z, l


class SetExpansion(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.channels = channels
        self.max_atoms = max_atoms

        self.members = nn.Parameter(torch.FloatTensor(
            max_atoms, channels).normal_(0, 1))

        self.to_query = LinearOutputStack(channels, 3, activation=activation)
        self.pos_encoding = PositionalEncoding(1, self.max_atoms, 8, channels)
        self.reduce = LinearOutputStack(
            channels, 3, in_channels=channels * 2 + self.pos_encoding.freq_channels, activation=activation)
        self.length = LinearOutputStack(
            channels, 3, out_channels=1, activation=activation)

    def forward(self, x):
        x = x.view(1, self.channels)

        # continuous/differentiable length ratio
        length = l = torch.clamp(torch.abs(self.length(x)), 0, 0.9999)

        # discrete length for indexing
        c = int(l * self.max_atoms) + 1

        p1, p2 = self.pos_encoding(torch.linspace(0, 0.9999, self.max_atoms))
        p1 = p1[:c]

        q = self.to_query(x)

        scores = torch.matmul(self.members, q.T).view(-1)
        indices = torch.argsort(scores)[-c:]

        members = self.members[indices]
        x = x.repeat(c, 1)

        x = torch.cat([members, x, p1], dim=1)
        x = self.reduce(x)

        return x, length


class Generator(nn.Module):
    def __init__(self, channels, embedding_weights):
        super().__init__()
        self.channels = channels

        self.register_buffer('embedding', torch.from_numpy(embedding_weights))

        self.max_atoms = 768

        self.set_expansion = SetExpansion(channels, self.max_atoms)

        self.net = nn.Sequential(
            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),

            MultiHeadAttention(channels, 8, layer_norm=False),
            LinearOutputStack(channels, 3, activation=activation),
        )

        # self.to_atom = nn.Sequential(
        #     ResidualStack(channels, layers=1, activation=activation),
        #     nn.Linear(128, 8)
        # )

        # self.to_pos = nn.Sequential(
        #     ResidualStack(channels, layers=1, activation=activation),
        #     nn.Linear(128, 1)
        # )

        # self.to_magnitude = nn.Sequential(
        #     ResidualStack(channels, layers=1, activation=activation),
        #     nn.Linear(128, 1)
        # )

        self.to_atom = LinearOutputStack(
            channels, 3, activation=activation, out_channels=8)
        self.to_pos = LinearOutputStack(
            channels, 3, activation=activation, out_channels=1)
        self.to_magnitude = LinearOutputStack(
            channels, 3, activation=activation, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        encodings, length = self.set_expansion(x)
        encodings = self.net(encodings)

        atoms = self.to_atom(encodings)
        pos = self.to_pos(encodings)
        mags = self.to_magnitude(encodings)

        recon = torch.cat([atoms, pos, mags], dim=-1)
        return recon, length


if __name__ == '__main__':
    # l = torch.FloatTensor([0.5])
    # x = torch.FloatTensor(700, 128).normal_(0, 1)

    latent = torch.FloatTensor(1, 128).normal_(0, 1)
    se = SetExpansion(128, 768)

    x, l = se.forward(latent)
    print(x.shape)

    # g = Generator(128)
    # x, l = g(latent)
    # print(x.shape, l.shape)

    # d = Discriminator(128, np.random.normal(0, 1, (3072, 8)))
    # x = d(x, l)
    # print(x.shape)
