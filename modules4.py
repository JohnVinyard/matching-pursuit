from torch.nn.modules.container import Sequential
from modules2 import Cluster, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack, ToThreeD, ToTwoD
from torch import nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualStack, get_best_matches
from torch.nn import functional as F


def activation(x):
    return F.leaky_relu(x, 0.2)


# class Activation(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, x):
#         return torch.sin(x)


def sine_one(x):
    return (torch.sin(x) + 1) * 0.5


def unit_norm(x):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=1, keepdims=True)
    else:
        n = torch.norm(x, dim=1, keepdim=True)
    return x / (n + 1e-12)


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.attn = nn.Sequential(*[Attention(channels) for _ in range(self.heads)])
        self.fc = nn.Linear(channels * heads, channels)

    def forward(self, x):
        orig = x

        results = []
        for attn in self.attn:
            z = attn(x)
            results.append(z)

        x = torch.cat(results, dim=-1)
        x = self.fc(x)
        x = x + orig


        return x


class AttentionClusters(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.clusters = nn.Sequential(
            *[Attention(channels, reduce=True) for _ in range(self.heads)])

    def forward(self, x):
        batch, time, channels = x.shape
        # x = x.view(-1, self.channels)
        clusters = []
        for cluster in self.clusters:
            z = cluster(x)
            clusters.append(z)

        x = torch.cat(clusters, dim=1)
        return x


class AttentionStack(nn.Module):
    def __init__(
            self,
            channels,
            attention_heads,
            attention_layers,
            intermediate_layers=3):

        super().__init__()
        self.channels = channels
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_layers = intermediate_layers

        self.net = nn.Sequential(*[
            nn.Sequential(
                MultiHeadAttention(channels, attention_heads),
                LinearOutputStack(channels, intermediate_layers, activation=activation))
            for _ in range(attention_layers)
        ])

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, channels, dense_judgments):
        super().__init__()
        self.channels = channels
        self.dense_judgements = dense_judgments

        self.atom_embedding = nn.Embedding(
            512 * 6, 8, max_norm=1, scale_grad_by_freq=True)

        # with torch.no_grad():
        #     self.atom_embedding.weight.data = torch.from_numpy(
        #         embedding_weights)

        # self.atom_embedding.requires_grad = False

        self.time_amp = LinearOutputStack(channels, 3, in_channels=2)
        self.atom = LinearOutputStack(channels, 3, in_channels=3072)
        self.combine = LinearOutputStack(channels, 3, in_channels=channels * 2)

        self.atom_judge = LinearOutputStack(
            channels, 3, in_channels=10)

        self.dense = nn.Sequential(
            LinearOutputStack(channels, 2,
                              activation=activation, in_channels=10),


            AttentionStack(
                channels,
                attention_heads=8,
                attention_layers=6,
                intermediate_layers=2)
        )

        self.length = LinearOutputStack(
            channels, 2, in_channels=1, activation=activation)
        self.final = nn.Linear(channels, channels)
        self.final_final = LinearOutputStack(
            channels, 3, in_channels=channels * 2)

        self.dense_judge = LinearOutputStack(channels, 3, out_channels=1)

        self.reduce = nn.Sequential(
            AttentionClusters(channels, 16),
            LinearOutputStack(channels, 2),

            AttentionClusters(channels, 4),
            LinearOutputStack(channels, 2),

            AttentionClusters(channels, 1),
            LinearOutputStack(channels, 2, out_channels=1)
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

    def forward(self, x):
        batch, time, channels = x.shape

        # if isinstance(x, list):
        #     x = self.get_embeddings(x)

        aj = self.atom_judge(x)

        x = self.dense(x)

        x = self.final(x)
        x = torch.cat([x, aj], dim=-1)
        x = self.final_final(x)

        j = self.dense_judge(x)

        if self.dense_judgements:
            return j

        j = j.mean(dim=1, keepdim=True)
        x = self.reduce(x)
        x = torch.cat([x, j], dim=-1)
        return x


class SetExpansion(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.channels = channels
        self.max_atoms = max_atoms

        self.register_buffer('members', torch.FloatTensor(
            self.max_atoms, channels).normal_(0, 1))

        self.bias = LinearOutputStack(channels, 3, activation=activation)
        self.weight = LinearOutputStack(channels, 3, activation=activation)

    def forward(self, x):
        x = x.view(-1, self.channels)

        b = self.bias(x)
        w = self.weight(x)

        x = ((self.members[None, ...] * w[:, None, :]) + b[:, None, :])

        return x


class ResidualUpscale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.expander = Expander(channels, factor=2)
        self.stack = LinearOutputStack(channels, 2)
        self.attn = MultiHeadAttention(channels, 4)

    def forward(self, x):
        exp = self.expander(x)
        x = self.stack(exp)
        x = x + exp
        x = self.attn(x)
        return x


class ConvExpander(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.max_atoms = 100
        # self.ln = nn.Linear(channels, channels * 8)
        self.ln = Expander(channels, factor=8)
        n_layers = int(np.log2(128) - np.log2(8))
        self.net = nn.Sequential(
            # ToThreeD(),
            nn.Sequential(*[
                ResidualUpscale(channels)
                for _ in range(n_layers)]),
            # ToTwoD()
        )

    def forward(self, x):
        x = x.view(-1, 1, self.channels)
        x = self.ln(x)
        x = self.net(x)
        return x[:, :self.max_atoms, :]


class RnnGenerator(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.max_atoms = max_atoms
        self.channels = channels
        self.rnn_layers = 4
        self.rnn = nn.RNN(channels, channels,
                          self.rnn_layers, nonlinearity='relu')

    def forward(self, x):
        x = x.view(-1, 1, self.channels)
        batch = x.shape[0]

        # input in shape (sequence_length, batch_size, input_dim)
        # hidden in shape (num_rnn_layers, batch, hidden_dim)
        inp = torch.zeros(1, batch, self.channels).to(x.device)
        hid = torch.zeros(self.rnn_layers, batch, self.channels).to(x.device)

        hid[0, :, :] = x.squeeze()

        seq = []
        for i in range(self.max_atoms):
            inp, hid = self.rnn.forward(inp, hid)
            inp = unit_norm(inp)
            hid = unit_norm(hid)
            # print(inp.shape, hid.shape)
            x = inp
            seq.append(x)

        seq = torch.cat(seq, dim=0)  # time, batch, channels
        seq = seq.permute(1, 0, 2).contiguous()
        return seq


class Generator(nn.Module):
    def __init__(self, channels, embeddings):
        super().__init__()
        self.channels = channels
        self.embeddings = [embeddings]

        self.max_atoms = 100

        self.rnn = RnnGenerator(channels, 100)

        self.set_expansion = SetExpansion(channels, self.max_atoms)
        self.conv_expander = ConvExpander(channels)

        self.net = AttentionStack(
            channels,
            attention_heads=8,
            attention_layers=8,
            intermediate_layers=2)

        self.atoms = LinearOutputStack(channels, 3, out_channels=3072)
        self.pos_loc = LinearOutputStack(channels, 3, out_channels=2)

        self.apply(init_weights)

    def forward(self, x):

        # Expansion
        # encodings = self.conv_expander(x)

        # Set Expansion
        encodings = self.set_expansion(x)

        # RNN
        # encodings = self.rnn(x)

        # End attention stack
        encodings = self.net(encodings)

        # softmax with disc embeddings
        e = self.embeddings[0].weight.clone()
        atoms = torch.softmax(self.atoms(encodings), dim=1)
        atoms = atoms @ e

        # atoms = torch.sin(self.atoms(encodings))

        pt = sine_one(self.pos_loc(encodings))

        recon = torch.cat([atoms, pt], dim=-1)

        return recon


if __name__ == '__main__':
    ew = np.random.normal(0, 1, (3072, 8))

    gen = Generator(128)
    latent = torch.FloatTensor(8, 128).normal_(0, 1)
    generated = gen.forward(latent)
    print(generated.shape)

    disc = Discriminator(128, ew)
    j = disc.forward(generated)
    print(j.shape)
