from modules2 import Cluster, Expander, init_weights
from modules3 import Attention, LinearOutputStack
from torch import nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualStack, get_best_matches



class Discriminator(nn.Module):
    def __init__(self, channels, embedding_weights):
        super().__init__()
        self.channels = channels

        self.atom_embedding = nn.Embedding(512 * 6, 8)

        with torch.no_grad():
            self.atom_embedding.weight.data = torch.from_numpy(
                embedding_weights)

        self.atom_embedding.requires_grad = False

        self.length = LinearOutputStack(channels, 2, in_channels=1)
        self.rnn_layers = 4

        self.network = nn.RNN(
            channels,
            channels,
            self.rnn_layers,
            batch_first=False,
            nonlinearity='relu')

        
        self.initial = LinearOutputStack(channels, 3, in_channels=10)
        self.final = LinearOutputStack(
            channels, 3, out_channels=1, in_channels=5 * channels)
        
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

        l = self.length(l)

        x = self.initial(x)

        inp = x.view(-1, 1, self.channels)
        hid = torch.zeros(self.rnn_layers, 1, self.channels).to(x.device)

        inp, hid = self.network(inp, hid)

        x = torch.cat([
            l.view(1, self.channels),
            hid.view(1, self.channels * self.rnn_layers)], dim=1)
        x = self.final(x)
        return x


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.length = LinearOutputStack(self.channels, 3, out_channels=1)
        self.max_atoms = 768
        self.rnn_layers = 4
        self.rnn = nn.RNN(
            channels,
            channels,
            self.rnn_layers,
            batch_first=False,
            nonlinearity='relu')

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
        c = int(l * self.max_atoms) + 1

        # input in shape (sequence_length, batch_size, input_dim)
        # hidden in shape (num_rnn_layers, batch, hidden_dim)
        inp = torch.zeros(1, 1, self.channels).to(x.device)
        hid = torch.zeros(self.rnn_layers, 1, self.channels).to(x.device)
        hid[0, :, :] = x

        seq = []
        for i in range(c):
            inp, hid = self.rnn.forward(inp, hid)
            x = inp.view(1, self.channels)
            seq.append(x)

        encodings = torch.cat(seq, dim=0)

        atoms = self.to_atom(encodings)
        pos = self.to_pos(encodings)
        mags = self.to_magnitude(encodings)
        recon = torch.cat([atoms, pos, mags], dim=-1)

        return recon, l


if __name__ == '__main__':
    seq = torch.FloatTensor(613, 128).normal_(0, 1)
    d = Discriminator(128, np.random.normal(0, 1, (3072, 8)))
    l = torch.FloatTensor([0.5])
    x = d(seq, l)
    print(x.shape)

    latent = torch.FloatTensor(1, 128).normal_(0, 1)
    g = Generator(128)
    x, l = g(latent)
    print(x.shape)
