from modules2 import BatShitSequenceGenerator, Cluster, Expander, GlobalContext, Reducer, SequenceGenerator, get_best_matches, init_weights
import torch
from torch import nn
from modules import PositionalEncoding, RecursiveNet, ResidualStack


class RecursiveExpander(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.depth_embedding = nn.Parameter(torch.FloatTensor(20, channels).normal_(0, 1))

        self.net = nn.Sequential(
            nn.Linear(channels * 2, channels),
            ResidualStack(channels, 3, bias=False),
            nn.Linear(channels, channels * 2, bias=False)
        )

    def forward(self, x, size):
        x = x.view(-1, self.channels)
        i = 0
        while x.shape[0] < size:
            x = self.net(torch.cat([x, self.depth_embedding[i]], dim=1))
            x = x.view(-1, self.channels)
            i += 1
        return x[:size]


class Discriminator(nn.Module):
    def __init__(self, channels, embedding_weights):
        super().__init__()

        self.atom_embedding = nn.Embedding(512 * 6, 8)

        with torch.no_grad():
            self.atom_embedding.weight.data = torch.from_numpy(
                embedding_weights)

        self.atom_embedding.requires_grad = False

        self.reduce = nn.Linear(8 + 2, 32)
        self.channels = channels

        self.net = nn.Sequential(
            GlobalContext(32),
            nn.Linear(32, 128),
            Cluster(channels, n_clusters=1),
            ResidualStack(channels, 3),
            nn.Linear(channels, channels)
        )

        self.atom_j = nn.Sequential(
            nn.Linear(10, channels),
            ResidualStack(channels, 3),
            nn.Linear(channels, 1)
        )

        self.cardinality = nn.Linear(1, 128)

        self.final = nn.Linear(256, 1)

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
        
        # judge each atom individually
        aj = self.atom_j(x).mean()

        c = self.cardinality(l.view(1, 1))

        x = self.reduce(x)
        x = self.net(x)

        x = torch.cat([c, x], dim=1)
        x = self.final(x)

        return torch.cat([x.view(-1), aj.view(-1)])


class BatShitSequenceGenerator(nn.Module):
    def __init__(self, channels, length):
        super().__init__()
        self.channels = channels
        self.length = length
        self.pos = PositionalEncoding(1, 16384, 8, 128)

        self.expander = RecursiveExpander(128)

        self.rs = nn.Linear(128 * 2, 128)

        self.net = nn.Sequential(
            ResidualStack(channels, 4),
            nn.Linear(channels, channels)
        )

        self.positions = nn.Linear(128, 769)

    def forward(self, x, length):
        # TODO: Predict the part of the domain to use embeddings from!

        # positions = torch.linspace(0, 0.999, 769).to(x.device)[:length]
        positions = torch.clamp(
            torch.abs(self.positions(x)), 0, 0.9999).view(-1)[:length]

        # exp = self.expander(x, length)

        x = x.view(1, self.channels)
        x = x.repeat(length, 1)
        p, p2 = self.pos(positions)
        x = torch.cat([x, p2], dim=1)
        x = self.rs(x)
        x = self.net(x)
        return x


class VariableExpander(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.is_member = nn.Linear(channels, 1)

        self.seq_gen = BatShitSequenceGenerator(channels, 35)
        # self.seq_gen = SequenceGenerator(128)

        # self.rnn_layers = 3
        # self.rnn = nn.GRU(
        #     channels,
        #     channels,
        #     self.rnn_layers,
        #     batch_first=False)

        self.cardinality = nn.Linear(128, 1)
        self.max_atoms = 768

    def forward(self, x):
        x = x.view(1, self.channels)
        l = torch.clamp(self.cardinality(x), 0, 1)
        c = int((l * self.max_atoms).item()) + 1

        x = self.seq_gen.forward(x, c)
        return x, l

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

            # member = self.is_member(x).view(1).item()
            # if member < 0.5:
            #     break

        seq = torch.cat(seq, dim=0)
        return seq, l


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.net = ResidualStack(channels, 4)

        self.variable = VariableExpander(channels)

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
        x = self.net(x)
        encodings, l = self.variable(x)

        atoms = self.to_atom(encodings)
        pos = self.to_pos(encodings)
        mags = self.to_magnitude(encodings)

        recon = torch.cat([atoms, pos, mags], dim=-1)
        return recon, l
