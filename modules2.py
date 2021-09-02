from torch import nn
import torch
from torch.nn import functional as F
from modules import PositionalEncoding, ResidualStack, get_best_matches
from os import environ

environ['CUDA_LAUNCH_BLOCKING'] = '1'


def init_weights(p):
    try:
        if not p.requires_grad:
            print('NO INIT FOR', p)
            return
    except AttributeError:
        pass

    with torch.no_grad():
        try:
            p.weight.uniform_(-0.1, 0.1)
        except AttributeError:
            pass

        try:
            p.bias.uniform_(-0.0001, 0.0001)
        except AttributeError:
            pass


class Dilated(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.net = nn.Conv1d(
            channels,
            channels,
            kernel_size=2,
            stride=1,
            dilation=dilation)

    def forward(self, x):
        orig = x
        x = F.pad(x, (self.dilation, 0))
        x = self.net(x)
        x = F.leaky_relu(x + orig, 0.2)
        return x


class SequenceGenerator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            Dilated(channels, dilation=1),
            Dilated(channels, dilation=2),
            Dilated(channels, dilation=4),
            Dilated(channels, dilation=8),
        )

        self.translate = nn.Sequential(
            ResidualStack(channels, 2),
            nn.Linear(channels, channels)
        )

    def forward(self, x, n_steps):
        context = x.view(1, self.channels, 1)

        x = torch.zeros(1, self.channels, 16).to(x.device)

        seq = []
        for _ in range(n_steps):
            z = self.net(x)
            z = z[..., -1:]
            z = self.translate((context + z).view(1, self.channels))
            seq.append(z)
            x = torch.cat([x[..., 1:], z[..., None]], dim=-1)

        seq = torch.cat(seq, dim=0)
        return seq


class BatShitSequenceGenerator(nn.Module):
    def __init__(self, channels, length):
        super().__init__()
        self.channels = channels
        self.length = length
        self.pos = PositionalEncoding(1, length, 8, 128)

        self.rs = nn.Linear(128 * 2, 128)

        self.net = nn.Sequential(
            ResidualStack(channels, 4),
            nn.Linear(channels, channels)
        )

    def forward(self, x, length):
        # TODO: Predict the part of the domain to use embeddings from!
        # TODO: Predict the cluster size

        x = x.view(1, self.channels)
        x = x.repeat(length, 1)
        p, p2 = self.pos(torch.linspace(0, 0.9999, self.length).to(x.device))
        x = torch.cat([x, p2], dim=1)
        x = self.rs(x)
        x = self.net(x)
        return x


class GlobalContext(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.contr = nn.Linear(channels, 1)
        self.reduce = nn.Linear(channels, channels)

    def forward(self, x):
        n_elements = x.shape[0]
        x = x.view(-1, self.channels)

        # TODO: possibly zero pad so this is really concatenation
        # and not addition
        # outer concatenate
        x = x[None, :] + x[:, None]

        x = x.view(n_elements, n_elements, self.channels)

        x = self.reduce(x)

        z = torch.sigmoid(self.contr(x))
        x = x * z

        # TODO: only aggregate over upper diagonal
        x = torch.mean(x, dim=1)
        return x


class Cluster(nn.Module):
    def __init__(
            self,
            channels,
            n_clusters,
            aggregate=lambda x: torch.max(x, dim=0)[0]):

        super().__init__()
        self.channels = channels
        self.n_clusters = n_clusters

        self.clusters = nn.Sequential(
            *[nn.Linear(channels, 1) for _ in range(n_clusters)])

        # self.assign = nn.Linear(channels, n_clusters)
        self.aggregate = aggregate

    def forward(self, x):
        orig = x

        # x = self.assign(x)
        # z = F.softmax(x, dim=-1)
        # mx, indices = torch.max(z, dim=1)

        output = torch.zeros(self.n_clusters, self.channels).to(x.device)

        for i in range(self.n_clusters):
            with_factor = self.clusters[i](orig)
            output[i] = self.aggregate(orig * with_factor)

        return output


class Reducer(nn.Module):
    def __init__(self, channels, factor):
        super().__init__()
        self.channels = channels
        self.factor = factor
        self.reduce = nn.Linear(factor * self.channels, channels)

    def forward(self, x):
        x = x.view(-1, self.channels * self.factor)
        x = self.reduce(x)
        return x


class Expander(nn.Module):
    def __init__(self, channels, factor):
        super().__init__()
        self.channels = channels
        self.factor = factor
        self.expand = nn.Linear(
            self.channels, self.channels * factor, bias=False)

    def forward(self, x):
        x = x.view(-1, self.channels)
        x = self.expand(x)
        x = x\
            .view(-1, self.channels, self.factor)\
            .permute(0, 2, 1).reshape(-1, self.channels)
        # x = x.view(-1, self.channels)
        return x


class VariableExpander(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.is_member = nn.Linear(channels, 1)
        # self.seq_gen = SequenceGenerator(channels)
        self.seq_gen = BatShitSequenceGenerator(channels, 35)

        # self.rnn_layers = 3
        # self.rnn = nn.RNN(
        #     channels,
        #     channels,
        #     self.rnn_layers,
        #     batch_first=False,
        #     nonlinearity='relu')

    def forward(self, x):
        x = x.view(1, self.channels)
        x = self.seq_gen.forward(x, 35)
        return x

        # input in shape (sequence_length, batch_size, input_dim)
        # hidden in shape (num_rnn_layers, batch, hidden_dim)
        inp = torch.zeros(1, 1, self.channels).to(x.device)
        hid = torch.zeros(self.rnn_layers, 1, self.channels).to(x.device)
        hid[0, :, :] = x

        seq = []
        for i in range(35):
            inp, hid = self.rnn.forward(inp, hid)
            x = inp.view(1, self.channels)
            seq.append(x)

            # member = self.is_member(x).view(1).item()
            # if member < 0.5:
            #     break

        seq = torch.cat(seq, dim=0)
        return seq


class Encoder(nn.Module):
    def __init__(self, channels, embedding_weights):
        super().__init__()

        self.atom_embedding = nn.Embedding(512 * 6, 8)

        with torch.no_grad():
            self.atom_embedding.weight.data = torch.from_numpy(
                embedding_weights)

        self.atom_embedding.requires_grad = False

        self.reduce = nn.Linear(8 + 2, 128)
        self.channels = channels

        self.net = nn.Sequential(
            GlobalContext(channels),
            Cluster(channels, n_clusters=16),  # 16
            ResidualStack(channels, 1),
            Reducer(channels, factor=4),  # 4
            ResidualStack(channels, 1),
            Reducer(channels, factor=4),  # 1
            ResidualStack(channels, 1),
            nn.Linear(channels, channels)
        )

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        nw = self.atom_embedding.weight
        # nw = torch.norm(self.atom_embedding.weight, dim=-1, keepdim=True)
        # nw = self.atom_embedding.weight / (nw + 1e-12)
        return get_best_matches(nw, embeddings)

    def get_embeddings(self, x):
        atom, time, mag = x
        ae = self.atom_embedding(atom).view(-1, 8)

        # norms = torch.norm(ae, dim=-1, keepdim=True)
        # ae = ae / (norms + 1e-12)

        pe = time.view(-1, 1)
        me = mag.view(-1, 1)

        return torch.cat([ae, pe, me], dim=-1)

    def forward(self, x):
        x = self.get_embeddings(x)
        x = self.reduce(x)
        x = self.net(x)
        # norms = torch.norm(x, dim=-1, keepdim=True)
        # x = x / (norms + 1e-12)
        return x


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.net = nn.Sequential(
            Expander(channels, factor=4),  # 4
            ResidualStack(channels, layers=1),
            Expander(channels, factor=4),  # 16
            ResidualStack(channels, layers=1),
        )

        self.to_atom = nn.Sequential(
            ResidualStack(channels, layers=1),
            nn.Linear(128, 8)
        )

        self.to_pos = nn.Sequential(
            ResidualStack(channels, layers=1),
            nn.Linear(128, 1)
        )

        self.to_magnitude = nn.Sequential(
            ResidualStack(channels, layers=1),
            nn.Linear(128, 1)
        )

        self.variable = VariableExpander(channels)

    def forward(self, x):
        x = self.net(x)
        output = []
        for i in range(x.shape[0]):
            seq = self.variable(x[i])
            output.append(seq)

        encodings = torch.cat(output, dim=0).view(-1, self.channels)

        atoms = torch.sin(self.to_atom(encodings))
        pos = (torch.sin(self.to_pos(encodings)) + 1) * 0.5
        mags = (torch.sin(self.to_magnitude(encodings)) + 1) * 0.5

        recon = torch.cat([atoms, pos, mags], dim=-1)
        return recon


class AutoEncoder(nn.Module):
    def __init__(self, channels, embedding_weights):
        super().__init__()
        self.channels = channels
        self.encoder = Encoder(channels, embedding_weights)
        self.decoder = Decoder(channels)

        self.apply(init_weights)

    def get_embeddings(self, x):
        with torch.no_grad():
            return self.encoder.get_embeddings(x)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        keys = self.encoder.get_atom_keys(embeddings)
        return keys

    def forward(self, x):

        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


if __name__ == '__main__':
    embedding = torch.FloatTensor(1, 128)
    net = SequenceGenerator(128)

    x = net.forward(embedding, 14)
    print(x.shape)
