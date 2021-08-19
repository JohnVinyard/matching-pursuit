from dtw import dtw_loss
import torch
from torch.nn import Module, Embedding, Linear, Sequential, GRU, BatchNorm1d, MaxPool1d, RNN, LayerNorm
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
from torch.nn.modules.conv import Conv1d


def activation(x):
    return F.leaky_relu(x, 0.2)


class Activation(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return activation(x)


def pos_encode(domain, n_samples, n_freqs):
    d = np.linspace(-domain, domain, n_samples)
    x = [d[None, :]]
    for i in range(n_freqs):
        x.extend([
            np.sin((2 ** i) * d)[None, :],
            np.cos((2 ** i) * d)[None, :]
        ])
    return np.concatenate(x, axis=0)


def get_best_matches(basis, recon):
    dist = torch.cdist(basis, recon)
    indices = torch.argmin(dist, dim=0)
    return indices


def init_weights(p):
    # with torch.no_grad():
    #     try:
    #         p.weight.uniform_(-0.02, 0.02)
    #     except AttributeError:
    #         pass

    #     try:
    #         p.bias.fill_(0)
    #     except AttributeError:
    #         pass
    pass


class RecursiveNet(Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = Conv1d(channels * 2, channels, 2, 1)
        self.pool = MaxPool1d(2, 2)
        self.depth = Embedding(32, 128)
        self.apply(init_weights)

    def forward(self, x):
        indices = torch.randperm(x.shape[0])
        x = x[indices]

        x = x.permute(1, 0).view(1, self.channels, -1)

        i = 0
        while x.shape[-1] > 1:
            e = self.depth(torch.LongTensor([i]).to(x.device)).view(
                1, 128, 1).repeat(1, 1, x.shape[-1])
            x = torch.cat([x, e], dim=1)
            x = self.net(x)
            x = activation(x)
            if x.shape[-1] == 1:
                break
            x = self.pool(x)
            i += 1

        x = x.view(self.channels, 1).permute(1, 0)
        return x


class ReducerNet(Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = GRU(channels, channels, 4)
        self.apply(init_weights)

    def forward(self, x):
        indices = torch.randperm(x.shape[0])
        x = x[indices]

        inp = x.view(-1, 1, self.channels)
        hid = torch.zeros(4, 1, self.channels).to(x.device)

        inp, hid = self.net(inp, hid)
        hid = hid.view(4, 1, self.channels)

        x = hid[-1, 0, :]

        return x


class PositionalEncoding(Module):
    """
    Take the appropriate slices from a positional encoding
    """

    def __init__(self, domain, n_samples, n_freqs, out_channels):
        super().__init__()
        self.domain = domain
        self.n_samples = n_samples
        self.n_freqs = n_freqs
        self.out_channels = out_channels

        self.freq_channels = 1 + n_freqs * 2

        pe = torch.from_numpy(pos_encode(
            self.domain, self.n_samples, self.n_freqs)).float().permute(1, 0)
        self.register_buffer('pos_encode', pe)

        self.l1 = Linear(self.freq_channels, self.out_channels, bias=False)

    def get_positions(self, embeddings):
        """
        Given embeddings, return continuous positions in the range [0 - 1]
        """
        return get_best_matches(self.pos_encode, embeddings) / float(self.n_samples)

    def forward(self, x):
        """
        x should be time encodings in the domain [0 - 1]
        """
        indices = (x * self.n_samples).long()
        orig = x = self.pos_encode[indices]
        x = self.l1(x)
        return orig, x


class ResidualBlock(Module):
    def __init__(self, channels, bias=True):
        super().__init__()
        self.channels = channels
        self.l1 = Linear(channels, channels, bias)
        self.l2 = Linear(channels, channels, bias)
        self.apply(init_weights)

    def forward(self, x):
        shortcut = x
        x = self.l1(x)
        x = activation(x)
        x = self.l2(x)
        x = activation(shortcut + x)
        return x


class ResidualStack(Module):
    def __init__(self, channels, layers, bias=True):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.net = Sequential(
            *[ResidualBlock(channels, bias) for _ in range(layers)]
        )

    def forward(self, x):
        return self.net(x)


def abs_max(x):
    return torch.sum(x, dim=0, keepdim=True)

    # dim = x.shape[1]
    # mx, indices = torch.max(torch.abs(x), dim=0)
    # x = x[indices, torch.arange(0, dim)]
    # return x


class Encoder(Module):
    """
    The encoder will take a (band, atom, time, magnitude) and produce
    a vector embedding

    - Band and atom will be combined into a single (6 x 512) number: 3,072
    - time will be a continuous value between [0 - 1]
    - magnitude will be encoded discretely [0 - 256]
    """

    def __init__(self, domain, n_samples=32768, n_freqs=16):
        super().__init__()
        self.domain = domain

        self.rnn = RNN(128, 128, 4, batch_first=False, nonlinearity='relu')

        self.atom_embedding = Embedding(512 * 6, 8)
        self.reduce = Linear(8 + 2, 128)

        self.independent = ResidualStack(128, 3)
        self.ind = Linear(128, 128)

        self.reduce2 = Linear(128 * 2, 128)
        self.context = ResidualStack(128, 3)
        self.ctxt = Linear(128, 128)

        self.attn = Linear(128, 128)
        self.attn2 = Linear(128, 128)

        self.encode = Linear(128, 128)

        self.apply(init_weights)

    # def get_magnitude_keys(self, embeddings):
    #     """
    #     Return discretized magnitudes
    #     """
    #     return get_best_matches(self.magnitude_embedding.weight, embeddings)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        print(embeddings.shape, self.atom_embedding.weight.shape)
        return get_best_matches(self.atom_embedding.weight, embeddings)

    # def get_positions(self, embeddings):
    #     """
    #     Return continuous positions in range [0 - 1]
    #     """
    #     return self.positional_encoding.get_positions(embeddings)

    def get_embeddings(self, x):
        atom, time, mag = x
        ae = self.atom_embedding(atom).view(-1, 8)
        pe = time.view(-1, 1)
        me = mag.view(-1, 1)

        # pe, te = self.positional_encoding(time)
        # me = self.magnitude_embedding(mag)

        return torch.cat([ae, pe, me], dim=-1)

    def forward(self, x, return_embeddings=False):

        x = self.get_embeddings(x)

        # input in shape (sequence_length, batch_size, input_dim)
        # hidden in shape (num_rnn_layers, batch, hidden_dim)

        x = self.reduce(x)

        inp = x.view(-1, 1, 128)
        hid = torch.zeros(4, 1, 128).to(inp.device)

        inp, hid = self.rnn(inp, hid)

        x = hid[-1, :, :].view(1, 128)
        x = self.encode(x)
        return x

        # n_points = x.shape[0]

        # x = self.reduce(x)

        # # embed independently
        # x = self.independent(x)
        # x = self.ind(x)

        # # x, _ = torch.max(torch.abs(x), dim=0, keepdim=True)
        # # embedded = x

        # attn = torch.sigmoid(self.attn(x))
        # # .sum(dim=0, keepdim=True) / n_points
        # embedded = abs_max(x * attn).view(1, 128)

        # x = torch.cat([x, embedded.repeat(x.shape[0], 1)], dim=1)

        # # embed with context
        # x = self.reduce2(x)
        # x = self.context(x)
        # x = self.ctxt(x)

        # attn = torch.sigmoid(self.attn2(x))
        # # .sum(dim=0, keepdim=True) / n_points
        # x = abs_max(x * attn).view(1, 128)

        # return x


class Decoder(Module):

    def __init__(self, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        self.rnn = RNN(
            128,
            128,
            num_layers=n_layers,
            batch_first=False,
            nonlinearity='relu')

        self.to_atom = Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            Linear(128, 8)
        )

        self.to_pos = Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            Linear(128, 1)
        )

        self.to_magnitude = Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            Linear(128, 1)
        )

        self.is_constituent = Linear(128, 1)

        self.apply(init_weights)

    def forward(self, x, max_steps):
        # input in shape (sequence_length, batch_size, input_dim)
        # hidden in shape (num_rnn_layers, batch, hidden_dim)

        inp = torch.zeros((1, 1, 128)).float().to(x.device)
        hid = torch.zeros((self.n_layers, 1, 128)).to(x.device)
        hid[0, 0, :] = x

        encodings = []

        for _ in range(max_steps):
            inp, hid = self.rnn.forward(inp, hid)
            e = inp.view(1, 128)
            encodings.append(e)
            # c = F.relu(self.is_constituent(e))
            # zl = torch.zeros_like(c)
            # if torch.all(c == zl):
            #     break

        encodings = torch.cat(encodings, dim=0)

        atoms = self.to_atom(encodings)
        pos = self.to_pos(encodings)
        mags = self.to_magnitude(encodings)

        recon = torch.cat([atoms, pos, mags], dim=-1)
        return recon





class AutoEncoder(Module):
    def __init__(
            self,
            domain=1,
            n_samples=32768,
            n_embedding_freqs=16,
            n_layers=4):

        super().__init__()
        self.encoder = Encoder(domain, n_samples, n_embedding_freqs)
        self.decoder = Decoder(n_layers)

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
        # TODO: Having to explicitly provide the number
        # of steps here is a problem
        n_steps = x[0].shape[0]
        z = self.encoder(x)
        recon = self.decoder(z, n_steps)
        return recon, z


if __name__ == '__main__':
    pe = pos_encode(1, 32768, 16)
    print(pe.shape)
    plt.matshow(pe[:, :1024])
    plt.show()

    # encoder = Encoder(domain=1)
    # decoder = Decoder()

    # # atom, time, mag
    # atoms = torch.from_numpy(np.random.randint(0, 6 * 512, 256)).long()
    # times = torch.from_numpy(np.random.uniform(0, 1, 256)).float()
    # mags = torch.from_numpy(np.random.randint(0, 256, 256)).long()

    # x = [atoms, times, mags]
    # ae, te, me, encoded = encoder.forward(x)
    # decoded = decoder.forward(encoded, n_steps=256)
    # ad, pd, md = decoded

    # enc = torch.cat([ae, te, me], dim=1).view(1, 256, -1)
    # print(enc.shape)
    # dec = torch.cat([ad, pd, md], dim=1).view(1, 256, -1)
    # print(dec.shape)

    # # the norm of this distance matrix is the loss
    # mat = torch.cdist(enc, dec).squeeze()
    # plt.matshow(mat.data.cpu().numpy())
    # plt.show()

    # to decode, look up the best-matching embeddings for
    # time, magnitude and position
