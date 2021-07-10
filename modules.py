import torch
from torch.nn import Module, Embedding, Linear, Sequential, GRU
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt


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

        pe = torch.from_numpy(pos_encode(
            self.domain, self.n_samples, self.n_freqs)).float().permute(1, 0)
        self.register_buffer('pos_encode', pe)

        self.l1 = Linear(1 + n_freqs * 2, self.out_channels)

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
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.l1 = Linear(channels, channels)
        self.l2 = Linear(channels, channels)

    def forward(self, x):
        shortcut = x
        x = self.l1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.l2(x)
        x = F.leaky_relu(shortcut + x, 0.2)
        return x


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

        self.atom_embedding = Embedding(512 * 6, 128)
        # self.magnitude_embedding = Embedding(256, 128)
        # self.positional_encoding = PositionalEncoding(
        #     self.domain, n_samples, n_freqs, 128)

        self.magnitude_embedding = Linear(1, 128)
        self.positional_encoding = Linear(1, 128)

        self.reduce = Linear(128 * 3, 128)

        self.net = Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.final = Linear(128, 128)

        self.reduce_global = Linear(128 * 2, 128)
        self.with_global = Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )
        self.global_final = Linear(128, 128)

    def get_magnitude_keys(self, embeddings):
        """
        Return discretized magnitudes
        """
        return get_best_matches(self.magnitude_embedding.weight, embeddings)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        return get_best_matches(self.atom_embedding.weight, embeddings)

    def get_positions(self, embeddings):
        """
        Return continuous positions in range [0 - 1]
        """
        return self.positional_encoding.get_positions(embeddings)

    def get_embeddings(self, x):
        atom, time, mag = x
        ae = self.atom_embedding(atom)
        # pe, te = self.positional_encoding(time)
        # me = self.magnitude_embedding(mag)
        return torch.cat([ae, time.view(-1, 1), mag.view(-1, 1)], dim=-1)

    def forward(self, x):
        atom, time, mag = x

        # embed individual points independently
        ae = self.atom_embedding(atom)
        te = self.positional_encoding(time.view(-1, 1))
        me = self.magnitude_embedding(mag.view(-1, 1))
        x = torch.cat([ae, te, me], dim=-1)
        x = self.reduce(x)
        x = self.net(x)
        x = self.final(x)

        # embed points with added global context
        cardinality = x.shape[0]
        individual = x
        x = torch.sum(x, dim=0, keepdim=True)
        x = torch.cat([individual, x.repeat(cardinality, 1)], dim=-1)
        x = self.reduce_global(x)
        x = self.with_global(x)
        x = self.global_final(x)

        # summarize into a single vector
        x = torch.sum(x, dim=0, keepdim=True)
        return ae, time, mag, x


class Decoder(Module):

    def __init__(self, n_layers=4):
        super().__init__()
        self.n_layers = n_layers
        # https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU
        self.rnn = GRU(128, 128, num_layers=n_layers, batch_first=False)

        self.to_atom = Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            Linear(128, 128)
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

    def forward(self, x, max_steps):
        # input in shape (sequence_length, batch_size, input_dim)
        # hidden in shape (num_rnn_layers, batch, hidden_dim)

        inp = torch.zeros((1, 1, 128)).float().to(x.device)
        hid = torch.zeros((self.n_layers, 1, 128)).to(x.device)
        inp[:] = x.view(1, 1, 128)

        encodings = []

        for _ in range(max_steps):
            inp, hid = self.rnn.forward(inp, hid)
            e = inp.view(1, 128)
            encodings.append(e)
            c = F.relu(self.is_constituent(e))
            zl = torch.zeros_like(c)
            if torch.all(c == zl):
                break

        encodings = torch.cat(encodings, dim=0)

        atoms = self.to_atom(encodings)
        pos = torch.clamp(self.to_pos(encodings), 0, 1)
        mags = self.to_magnitude(encodings)

        return [atoms, pos, mags]


def loss_func(a, b):
    """
    Align points/atoms with their best matches from the
    decoded signal and compute overall distance
    """
    l = max(a.shape[0], b.shape[0])
    a_diff = l - a.shape[0]
    b_diff = l - b.shape[0]

    a = torch.pad(a, ((0, 0), (0, a_diff)))
    b = torch.pad(b, ((0, 0), (0, b_diff)))

    dist = torch.cdist(a, b)
    indices = torch.argmin(dist, dim=0)
    return F.mse_loss(a[indices], b)


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

    def forward(self, x):
        # TODO: Having to explicitly provide the number
        # of steps here is a problem
        n_steps = x[0].shape[0]
        _, _, _, z = self.encoder(x)
        a, p, m = self.decoder(z, n_steps)
        return a, p, m, z


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
