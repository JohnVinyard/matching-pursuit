import torch
from torch import nn
from torch.nn import functional as F
from modules import \
    ResidualBlock, PositionalEncoding, init_weights, get_best_matches
import numpy as np
from collections import defaultdict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def shuffled(x):
    size = x.shape[0]
    indices = torch.randperm(size)
    return x[indices]


class Embedder(nn.Module):
    """
    Translate a set of three-tuples of:

    (atom_index, magnitude_index, cont. position)

    Into a set of dense embeddings
    """

    def __init__(self, domain, n_samples=32768, n_freqs=16):
        super().__init__()
        self.domain = domain
        self.channels = 128
        self.pos_encoding_channels = n_freqs * 2 + 1
        # self.pos_encoding_channels = 16

        self.atom_embedding = nn.Embedding(512 * 6, 8)
        # self.magnitude_embedding = nn.Embedding(256, self.channels)
        # self.positional_encoding = PositionalEncoding(
        #     self.domain, n_samples, n_freqs, self.channels
        # )

        self.reduce = nn.Linear(
            8 + 2, self.channels)

        self.apply(init_weights)

    def raw_embedding(self, x):
        atom, time, mag = x
        ae = self.atom_embedding(atom).view(-1, 8)
        # pe, te = self.positional_encoding(time.view(-1, 1))
        # me = self.magnitude_embedding(mag.view(-1, 1)).view(-1, self.channels)
        pe = time.view(-1, 1)
        me = mag.view(-1, 1)

        x = torch.cat([ae, pe, me], dim=-1)
        return x

    def forward(self, x):
        x = self.raw_embedding(x)
        x = self.reduce(x)
        return x


class BinaryClassifier(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.net = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(layers)])
        self.final = nn.Linear(self.channels, 1)

    def forward(self, x):
        x = self.net(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        return x


class ResidualStack(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.net = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(layers)]
        )
    
    def forward(self, x):
        return self.net(x)


class LocalGlobal(nn.Module):
    def __init__(self, channels, layers, aggregate):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.aggregate = aggregate

        self.local = nn.Sequential(
            ResidualStack(channels, layers),
            nn.Linear(channels, channels)
        )

        self.glob = nn.Sequential(
            nn.Linear(channels * 2, channels),
            ResidualStack(channels, layers),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        n = x.shape[0]
        loc = x = self.local(x)
        context = self.aggregate(x)
        context = context.repeat(n, 1)
        x = torch.cat([loc, context], dim=-1)
        x = self.glob(x)
        x = self.aggregate(x)
        return x



class Leaf(nn.Module):
    def __init__(
            self,
            channels,
            layers,
            domain,
            n_samples,
            n_freqs=16):

        super().__init__()
        self.channels = channels
        self.layers = layers
        self.embedder = Embedder(domain, n_samples, n_freqs)

        self.locglob = LocalGlobal(
            channels, 
            layers, 
            aggregate=lambda x: torch.max(torch.abs(x), dim=0, keepdim=True)[0])

        # self.encoder = nn.RNN(channels, channels, layers,
        #                       bias=False, dropout=0.2, nonlinearity='relu')
        self.decoder = nn.GRU(channels, channels, layers)
        self.membership_classifier = BinaryClassifier(channels, layers)

        self.to_latent = nn.Linear(self.channels, self.channels)

        self.to_atom = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            nn.Linear(channels, 8)
        )

        self.to_pos = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            nn.Linear(channels, 1)
        )

        self.to_magnitude = nn.Sequential(
            ResidualBlock(channels),
            ResidualBlock(channels),
            nn.Linear(channels, 1)
        )

        self.max_iters = 50

    def encode(self, x):
        x = self.embedder(x)

        # encode the set of latent embeddings into a single
        # latent vector
        # x = shuffled(x)
        
        # x = x.view(-1, 1, self.channels)  # seq_length, batch, dim
        # hid = torch.zeros((self.layers, 1, self.channels)).to(x.device)
        # inp, hid = self.encoder(x, hid)
        # latent = inp[-1, :, :].view(1, self.channels)

        latent = self.locglob(x)

        latent = self.to_latent(latent)

        return latent

    def decode(self, latent, count):
        inp = torch.zeros((1, 1, self.channels)).to(latent.device)
        hid = torch.zeros((self.layers, 1, self.channels)).to(latent.device)
        hid[0, :, :] = latent

        embeddings = []
        for i in range(count):
            inp, hid = self.decoder(inp, hid)
            embedding = inp.view(1, self.channels)

            # determine if the next prediction is a member of the set
            # is_member = self.membership_classifier(embedding)
            # if i > 1 and (is_member.mean() < 0.5).item():
            #     print(i)
            #     break

            embeddings.append(embedding)

        embedding_set = torch.cat(embeddings, dim=0)

        atoms = self.to_atom(embedding_set)
        pos = self.to_pos(embedding_set)
        mags = self.to_magnitude(embedding_set)
        return torch.cat([atoms, pos, mags], dim=-1)


class Branch(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers

        self.reduce = nn.Linear(channels * 2, channels)
        self.encoder = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(layers)])
        self.to_latent = nn.Linear(channels, channels)

        self.decoder = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(layers)])
        self.expand = nn.Linear(channels, channels * 2)

    def encode(self, x):
        x = x.view(-1, self.channels * 2)
        x = self.reduce(x)
        x = self.encoder(x)
        x = self.to_latent(x)
        return x

    def decode(self, x):
        x = self.decoder(x)
        x = self.expand(x)
        x = x.view(-1, self.channels)
        return x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            domain=1,
            total_samples=32768,
            blocks=8,
            n_embedding_freqs=16,
            n_layers=4,
            channels=128):

        super().__init__()

        if domain != 1:
            raise ValueError(
                'Not currently handling domains other than [0 -1]')

        self.domain = domain
        self.total_samples = total_samples
        self.n_embedding_freqs = n_embedding_freqs
        self.n_layers = n_layers
        self.channels = channels

        # number of sub-segments
        self.blocks = blocks

        # number of samples per sub-segment
        self.block_size = self.total_samples // blocks

        self.leaf = Leaf(
            channels,
            n_layers,
            domain,
            self.block_size,
            n_embedding_freqs)

        n_levels = int(np.log2(self.blocks))

        self.net = nn.Sequential(
            *[Branch(channels, n_layers) for _ in range(n_levels)])

        self.apply(init_weights)

    # def get_magnitude_keys(self, embeddings):
    #     """
    #     Return discretized magnitudes
    #     """
    #     return get_best_matches(self.leaf.embedder.magnitude_embedding.weight, embeddings)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        return get_best_matches(self.leaf.embedder.atom_embedding.weight, embeddings)

    # def get_positions(self, embeddings):
    #     """
    #     Return continuous positions in range [0 - 1]
    #     """
    #     return self.positional_encoding.get_positions(embeddings)

    def edges(self):
        return np.linspace(0, self.domain, self.blocks, endpoint=False)

    def bucket(self, edges, t):
        return np.searchsorted(edges, t, side='right') - 1

    def get_embeddings(self, x):
        """
        Get a dictionary mapping sub-segment index to atom, pos (local), mag
        embeddings in that sub-segment

        sub-segment losses should be computed independently and added
        """
        embeddings = dict()

        time_slices = self._segment(x)
        for k, v in time_slices.items():
            if v is None:
                embeddings[k] = torch.zeros(
                    1, 8 + 2).to(device)
            else:
                embeddings[k] = self.leaf.embedder.raw_embedding(v)

        return embeddings

    def decode_pos(self, x):
        pos = []

        edges = self.edges()

        for k, v in x.items():
            a, p, m = v
            # local = self.leaf.embedder.positional_encoding.get_positions(p)
            local = p
            # scale
            local /= self.blocks
            # translate
            local += edges[k]
            pos.extend(local)

        return pos

    def decode_mag(self, x):
        mag = []
        for k, v in x.items():
            a, p, m = v
            # mag.extend(get_best_matches(
            #     self.leaf.embedder.magnitude_embedding.weight, m))
            mag.extend(m)
        return mag

    def decode_atom(self, x):
        atom = []
        for k, v in x.items():
            a, p, m = v
            atom.extend(get_best_matches(
                self.leaf.embedder.atom_embedding.weight, a))
        return atom

    def _segment(self, x):
        a, p, m = x

        time_slices = {i: [] for i in range(self.blocks)}
        edges = self.edges()

        for atom, time, mag in zip(a, p, m):
            b = self.bucket(edges, time)
            # translate and scale the time_slice so time values are still
            # in the domain [0 - 1]
            time = (time - edges[b]) * self.blocks
            time_slices[b].append((atom, time, mag))

        fused = dict()

        for k, v in time_slices.items():
            try:
                a, p, m = zip(*v)
                a = torch.LongTensor(a).to(device)
                p = torch.FloatTensor(p).to(device)
                m = torch.FloatTensor(m).to(device)
                fused[k] = (a, p, m)
            except ValueError:
                fused[k] = None

        return fused

    def flatten(self, segment_dict):

        split = {k: (v[:, :8], v[:, 8:9], v[:, 9:])
                 for k, v in segment_dict.items()}

        atom_chunks = list(x[0] for x in split.values())
        pos_chunks = []
        mag_chunks = list(x[2] for x in split.values())

        edges = self.edges()

        for k, v in split.items():
            a, p, m = v
            # local = self.leaf.embedder.positional_encoding.get_positions(p)
            local = p
            local = local / self.blocks
            local = local + edges[k]
            pos_chunks.append(local)

        a = torch.cat(atom_chunks, dim=0)
        p = torch.cat(pos_chunks, dim=0)
        m = torch.cat(mag_chunks, dim=0)

        return a, p, m

    def forward(self, x):
        """
        TODO:
            1. group tuples by time
            2. forward pass of leaves and then branches for each group
            3. backward pass of branches and then leaves for each group
            4. restore original times
        """
        # time_slices = defaultdict(list)
        # edges = self.edges()

        # for atom, time, mag in x:
        #     b = self.bucket(edges, time)

        #     # translate and scale the time_slice so time values are still
        #     # in the domain
        #     time = (time - edges[b]) * self.blocks
        #     time_slices[b].append((atom, time - edges[b], mag))

        time_slices = self._segment(x)

        counts = dict()

        latents = []
        for k, v in time_slices.items():
            if v is None:
                latents.append(torch.zeros(1, self.channels).to(device))
                counts[k] = 1
            else:
                counts[k] = len(v[0])
                latent = self.leaf.encode(v)
                latents.append(latent.view(1, self.channels))

        x = latents = torch.cat(latents, dim=0)

        # for layer in self.net:
        #     x = layer.encode(x)

        # at this point, we've encoded the entire segment into
        # a single latent vector
        z = x

        # for layer in self.net[::-1]:
        #     x = layer.decode(x)

        # we now have encodings for each leaf node
        x = x.view(self.blocks, self.channels)

        decoded = dict()
        for k, latent in zip(time_slices.keys(), x):
            d = self.leaf.decode(latent, counts[k])
            decoded[k] = d

        # we now have a dictionary grouping atom, pos and mag embeddings
        # for each time_slice bucket
        #
        # TODO: This is fine for computing loss, as long as loss is computed
        # independently for each bucket and then summed, but in order to
        # decode it, we must
        #   1. translate the time embedding back into a local position
        #   2. translate the local position into a global one

        return decoded, z
