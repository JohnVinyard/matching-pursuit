from torch.nn.modules.container import Sequential
from modules2 import Cluster, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack, ToThreeD, ToTwoD
from torch import device, nn
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
        n = np.linalg.norm(x, axis=-1, keepdims=True)
    else:
        n = torch.norm(x, dim=-1, keepdim=True)
    return x / (n + 1e-12)


class SelfSimilarity(nn.Module):
    def __init__(self, max_atoms):
        super().__init__()
        self.max_atoms = max_atoms

        self.size = (max_atoms ** 2) // 2

        self.net = nn.Sequential(
            nn.Conv1d(1, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 8, 2, 2, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 1, 2, 2, 0),
        )
    
    def forward(self, x):
        batch, time, channels = x.shape
        dist = torch.cdist(x, x)

        upper = []
        indices = torch.triu_indices(time, time, offset=1)
        for i in range(batch):
            upper.append(dist[i, indices[0], indices[1]])
        
        upper = torch.stack(upper) # batch * (time ** 2 / 2)
        upper = upper[:, None, :]


        x = self.net(upper)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.attn = nn.Sequential(*[Attention(channels)
                                  for _ in range(self.heads)])
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
    def __init__(self, channels, dense_judgments, embedding_size, one_hot, noise_level):
        super().__init__()
        self.channels = channels
        self.dense_judgements = dense_judgments
        self.embedding_size = embedding_size
        self.one_hot = one_hot
        self.noise_level = noise_level

        self.time_encode = PositionalEncoding(1, 16384, 8)
        self.mag_encode = PositionalEncoding(1, 512, 8)

        # self.self_similarity = SelfSimilarity(100)

        self.atom_embedding = nn.Embedding(
            512 * 6, embedding_size, scale_grad_by_freq=True)

        self.time = nn.Linear(17, channels)
        self.mag = nn.Linear(17, channels)
        self.atom = nn.Linear(3072 if one_hot else embedding_size, channels)
        self.combine = LinearOutputStack(channels, 3, in_channels=channels * 3)

        self.dense = nn.Sequential(
            LinearOutputStack(
                channels,
                2,
                activation=activation),

            AttentionStack(
                channels,
                attention_heads=8,
                attention_layers=8,
                intermediate_layers=2)
        )

        self.dense_judge = LinearOutputStack(channels, 4, out_channels=1)

        self.reduce = nn.Sequential(
            AttentionClusters(channels, 16),
            LinearOutputStack(channels, 2),

            AttentionClusters(channels, 4),
            LinearOutputStack(channels, 2),

            AttentionClusters(channels, 1),
            LinearOutputStack(channels, 2, out_channels=1)
        )

        self.apply(init_weights)

        with torch.no_grad():
            self.atom_embedding.weight.uniform_(0, 1)
    
    def get_times(self, embeddings):
        return self.time_encode.get_positions(embeddings)
    
    def get_mags(self, embeddings):
        return self.mag_encode.get_positions(embeddings)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        if self.one_hot:
            return torch.argmax(embeddings, dim=1)
        else:
            nw = self.atom_embedding.weight
            return get_best_matches(nw, embeddings)

    def get_embeddings(self, x):
        atom, time, mag = x

        if self.one_hot:
            n_atoms = atom.shape[0]
            oh = torch.zeros(n_atoms, 3072).to(
                atom.device).uniform_(0, self.noise_level)
            # ones = torch.ones_like(oh)
            # ones = ones * torch.zeros_like(ones).normal_(1, 0.01)
            indices = torch.arange(n_atoms)
            oh[indices, atom] = 1
            ae = oh
        else:
            ae = self.atom_embedding(atom).view(-1, self.embedding_size)

            # add noise so the discriminator can't use *exact* embeddings
            # to judge real vs fake.
            ae = ae + torch.zeros_like(ae).normal_(0, 0.01).to(ae.device)

        # pe = time.view(-1, 1)
        # me = mag.view(-1, 1)
        pe = self.time_encode(time)
        me = self.mag_encode(mag)
        return torch.cat([ae, pe, me], dim=-1)

    def forward(self, x):
        batch, time, channels = x.shape

        # if isinstance(x, list):
        #     x = self.get_embeddings(x)

        # ss = self.self_similarity(x)

        cutoff = 3072 if self.one_hot else self.embedding_size
        a = self.atom(x[..., :cutoff])
        p = self.time(x[..., cutoff: cutoff + 17])
        m = self.mag(x[..., cutoff + 17:])
        x = torch.cat([a, p, m], dim=-1)
        x = self.combine(x)

        x = self.dense(x)
        j = self.dense_judge(x)

        if self.dense_judgements:
            return j

        j = j.mean(dim=1, keepdim=True)
        x = self.reduce(x)
        x = torch.cat([x, j], dim=-1)

        x = torch.cat([x.view(-1)])
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


class Generator(nn.Module):
    def __init__(
            self,
            channels,
            embeddings,
            use_disc_embeddings=True,
            embedding_size=8,
            one_hot=False,
            noise_level=0.1):

        super().__init__()
        self.channels = channels
        self.embeddings = [embeddings]
        self.embedding_size = embedding_size
        self.one_hot = one_hot
        self.noise_level = noise_level

        self.max_atoms = 100
        self.use_disc_embeddings = use_disc_embeddings

        self.set_expansion = SetExpansion(channels, self.max_atoms)
        self.conv_expander = ConvExpander(channels)

        self.net = AttentionStack(
            channels,
            attention_heads=8,
            attention_layers=8,
            intermediate_layers=2)

        out_channels = 3072 if (
            use_disc_embeddings or one_hot) else self.embedding_size
        
        self.atoms = LinearOutputStack(
            channels, 3, out_channels=out_channels)
        self.pos = LinearOutputStack(channels, 3, out_channels=17)
        self.mag = LinearOutputStack(channels, 3, out_channels=17)

        self.apply(init_weights)

    def forward(self, x):

        batch = x.shape[0]
        time = self.max_atoms

        # Expansion
        # encodings = self.conv_expander(x)

        # Set Expansion
        encodings = self.set_expansion(x)

        # RNN
        # encodings = self.rnn(x)

        # End attention stack
        encodings = self.net(encodings)

        if self.use_disc_embeddings:
            # softmax with disc embeddings
            e = self.embeddings[0].weight.clone()
            atoms = torch.softmax(self.atoms(encodings), dim=1)
            atoms = atoms @ e
        elif self.one_hot:
            atoms = self.atoms(encodings)
            noise = torch.zeros_like(atoms).uniform_(0, self.noise_level)
            atoms = atoms + noise
            atoms = F.relu(atoms)

            # atoms = torch.softmax(atoms, dim=-1)

            # atoms = torch.clamp(atoms.reshape(batch * time, -1), 0, 1)
            # indices = torch.argmax(atoms, dim=-1)
            # oh = torch.zeros_like(atoms)
            # first_dim = torch.arange(batch * time)
            # oh[first_dim, indices] = atoms[first_dim, indices]
            # oh = oh.reshape(batch, time, -1)
            # atoms = oh
        else:
            atoms = self.atoms(encodings)

        p = torch.sin(self.pos(encodings))
        m = torch.sin(self.mag(encodings))

        recon = torch.cat([atoms, p, m], dim=-1)

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
