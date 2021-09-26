from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import Sequential
from modules2 import Cluster, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack, ToThreeD, ToTwoD
from torch import device, nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualStack, get_best_matches, pos_encode, pos_encode_feature
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


class SelfSimilaritySummarizer(nn.Module):
    def __init__(self, max_atoms):
        super().__init__()
        self.max_atoms = max_atoms
        layers = int(np.log2(max_atoms)) - 2
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, (3, 3), (1, 1), (1, 1)),
            *[
                nn.Sequential(
                    nn.Conv2d(16, 16, (3, 3), (1, 1), (1, 1)),
                    nn.MaxPool2d((3, 3), (2, 2), (1, 1)),
                    nn.LeakyReLU(0.2),
                ) for _ in range(layers)
            ],
            nn.Conv2d(16, 1, (1, 1), (1, 1), (0, 0)))

    def forward(self, x, y):
        batch, time, channels = x.shape
        dist = torch.cdist(x, y)
        dist = dist.reshape(batch, 1, time, time)
        return self.net(dist)


class SelfSimilarity2(nn.Module):
    def __init__(self, max_atoms):
        super().__init__()

        self.all = SelfSimilaritySummarizer(max_atoms)
        self.time = SelfSimilaritySummarizer(max_atoms)
        self.atom = SelfSimilaritySummarizer(max_atoms)
        self.batch = SelfSimilaritySummarizer(max_atoms)

    def forward(self, x):
        batch, time, channels = x.shape

        # t = x[..., 17:].contiguous()
        # a = unit_norm(x[..., :17].contiguous())

        # total = self.all(x, x)
        # time = self.time(t, t)

        # atom = self.atom(a, a)

        # similarity with other samples in the batch
        # to promote sample diversity.
        indices1 = np.random.permutation(batch)
        indices2 = np.roll(indices1, 1)
        b = self.batch(x[indices1], x[indices2])

        # TODO: Consider attention layers that create
        # query and value based on a subset of features
        return torch.cat([
            # total.view(-1),
            # time.view(-1),
            # atom.view(-1),
            b.view(-1)
        ])


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
        dist = torch.cdist(x, x)  # (batch, max_atoms, max_atoms)

        upper = []
        indices = torch.triu_indices(time, time, offset=1)
        for i in range(batch):
            upper.append(dist[i, indices[0], indices[1]])

        upper = torch.stack(upper)  # batch * (time ** 2 / 2)
        upper = upper[:, None, :]

        x = self.net(upper)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, heads, in_channels=None):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.in_channels = in_channels

        if in_channels:
            self.alter = nn.Linear(in_channels, channels)

        self.attn = nn.Sequential(*[
            Attention(channels) for _ in range(self.heads)
        ])
        self.fc = nn.Linear(channels * heads, channels)

    def forward(self, x):
        orig = x

        if self.in_channels:
            x = self.ater(x)

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
                MultiHeadAttention(
                    channels,
                    attention_heads),
                LinearOutputStack(channels, intermediate_layers, activation=activation))
            for i in range(attention_layers)
        ])

    def forward(self, x):
        return self.net(x)


class DiscReducer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 1, 3),
                nn.MaxPool1d(7, 2, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 1, 3),
                nn.MaxPool1d(7, 2, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 1, 3),
                nn.MaxPool1d(7, 2, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 1, 3),
                nn.MaxPool1d(7, 2, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.MaxPool1d(3, 2, 1),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.MaxPool1d(3, 2, 1),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.MaxPool1d(3, 2, 1),
                nn.LeakyReLU(0.2),
            ),



            nn.Conv1d(channels, 1, 7, 1, 3)

        )

    def forward(self, x):
        batch, time, channels = x.shape
        x = x.permute(0, 2, 1)  # (batch, channels, time) for conv
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels, dense_judgments, embedding_size, one_hot, noise_level):
        super().__init__()
        self.channels = channels
        self.dense_judgements = dense_judgments
        self.embedding_size = embedding_size
        self.one_hot = one_hot
        self.noise_level = noise_level

        self.self_similarity = SelfSimilarity2(128)

        self.time_embedding = PositionalEncoding(1, 2 ** 15, 8)

        input_size = self.embedding_size + 1

        self.atom_embedding = nn.Embedding(
            512 * 6, embedding_size, scale_grad_by_freq=True)

        self.dense = nn.Sequential(
            LinearOutputStack(
                channels,
                2,
                activation=activation,
                in_channels=input_size),

            AttentionStack(
                channels,
                attention_heads=8,
                attention_layers=8,
                intermediate_layers=2)
        )

        self.atoms = LinearOutputStack(
            channels, 3, in_channels=input_size)
        self.contextual = LinearOutputStack(
            channels, 3, in_channels=channels * 2, out_channels=1)

        self.dist_judge = LinearOutputStack(
            channels, 3, in_channels=self.embedding_size, out_channels=1)

        self.dense_judge = LinearOutputStack(channels, 4, out_channels=1)

        self.reduce = nn.Sequential(
            AttentionClusters(channels, 16),
            LinearOutputStack(channels, 2),

            AttentionClusters(channels, 4),
            LinearOutputStack(channels, 2),

            AttentionClusters(channels, 1),
            LinearOutputStack(channels, 2, out_channels=1)
        )

        self.reducer = DiscReducer(channels)

        self.init_atoms = set()

        self.apply(init_weights)

        with torch.no_grad():
            self.atom_embedding.weight.fill_(0)

    def get_times(self, embeddings):
        return embeddings.view(-1)

    def get_mags(self, embeddings):
        return torch.norm(embeddings, dim=-1)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        nw = self.atom_embedding.weight
        return get_best_matches(unit_norm(nw), unit_norm(embeddings))

    def _init_atoms(self, atom):
        indices = set([int(a) for a in atom.view(-1)])
        to_init = indices - self.init_atoms
        self.init_atoms.update(to_init)
        for ti in to_init:
            with torch.no_grad():
                self.atom_embedding.weight[ti] = torch.FloatTensor(
                    17).uniform_(-1, 1).to(atom.device)

    def get_embeddings(self, x):
        atom, time, mag = x
        self._init_atoms(atom)
        ae = self \
            .atom_embedding.weight[atom.view(-1)] \
            .view(-1, 
            self.embedding_size)

        # STOP: Do not remove!! Adding noise is crucial to
        # keep the discriminator from simply distinguising
        # based on exact atom matches
        ae = unit_norm(ae) + torch.zeros_like(ae).uniform_(-0.1, 0.1)

        pe = time.view(-1, 1)

        return torch.cat([ae * mag[:, None], pe], dim=-1)

    def forward(self, x):
        batch, time, channels = x.shape

        # ss = self.self_similarity(x)

        # atoms = x[..., :self.embedding_size]
        # keys = self.get_atom_keys(atoms.view(-1, self.embedding_size))
        # embeddings = unit_norm(self.atom_embedding.weight[keys] \
        #     .reshape(batch, time, self.embedding_size))
        # diff = embeddings - unit_norm(atoms)
        # dj = self.dist_judge(diff)

        a = self.atoms(x)

        x = self.dense(x)
        c = self.contextual(torch.cat([a, x], dim=-1))

        j = self.dense_judge(x)

        if self.dense_judgements:
            return torch.cat([
                j.view(-1),
                # dj.view(-1),
                c.view(-1),
                # ss.view(-1)
            ])

        # j = j.mean(dim=1, keepdim=True)
        # x = self.reduce(x)
        # x = torch.cat([x, j], dim=-1)

        x = self.reducer(x)

        x = torch.cat([
            x.view(-1),
            # ss.mean().view(-1),
            c.mean().view(-1),
            # dj.mean().view(-1)
        ])
        return x


class SetExpansion(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.channels = channels
        self.max_atoms = max_atoms

        self.register_buffer('members', torch.FloatTensor(
            self.max_atoms, channels).normal_(0, 1))

        self.bias = LinearOutputStack(
            channels, 3, activation=activation, in_channels=channels * 2)
        self.weight = LinearOutputStack(
            channels, 3, activation=activation, in_channels=channels * 2)

    def forward(self, x):
        x = x.view(-1, 1, self.channels).repeat(1, self.max_atoms, 1)
        x = torch.cat([x, self.members[None, ...].repeat(16, 1, 1)], dim=-1)

        b = self.bias(x)
        w = self.weight(x)

        x = (self.members[None, ...] * w) + b

        # x = ((self.members[None, ...] * w[:, None, :]) + b[:, None, :])

        return x


class ResidualUpscale(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()

        # self.summarize = nn.Linear(channels * 2, channels)

        self.expander = Expander(channels, factor=2)
        self.stack = LinearOutputStack(channels, 2)
        self.attn = MultiHeadAttention(channels, heads)

    def forward(self, x):

        batch, time, channels = x.shape

        # noise = torch.zeros(batch, 1, channels)\
        #     .normal_(0, 1).repeat(1, time, 1).to(x.device)

        # x = torch.cat([x, noise], dim=-1)
        # x = self.summarize(x)

        exp = self.expander(x)
        x = self.stack(exp)
        x = x + exp

        # x = unit_norm(x) * 3.2

        x = self.attn(x)

        return x


class ConvExpander(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.channels = channels
        self.max_atoms = max_atoms
        # self.ln = nn.Linear(channels, channels * 8)
        self.ln = Expander(channels, factor=8)
        n_layers = int(np.log2(128) - np.log2(8))
        self.net = nn.Sequential(
            # ToThreeD(),
            nn.Sequential(*[
                ResidualUpscale(channels, heads=i + 1)
                for i in range(n_layers)]),
            # ToTwoD()
        )

    def forward(self, x):
        x = x.view(-1, 1, self.channels)
        x = self.ln(x)
        x = self.net(x)
        return x[:, :self.max_atoms, :]


class ToAtom(nn.Module):
    def __init__(self, channels, embeddings):
        super().__init__()
        self.channels = channels
        self.embeddings = [embeddings]

        self.atoms = LinearOutputStack(
            channels, 3, out_channels=3072)
        self.pos = LinearOutputStack(
            channels, 3, out_channels=1)
        
    
    def _decompose(self, x):
        a = x[..., :17]
        p = x[..., -1:]
        return a, p

    def _recompose(self, a, p):
        a = a
        p = torch.clamp(p, -1, 1)
        return torch.cat([a, p], dim=-1)
    
    def forward(self, encodings, mother_atoms):
        a = \
            F.relu(self.atoms(encodings)) @ self.embeddings[0].weight.clone()

        # a = self.atoms(encodings)
        p = self.pos(encodings)

        ma, mp = self._decompose(mother_atoms.repeat_interleave(2, 1))

        a = self._recompose(a, p + mp)

        return a


class Transform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = LinearOutputStack(
            channels, 3, in_channels=128 + 18 + channels)
    
    def forward(self, latent, atom, local_latent):
        batch, time, channels = atom.shape
        factor = time // latent.shape[1]
        latent = latent.repeat(1, factor, 1)
        x = torch.cat([atom, latent, local_latent], dim=-1)
        x = self.net(x)
        return x


class PleaseDearGod(nn.Module):
    def __init__(self, channels, max_atoms, embeddings, heads):
        super().__init__()
        self.channels = channels
        self.transformer = Transform(channels)
        self.max_atoms = max_atoms
        self.to_atom = ToAtom(channels, embeddings)
        self.expander = Expander(channels, 2)
        self.attn = MultiHeadAttention(channels, heads)
        self.heads = heads
    
    def forward(self, z, a, local_latent):
        z = z.view(-1, 1, self.channels)
        batch = z.shape[0]
        x = self.transformer(z, a, local_latent)
        x = self.attn(x)
        x = self.expander(x)
        # TODO: Try positional encoding
        a = self.to_atom(x, a)

        return a, x


class Generator(nn.Module):
    def __init__(
            self,
            channels,
            embeddings,
            use_disc_embeddings=True,
            embedding_size=8,
            one_hot=False,
            noise_level=0.1,
            max_atoms=128):

        super().__init__()
        self.channels = channels
        self.embeddings = [embeddings]
        self.embedding_size = embedding_size
        self.one_hot = one_hot
        self.noise_level = noise_level

        self.max_atoms = max_atoms
        self.use_disc_embeddings = use_disc_embeddings

        self.please_dear_god = nn.Sequential(*[
            PleaseDearGod(channels, max_atoms, embeddings, i + 1) for i in range(6)
        ])

        self.conv_expander = ConvExpander(channels, self.max_atoms)

        self.set_expansion = SetExpansion(channels, self.max_atoms)

        self.net = AttentionStack(
            channels,
            attention_heads=8,
            attention_layers=8,
            intermediate_layers=2)

        out_channels = 3072 if (
            use_disc_embeddings or one_hot) else self.embedding_size

        self.atoms = LinearOutputStack(
            channels, 3, out_channels=3072, shortcut=False)
        self.pos = LinearOutputStack(
            channels, 3, out_channels=1, shortcut=False)

        
        # self.model = PleaseDearGod(channels, max_atoms, embeddings)

        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]

        z = x
        a = torch.zeros(batch, 1, 18).to(z.device)
        local_latent = torch.zeros(batch, 1, self.channels).to(z.device)
        atoms = []

        for layer in self.please_dear_god:
            a, local_latent = layer(z, a, local_latent)
            atoms.append(a)
        
        x = torch.cat(atoms, dim=1)
        return x

        encodings = self.conv_expander(x)

        # Interestingly, this doesn't work when trying to use the
        # discriminator embeddings.  I wonder if it's too hard to produce
        # embeddings in the two different domains from the same feature

        # encodings = self.net(encodings)

        atoms = \
            F.relu(self.atoms(encodings)) @ self.embeddings[0].weight.clone()
        p = torch.clamp(self.pos(encodings), -1, 1)

        # m = sine_one(self.mag(encodings))
        # p = pos_encode_feature(torch.clamp(self.pos(encodings), -1, 1), 1, 8)
        # atoms = atoms * m

        recon = torch.cat([atoms, p], dim=-1)

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
