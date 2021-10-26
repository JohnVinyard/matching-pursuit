import math
from torch.nn.modules.activation import LeakyReLU
from torch.nn.modules.container import Sequential
from torch.nn.modules.normalization import LayerNorm
from modules2 import Cluster, DilatedBlock, Expander, init_weights, PositionalEncoding
from modules3 import Attention, LinearOutputStack, ToThreeD, ToTwoD
from torch import device, nn
import torch
import numpy as np
from torch.nn.modules.linear import Linear
from modules import ResidualBlock, ResidualStack, get_best_matches, pos_encode, pos_encode_feature
from torch.nn import functional as F

from relationship import Relationship


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


class Conv(nn.Module):
    def __init__(self, channels, kernel=7):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, 1, kernel // 2)
        self.kernel = kernel
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0,2, 1)
        # Pixel Norm
        x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
        return x


class Mixer(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.net = nn.Sequential(
            Conv(channels), 
            *[
                nn.Sequential(
                    
                    ResidualBlock(channels),
                )
                for _ in range(layers)
        ])
    
    def forward(self, x):
        for layer in self.net:
            x = layer(x)
            # Pixel Norm
            x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)
            x = x.permute(0, 2, 1)
        return x


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

        t = x[..., 17:].contiguous()
        a = unit_norm(x[..., :17].contiguous())

        total = self.all(x, x)
        time = self.time(t, t)

        atom = self.atom(a, a)

        # similarity with other samples in the batch
        # to promote sample diversity.
        indices1 = np.random.permutation(batch)
        indices2 = np.roll(indices1, 1)
        b = self.batch(x[indices1], x[indices2])

        # TODO: Consider attention layers that create
        # query and value based on a subset of features
        return torch.cat([
            total.view(-1),
            time.view(-1),
            atom.view(-1),
            b.view(-1)
        ])


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
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        orig = x

        if self.in_channels:
            x = self.alter(x)

        results = []
        for attn in self.attn:
            z = attn(x)
            results.append(z)

        x = torch.cat(results, dim=-1)
        x = self.fc(x)
        x = x + orig

        # x = x.permute(0, 2, 1)
        # x = self.norm(x)
        # x = x.permute(0, 2, 1)

        # x = unit_norm(x) * 6.2

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
            intermediate_layers=3,
            discriminator=False):

        super().__init__()
        self.channels = channels
        self.attention_heads = attention_heads
        self.attention_layers = attention_layers
        self.intermediate_layers = intermediate_layers
        self.discriminator = discriminator

        if self.discriminator:
            self.disc = nn.Sequential(*[
                LinearOutputStack(channels, 3, out_channels=1) 
                for _ in range(attention_layers)
            ])
        

        self.net = nn.Sequential(*[
            nn.Sequential(
                MultiHeadAttention(
                    channels,
                    attention_heads),
                LinearOutputStack(channels, intermediate_layers, activation=activation))
            for i in range(attention_layers)
        ])

    def forward(self, x):
        if not self.discriminator:
            return self.net(x)
        
        d = []

        for i, layer in enumerate(self.net):
            x = layer(x)
            d.append(torch.sigmoid(self.disc[i](x)).view(-1))
        
        return x, torch.cat(d)


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


class BatchDisc(nn.Module):
    def __init__(self, input_channels, channels):
        super().__init__()
        self.net = LinearOutputStack(
            channels, 3, in_channels=input_channels, out_channels=1)
        self.input_channels = input_channels

    def forward(self, x):
        batch, atoms, channels = x.shape
        x = x.permute(1, 0, 2).reshape(-1, self.input_channels)
        x = torch.sigmoid(self.net(x))
        return x.mean()


class Discriminator(nn.Module):
    def __init__(self, channels, dense_judgments, embedding_size, one_hot, noise_level, embeddings):
        super().__init__()
        self.channels = channels
        self.dense_judgements = dense_judgments
        self.embedding_size = embedding_size
        self.one_hot = one_hot
        self.noise_level = noise_level

        self.self_similarity = SelfSimilarity2(128)


        # input_size = self.embedding_size + 1

        self.atom_embedding = nn.Embedding(
            512 * 6,
            embedding_size)
        # self.atom_embedding.weight.requires_grad = False

        in_channels = embedding_size + 2 + 17
        self.bd = BatchDisc(in_channels, channels)
        self.rel = Relationship(in_channels, channels)
        self.abs = LinearOutputStack(channels, 3, in_channels=in_channels)

        self.comb = LinearOutputStack(channels, 3, in_channels=channels * 2)

        self.stack = nn.Sequential(
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 2),
            DilatedBlock(channels, 4),
            DilatedBlock(channels, 8),
            DilatedBlock(channels, 16),
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 2),
            DilatedBlock(channels, 4),
            DilatedBlock(channels, 8),
            DilatedBlock(channels, 16),
            DilatedBlock(channels, 1),
        )

        self.mixer = Mixer(channels, 12)

        self.dense = nn.Sequential(
            LinearOutputStack(
                channels,
                2,
                activation=activation,
                in_channels=channels),

            AttentionStack(
                channels,
                attention_heads=8,
                attention_layers=12,
                intermediate_layers=2)
        )

        # self.atoms = LinearOutputStack(
        #     channels, 3, in_channels=input_size)
        self.contextual = LinearOutputStack(
            channels, 3, in_channels=channels * 2, out_channels=1)

        self.dense_judge = LinearOutputStack(channels, 4, out_channels=1)


        self.atom_judge = LinearOutputStack(in_channels, 4, out_channels=1)

        self.init_atoms = set()
        self.final = LinearOutputStack(channels, 3, out_channels=1)

        self.apply(init_weights)

        with torch.no_grad():
            # self.atom_embedding.weight.fill_(0)
            self.atom_embedding.weight[:] = torch.from_numpy(embeddings)

    def get_times(self, embeddings):
        return embeddings.view(-1)

    def get_mags(self, embeddings):
        # return torch.norm(embeddings, dim=-1)
        return embeddings.view(-1)

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
                    self.embedding_size).uniform_(-1, 1).to(atom.device)

    def get_embeddings(self, x):
        atom, time, mag = x
        # self._init_atoms(atom)
        ae = self \
                .atom_embedding.weight[atom.view(-1)] \
                .view(-1, self.embedding_size)

        # STOP: Do not remove!! Adding noise is crucial to
        # keep the discriminator from simply distinguising
        # based on exact atom matches
        ae = unit_norm(ae) + torch.zeros_like(ae).uniform_(-0.1, 0.1)

        pe = time.view(-1, 1)
        m = mag.view(-1, 1)

        return torch.cat([ae, pe, m], dim=-1)

    def forward(self, x):
        batch, time, channels = x.shape

        t = pos_encode_feature(x[..., -2:-1], 1, 2**15, 8)
        x = torch.cat([x, t], dim=-1)
        # j = torch.sigmoid(self.atom_judge(x))

        # b = self.bd(x)

        

        # look at atom relationships
        r = self.rel(x)
        # look at absolute atom positions
        a = self.abs(x)

        # combine and reduce
        x = torch.cat([r, a], dim=-1)
        x = self.comb(x)

        
        x = x.permute(0, 2, 1)
        x = self.stack(x)
        x = x.permute(0, 2, 1)

        # multiple attention layers
        # x = self.dense(x)

        # x = self.mixer(x)
        # x = self.dense_judge(x)
        # x = torch.sigmoid(x)

        # global pooling
        x = x.mean(dim=1)
        x = self.final(x)
        x = torch.sigmoid(x)

        return torch.cat([
            x.view(-1), 
            # j.view(-1), 
            # b.view(-1)
        ])
        

        return torch.cat([x.view(-1), b.view(-1)])


        

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
        self.stack = LinearOutputStack(channels, 2, bias=False)
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

        # x = unit_norm(x) * 8

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
        self.embedding_size = embeddings.weight.shape[1]

        self.combine = LinearOutputStack(
            channels, 3, in_channels=channels + embeddings.weight.shape[1] + 2, bias=False)

        self.atoms = LinearOutputStack(
            channels, 3, out_channels=17, bias=False)
        # self.band = LinearOutputStack(
        #     channels, 3, out_channels=6, bias=False)
        self.pos = LinearOutputStack(
            channels, 3, out_channels=1, bias=False)
        self.mag = LinearOutputStack(
            channels, 3, out_channels=1, bias=False)
        

    def _decompose(self, x):
        a = x[..., :17]
        p = x[..., -2:-1]
        m = x[..., -1:]
        return a, p, m

    def _recompose(self, a, p, m):
        return torch.cat([a, p, m], dim=-1)

    def forward(self, encodings, mother_atoms):

        encodings = torch.cat(
            [encodings, mother_atoms.repeat_interleave(2, 1)], dim=-1)
        encodings = self.combine(encodings)

        # a = F.relu(a) @ self.embeddings[0].weight
        p = self.pos(encodings)

        ma, mp, mm = self._decompose(mother_atoms.repeat_interleave(2, 1))

        a = self.atoms(encodings)
        # b = torch.softmax(self.band(encodings), dim=-1)

        mag = self.mag(encodings)

        # positions are relative
        a = self._recompose(unit_norm(a + ma), p + mp, mag + mm)

        return a


class Transform(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = LinearOutputStack(
            channels, 3, in_channels=128 + 19 + channels, bias=False)

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

        self.pos_encoder = PositionalEncoding(1, 128, 32)

        self.conv_expander = ConvExpander(channels, self.max_atoms)

        self.set_expansion = SetExpansion(channels, self.max_atoms)

        self.net = AttentionStack(
            channels,
            attention_heads=8,
            attention_layers=12,
            intermediate_layers=2)
        

        self.transform_pos = nn.Conv1d(channels + 65, channels, 1, 1, 0)
        self.stack = nn.Sequential(
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 2),
            DilatedBlock(channels, 4),
            DilatedBlock(channels, 8),
            DilatedBlock(channels, 16),
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 2),
            DilatedBlock(channels, 4),
            DilatedBlock(channels, 8),
            DilatedBlock(channels, 16),
            DilatedBlock(channels, 1),
        )

        self.mixer = Mixer(channels, 12)

        # out_channels = 3072 if (
        #     use_disc_embeddings or one_hot) else self.embedding_size

        self.atoms = LinearOutputStack(
            channels, 3, out_channels=17)
        # self.band = LinearOutputStack(
        #     channels, 3, out_channels=6)
        self.pos = LinearOutputStack(
            channels, 3, out_channels=1)
        self.mag = LinearOutputStack(
            channels, 3, out_channels=1)

        # self.model = PleaseDearGod(channels, max_atoms, embeddings)

        # self.to_atom = ToAtom(channels, embeddings)

        self.apply(init_weights)

    def forward(self, x, batch):

        just_shuffle = False

        if just_shuffle:
            b, time, channels = batch.shape

            batch = batch.clone().detach()
            mags = batch[:, :, -1].reshape((b * time))
            indices = np.random.permutation(mags.shape[0])
            mags = mags[indices].reshape((b, time))


            # shuffle atoms in the batch
            atoms = batch[..., :-2].reshape((b * time, -1))
            indices = np.random.permutation(atoms.shape[0])
            atoms = atoms[indices].reshape((b, time, -1))

            # mess with atoms
            # batch[..., :-2] = atoms

            # mess with positions
            batch[:, :, -2] = torch.from_numpy(np.random.uniform(-1, 1, (batch.shape[0], batch.shape[1]))).to(x.device).float()

            # mess with magnitudes
            # batch[:, :, -1] = mags
            return batch
        
        else:
            batch = x.shape[0]


            x = x.view(-1, 128, 1).repeat(1, 1, 128)
            p = self.pos_encoder.pos_encode.view(1, 128, 65).permute(0, 2, 1).repeat(batch, 1, 1)

            x = torch.cat([x, p], dim=1)
            x = self.transform_pos(x)
            x = self.stack(x)
            encodings = x.permute(0, 2, 1)
            # encodings = self.mixer(encodings)
            # encodings = self.net(encodings)

            # progressive generation
            # z = x
            # a = torch.zeros(batch, 1, 19).to(z.device)
            # local_latent = torch.zeros(batch, 1, self.channels).to(z.device)
            # atoms = []
            # for layer in self.please_dear_god:
            #     a, local_latent = layer(z, a, local_latent)
            #     atoms.append(a)
            # recon = x = torch.cat(atoms, dim=1)

            # encodings = self.conv_expander(x)

            # a = unit_norm(F.relu(self.atoms(encodings)))
            # b = torch.softmax(self.band(encodings), dim=-1)
            # a = torch.cat([a, b], dim=-1)

            

            # transform encodings
            a = unit_norm(self.atoms(encodings))
            p = torch.tanh(self.pos(encodings))
            m = torch.sigmoid(self.mag(encodings)) * 20
            recon = torch.cat([a, p, m], dim=-1)


            # shuffle so order can't be used by the discriminator
            output = torch.zeros_like(recon)
            for i in range(batch):
                output[i] = recon[i, torch.randperm(x.shape[1]), :]
            return output


if __name__ == '__main__':
    ew = np.random.normal(0, 1, (3072, 8))

    gen = Generator(128)
    latent = torch.FloatTensor(8, 128).normal_(0, 1)
    generated = gen.forward(latent)
    print(generated.shape)

    disc = Discriminator(128, ew)
    j = disc.forward(generated)
    print(j.shape)
