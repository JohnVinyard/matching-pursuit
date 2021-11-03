import math
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


class MixerBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.a = ResidualBlock(channels)
        self.b = ResidualBlock(channels)
    
    def forward(self, x):
        orig = x
        x = self.a(x)
        x = x.permute(0, 2, 1)
        x = self.b(x)
        x = x.permute(0, 2, 1)
        x = x + orig
        return x

class Mixer(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.net = nn.Sequential(
            # Conv(channels), 
            *[
                nn.Sequential(
                    
                    # ResidualBlock(channels),
                    MixerBlock(channels)
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
        self.norm = nn.LayerNorm(channels)

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

        # Pixel Norm
        # x = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-8)

        # x = unit_norm(x)

        x = self.norm(x)

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



class BatchDisc(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = LinearOutputStack(channels, 5, in_channels=channels * 2, out_channels=1)
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.channels).permute(1, 0, 2)
        # (time, batch, channels)
        x = x.reshape(-1, batch_size // 2, 2 * self.channels)
        x = self.net(x)
        x = torch.sigmoid(x).mean()
        return x


class Discriminator(nn.Module):
    def __init__(self, channels, dense_judgments, embedding_size, one_hot, noise_level, embeddings):
        super().__init__()
        self.channels = channels
        self.dense_judgements = dense_judgments
        self.embedding_size = embedding_size
        self.one_hot = one_hot
        self.noise_level = noise_level

        self.bd = BatchDisc(channels)

        # input_size = self.embedding_size + 1

        self.atom_embedding = nn.Embedding(
            512 * 6,
            embedding_size)
        self.atom_embedding.weight.requires_grad = False

        in_channels = embedding_size + 2 + 17
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

        self.reducer = nn.Sequential(
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 4, 4, 0),
            nn.LeakyReLU(0.2)
        )

        self.mixer = Mixer(channels, 6)

        self.dense = nn.Sequential(
            LinearOutputStack(
                channels,
                2,
                activation=activation,
                in_channels=embedding_size + 2),

            AttentionStack(
                channels,
                attention_heads=8,
                attention_layers=8,
                intermediate_layers=2)
        )

        self.dense_judge = LinearOutputStack(channels, 3, out_channels=1)
        self.final = LinearOutputStack(channels, 3, out_channels=1)

        self.apply(init_weights)

        # with torch.no_grad():
        #     self.atom_embedding.weight[:] = torch.from_numpy(embeddings)

    def get_times(self, embeddings):
        return embeddings.view(-1)

    def get_mags(self, embeddings):
        return embeddings.view(-1)

    def get_atom_keys(self, embeddings):
        """
        Return atom indices
        """
        nw = self.atom_embedding.weight
        return get_best_matches(unit_norm(nw), unit_norm(embeddings))

    
    def get_embeddings(self, x):
        atom, time, mag = x
        ae = unit_norm(self \
                .atom_embedding.weight[atom.view(-1)] \
                .view(-1, self.embedding_size))

        # STOP: Do not remove!! Adding noise is crucial to
        # keep the discriminator from simply distinguising
        # based on exact atom matches
        ae = unit_norm(ae) + torch.zeros_like(ae).uniform_(-0.1, 0.1)

        pe = time.view(-1, 1)
        m = mag.view(-1, 1)

        return torch.cat([ae, pe, m], dim=-1)

    def forward(self, x):
        batch, time, channels = x.shape

        # this should encourage the generator to produce
        # diverse outputs
        # b = self.bd(x)

        # t = pos_encode_feature(x[..., -2:-1], 1, 2**15, 8)
        # x = torch.cat([x, t], dim=-1)

        # # look at atom relationships
        # r = self.rel(x)
        # # look at absolute atom positions
        # a = self.abs(x)

        # # combine and reduce
        # x = torch.cat([r, a], dim=-1)
        # x = self.comb(x)
        
        # x = x.permute(0, 2, 1)
        # x = self.stack(x)
        # # x = self.reducer(x)
        # x = x.permute(0, 2, 1)

        # x = self.mixer(x)
        x = self.dense(x)

        x = x.mean(dim=1)

        x = self.dense_judge(x)
        x = torch.sigmoid(x)
        # return torch.cat([b.view(-1), x.view(-1)])
        return x



# class ResidualUpscale(nn.Module):
#     def __init__(self, channels, heads=4):
#         super().__init__()

#         # self.summarize = nn.Linear(channels * 2, channels)

#         self.expander = Expander(channels, factor=2)
#         self.stack = LinearOutputStack(channels, 2, bias=False)
#         self.attn = MultiHeadAttention(channels, heads)

#     def forward(self, x):

#         batch, time, channels = x.shape

#         # noise = torch.zeros(batch, 1, channels)\
#         #     .normal_(0, 1).repeat(1, time, 1).to(x.device)

#         # x = torch.cat([x, noise], dim=-1)
#         # x = self.summarize(x)

#         exp = self.expander(x)
#         x = self.stack(exp)
#         x = x + exp

#         # x = unit_norm(x) * 8

#         x = self.attn(x)

        
#         return x


# class ConvExpander(nn.Module):
#     def __init__(self, channels, max_atoms):
#         super().__init__()
#         self.channels = channels
#         self.max_atoms = max_atoms
#         # self.ln = nn.Linear(channels, channels * 8)
#         self.ln = Expander(channels, factor=8)
#         n_layers = int(np.log2(128) - np.log2(8))
#         self.net = nn.Sequential(
#             # ToThreeD(),
#             nn.Sequential(*[
#                 ResidualUpscale(channels, heads=i + 1)
#                 for i in range(n_layers)]),
#             # ToTwoD()
#         )

#     def forward(self, x):
#         x = x.view(-1, 1, self.channels)
#         x = self.ln(x)
#         x = self.net(x)
#         return x[:, :self.max_atoms, :]


# class ToAtom(nn.Module):
#     def __init__(self, channels, embeddings):
#         super().__init__()
#         self.channels = channels
#         self.embeddings = [embeddings]
#         self.embedding_size = embeddings.weight.shape[1]

#         self.combine = LinearOutputStack(
#             channels, 3, in_channels=channels + embeddings.weight.shape[1] + 2, bias=False)

#         self.atoms = LinearOutputStack(
#             channels, 3, out_channels=17, bias=False)
#         # self.band = LinearOutputStack(
#         #     channels, 3, out_channels=6, bias=False)
#         self.pos = LinearOutputStack(
#             channels, 3, out_channels=1, bias=False)
#         self.mag = LinearOutputStack(
#             channels, 3, out_channels=1, bias=False)
        

#     def _decompose(self, x):
#         a = x[..., :17]
#         p = x[..., -2:-1]
#         m = x[..., -1:]
#         return a, p, m

#     def _recompose(self, a, p, m):
#         return torch.cat([a, p, m], dim=-1)

#     def forward(self, encodings, mother_atoms):

#         encodings = torch.cat(
#             [encodings, mother_atoms.repeat_interleave(2, 1)], dim=-1)
#         encodings = self.combine(encodings)

#         # a = F.relu(a) @ self.embeddings[0].weight
#         p = self.pos(encodings)

#         ma, mp, mm = self._decompose(mother_atoms.repeat_interleave(2, 1))

#         a = self.atoms(encodings)
#         # b = torch.softmax(self.band(encodings), dim=-1)

#         mag = self.mag(encodings)

#         # positions are relative
#         a = self._recompose(unit_norm(a + ma), p + mp, mag + mm)

#         return a


# class Transform(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.net = LinearOutputStack(
#             channels, 3, in_channels=128 + 19 + channels, bias=False)

#     def forward(self, latent, atom, local_latent):
#         batch, time, channels = atom.shape
#         factor = time // latent.shape[1]
#         latent = latent.repeat(1, factor, 1)
#         x = torch.cat([atom, latent, local_latent], dim=-1)
#         x = self.net(x)
#         return x


# class PleaseDearGod(nn.Module):
#     def __init__(self, channels, max_atoms, embeddings, heads):
#         super().__init__()
#         self.channels = channels
#         self.transformer = Transform(channels)
#         self.max_atoms = max_atoms
#         self.to_atom = ToAtom(channels, embeddings)
#         self.expander = Expander(channels, 2)
#         self.attn = MultiHeadAttention(channels, heads)
#         self.heads = heads

#     def forward(self, z, a, local_latent):
#         z = z.view(-1, 1, self.channels)
#         batch = z.shape[0]

#         x = self.transformer(z, a, local_latent)

#         x = self.attn(x)

#         x = self.expander(x)
#         # TODO: Try positional encoding
#         a = self.to_atom(x, a)

#         return a, x


class ToAtom(nn.Module):
    def __init__(self, channels, embedding_size):
        super().__init__()
        self.channels = channels
        self.embedding_size = embedding_size

        self.a = LinearOutputStack(channels, 3, out_channels=self.embedding_size)
        self.p = LinearOutputStack(channels, 3, out_channels=1)
        self.m = LinearOutputStack(channels, 3, out_channels=1)
    
    def forward(self, latent, mother_atoms):
        ma = mother_atoms[..., :self.embedding_size]
        mp = mother_atoms[..., -2:-1]
        mm = mother_atoms[..., -1:]

        da = self.a(latent)
        dp = self.p(latent)
        dm = self.m(latent)

        a = unit_norm(ma + da)
        p = torch.tanh(mp + dp)
        m = F.relu(mm + dm)
        
        new_atoms = torch.cat([a, p, m], dim=-1)
        return latent, new_atoms

class Up(nn.Module):
    def __init__(self, channels, factor=2):
        super().__init__()
        self.channels = channels
        self.expand = Expander(channels, factor=factor)
        self.transform = LinearOutputStack(channels, 3)
        self.factor = factor
    
    def forward(self, latent, mother_atoms):
        latent = self.expand(latent)
        latent = self.transform(latent)
        atoms = mother_atoms.repeat_interleave(self.factor, dim=1)
        return latent, atoms


class DeltaGenerator(nn.Module):
    def __init__(self, channels, embedding_size):
        super().__init__()
        self.channels = channels
        self.transform = LinearOutputStack(channels, 3)
        self.first = ToAtom(channels, embedding_size)
        self.net = nn.Sequential(*[
            nn.Sequential(
                Up(channels),
                ToAtom(channels, embedding_size)
            )
            for _ in range(6)])
        self.embedding_size = embedding_size
    
    def forward(self, x):
        x = x.view(-1, 1, self.channels)
        output_atoms = []

        mother_atoms = torch.zeros(x.shape[0], 1, self.embedding_size + 2).to(x.device)
        latent = x

        latent = self.transform(latent)
        latent, mother_atoms = self.first(latent, mother_atoms)

        output_atoms.append(mother_atoms)

        for step in self.net:
            for layer in step:
                latent, mother_atoms = layer(latent, mother_atoms)
            output_atoms.append(mother_atoms)
        
        x = torch.cat(output_atoms, dim=1)
        return x

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

        # self.please_dear_god = nn.Sequential(*[
        #     PleaseDearGod(channels, max_atoms, embeddings, i + 1) for i in range(6)
        # ])

        self.delta = DeltaGenerator(channels, embedding_size)

        self.pos_encoder = PositionalEncoding(1, 128, 64)

        # self.conv_expander = ConvExpander(channels, self.max_atoms)


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

        # self.mixer = Mixer(channels, 6)

        self.mlp = LinearOutputStack(channels, 8)

        self.atoms = LinearOutputStack(
            channels, 3, out_channels=self.embedding_size)
        self.pos = LinearOutputStack(
            channels, 3, out_channels=1)
        self.mag = LinearOutputStack(
            channels, 3, out_channels=1)

        self.register_buffer('noise', torch.FloatTensor(1, 128, 128).normal_(0, 10))


        self.apply(init_weights)

    def forward(self, x, batch):

        x = self.delta(x)
        
        a = x[..., :self.embedding_size]

        # quantize
        real_atoms = unit_norm(self.embeddings[0].weight)
        dist = torch.cdist(a, real_atoms).reshape(-1, real_atoms.shape[0])
        indices = torch.argmin(dist, dim=1)
        diff = a.reshape(-1, self.embedding_size) - real_atoms.reshape(-1, self.embedding_size)[indices]
        
        return x, diff
        
        batch = x.shape[0]

        x = x.view(-1, 128, 1)

        # x = x + self.noise

        # combine repeated latent with set-positional encoding
        # x = x.view(-1, 1, 128).repeat(1, 128, 1)
        # p = self.pos_encoder.pos_encode.view(1, 128, 129)

        # x = x + p[:, :, :128]

        x = x * self.noise

        # x = torch.cat([x, p], dim=1)
        # x = self.transform_pos(x)

        x = x.permute(0, 2, 1)
        x = self.stack(x)
        encodings = x.permute(0, 2, 1)

        # encodings = self.mixer(x)
        # encodings = self.net(x)
        # encodings = self.mlp(x)
        
        # transform encodings
        a = unit_norm(self.atoms(encodings))
        p = torch.tanh(self.pos(encodings))
        m = torch.sigmoid(self.mag(encodings)) * 20

        # quantize
        real_atoms = unit_norm(self.embeddings[0].weight)
        dist = torch.cdist(a, real_atoms).reshape(-1, real_atoms.shape[0])
        indices = torch.argmin(dist, dim=1)
        diff = a.reshape(-1, self.embedding_size) - real_atoms.reshape(-1, self.embedding_size)[indices]
        # a = a - diff.reshape(*a.shape)

        recon = torch.cat([a, p, m], dim=-1)
        

        # shuffle so order can't be used by the discriminator
        # output = torch.zeros_like(recon)
        # for i in range(batch):
        #     output[i] = recon[i, torch.randperm(x.shape[1]), :]
        return recon, diff


if __name__ == '__main__':
    ew = np.random.normal(0, 1, (3072, 8))

    gen = Generator(128)
    latent = torch.FloatTensor(8, 128).normal_(0, 1)
    generated = gen.forward(latent)
    print(generated.shape)

    disc = Discriminator(128, ew)
    j = disc.forward(generated)
    print(j.shape)
