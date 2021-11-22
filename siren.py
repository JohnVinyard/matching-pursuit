from typing import ForwardRef
from scipy.sparse.construct import random
import torch
from torch.optim.adam import Adam
import zounds
from torch import nn

from datastore import batch_stream
from modules import pos_encode_feature
from modules3 import LinearOutputStack
from torch.nn import functional as F
import numpy as np
from scipy.fftpack import dct, idct

sr = zounds.SR22050()
batch_size = 2
n_samples = 2**15
n_sub_samples = 4096
do_cumsum = False
final_activation = False
n_embedding_freqs = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# translate max frequencies in hz to radians
omegas = sr.nyquist * 2 * np.pi
omegas = omegas / int(sr)

path = '/hdd/musicnet/train_data'


torch.backends.cudnn.benchmark = True


def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.02, 0.02)
        except AttributeError:
            pass


def init_disc_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.1, 0.1)
        except AttributeError:
            pass


def activation(x):
    return torch.sin(x)
    # return F.leaky_relu(x, 0.2)

class Activation(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return activation(x)


def transform(x):
    # return x
    return dct(x, axis=-1, norm='ortho')


def inverse_transform(x):
    # return x
    return idct(x, axis=-1, norm='ortho')


# class Encoder(nn.Module):
#     def __init__(self, channels):
#         super().__init__()

#         self.embed_pos = nn.Conv1d(33, 8, 1, 1, 0, bias=False)
#         self.embed_sample = nn.Conv1d(1, 8, 7, 1, 3, bias=False)

#         self.factor = nn.Parameter(torch.FloatTensor(1).fill_(10))

#         self.net = nn.Sequential(
#             nn.Conv1d(16, 16, 8, 8, 0, bias=False),
#             nn.Conv1d(16, 32, 8, 8, 0, bias=False),
#             nn.Conv1d(32, 64, 8, 8, 0, bias=False),
#             nn.Conv1d(64, 128, 8, 8, 0, bias=False),
#             nn.Conv1d(128, 128, 8, 8, 0, bias=False),
#         )

#         self.apply(init_weights)

#     def forward(self, pos, sample):

#         pos = pos.permute(0, 2, 1)
#         sample = sample.permute(0, 2, 1)

#         pe = self.embed_pos(pos)
#         se = self.embed_sample(sample)

#         x = torch.cat([pe, se], dim=1)
#         x = activation(x * self.factor)

#         for layer in self.net:
#             x = layer(x)
#             x = activation(x * self.factor)

#         x = x.view(-1, 128)
#         return x


class Disc(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.pos_embed = LinearOutputStack(
            channels, 3, in_channels=n_embedding_freqs * 2 + 1, activation=activation)
        self.sample_embed = LinearOutputStack(channels, 3, in_channels=1, activation=activation)
        self.comb = LinearOutputStack(channels, 3, in_channels=channels * 2, activation=activation)

        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, channels, 7, 2, 3),
            Activation(),
            nn.Conv1d(channels, 1, 1, 1, 0)
        )
        self.apply(init_disc_weights)

    def forward(self, pos, sample):
        p = self.pos_embed(pos)
        s = self.sample_embed(sample)

        x = torch.cat([p, s], dim=-1)
        x = self.comb(x)

        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels, apply_activation=True):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.factor = nn.Parameter(torch.FloatTensor(out_channels).fill_(10))
        self.apply_activation = apply_activation
        self.out_channels = out_channels

    def forward(self, x, f, b):
        x = self.linear(x)
        if self.apply_activation:
            # f = activation(f)
            v = (self.factor * x) + b.view(batch_size, 1, -1)
            x = activation(v)
        return x


class Network(nn.Module):
    def __init__(self, layers, in_channels, hidden_channels):
        super().__init__()
        self.layers = layers
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.pos_bias = LinearOutputStack(
            hidden_channels, 3, out_channels=33, activation=activation)

        self.mod_factor = nn.Sequential(
            *[LinearOutputStack(hidden_channels, 3, activation=activation) for layer in range(layers + 1)])
        self.mod_bias = nn.Sequential(
            *[LinearOutputStack(hidden_channels, 3, activation=activation) for layer in range(layers + 1)])

        self.net = nn.Sequential(
            Layer(in_channels, hidden_channels),
            *[Layer(hidden_channels, hidden_channels)
              for layer in range(layers)],
            Layer(hidden_channels, 1, apply_activation=final_activation)
        )

        self.apply(init_weights)

    def forward(self, x, latent):
        pb = self.pos_bias(latent)
        x = x + pb

        for i, layer in enumerate(self.net):
            try:
                mod_bias = self.mod_bias[i]
                mod_factor = self.mod_factor[i]
                factor = mod_factor(latent)
                bias = mod_bias(latent)
                x = layer(x, factor, bias)
            except IndexError:
                factor = torch.ones(batch_size, 1).to(x.device)
                bias = torch.zeros(batch_size, 1).to(x.device)
                x = layer(x, factor, bias)

        # x = x * omegas
        # x = torch.sigmoid(x) * np.pi
        if do_cumsum:
            x = torch.sin(torch.cumsum(x, dim=-1))
        
        # x = x - x.mean()
        return x


def to_pairs(signal, random_indices=True):

    if random_indices:
        indices = torch.randperm(n_samples)[:n_sub_samples]
    else:
        indices = torch.arange(n_samples)

    signal = torch.from_numpy(signal).to(device)
    signal = signal.permute(1, 0)
    signal = signal[indices]
    signal = signal.permute(1, 0)

    pos = torch.linspace(-1, 1, n_samples).view(-1, 1).to(device)[indices]

    pos = pos_encode_feature(
        pos, 1, n_samples, n_embedding_freqs).repeat(batch_size, 1, 1)

    samples = signal[..., None]

    return pos, samples


def real():
    _, samples = get_batch(random_indices=False, batch_size=1)
    return zounds.AudioSamples(inverse_transform(samples.data.cpu().numpy().reshape(-1)), sr).pad_with_silence()


def fake():
    with torch.no_grad():
        pos, _ = get_batch(random_indices=False, batch_size=1)
        z = latent()
        samples = gen(pos, z)
    return zounds.AudioSamples(inverse_transform(samples.data.cpu().numpy().reshape(-1)), sr).pad_with_silence()


def latent():
    return torch.FloatTensor(batch_size, 1, 128).normal_(0, 1).to(device)


def train_gen(pos, samples):
    
    gen_optim.zero_grad()
    z = latent()
    fake = gen(pos, z)
    fj = disc(pos, fake)
    loss = torch.abs(1 - fj).mean()
    loss.backward()
    gen_optim.step()
    print('G', loss.item())


def train_disc(pos, samples):
    disc_optim.zero_grad()
    z = latent()
    fake = gen(pos, z)
    rj = disc(pos, samples)
    fj = disc(pos, fake)
    loss = (torch.abs(rj - 1).mean() + torch.abs(fj - 0).mean()) * 0.5
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


def get_batch(random_indices=True, batch_size=batch_size):
    sig = next(batch_stream(path, '*.wav', batch_size, n_samples))
    sig /= (sig.max(axis=-1, keepdims=True) + 1e-12)
    sig = transform(sig)
    pos, samples = to_pairs(sig, random_indices)
    return pos, samples


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    gen = Network(6, 33, 128).to(device)
    gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
    disc = Disc(128).to(device)
    disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

    while True:

        pos, samples = get_batch()
        train_disc(pos, samples)

        pos, samples = get_batch()
        train_gen(pos, samples)
