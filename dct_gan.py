

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
batch_size = 8
n_samples = 2**14
n_embedding_freqs = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_channels = 128


path = '/home/john/workspace/audio-data/musicnet/train_data'


torch.backends.cudnn.benchmark = True


def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.13, 0.13)
        except AttributeError:
            pass


def activation(x):
    return F.leaky_relu(x, 0.2)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return activation(x)


class Gen(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pos_embedding = LinearOutputStack(
            channels,
            3,
            in_channels=n_embedding_freqs * 2 + 1,
            activation=activation)
        self.latent_embedding = LinearOutputStack(
            channels,
            3,
            activation=activation)

        self.comb = LinearOutputStack(
            channels, 3, in_channels=channels * 2, activation=activation)
        self.final = LinearOutputStack(
            channels, 3, out_channels=1, activation=activation)

        self.apply(init_weights)
        self.envelope = nn.Parameter(torch.FloatTensor(n_samples).fill_(1))


    def forward(self, pos, latent):
        p = self.pos_embedding(pos)
        l = self.latent_embedding(latent).repeat(1, n_samples, 1)

        x = torch.cat([p, l], dim=-1)
        x = self.comb(x)
        x = self.final(x)
        x = x * self.envelope[None, :, None]
        return x


class Disc(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.pos_embedding = LinearOutputStack(
            channels,
            3,
            in_channels=n_embedding_freqs * 2 + 1,
            activation=activation)

        self.coeff_embedding = LinearOutputStack(
            channels, 3, in_channels=1, activation=activation)
        self.comb = LinearOutputStack(
            channels, 3, in_channels=channels * 2, activation=activation)

        n_reduction_layers = int(np.log2(n_samples))
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(channels, channels, 2, 2, 0),
                    Activation()
                )
                for _ in range(n_reduction_layers)
            ])
        self.final = LinearOutputStack(
            channels, 3, out_channels=1, activation=None)

        self.apply(init_weights)
        self.envelope = nn.Parameter(torch.FloatTensor(n_samples).fill_(1))


    def forward(self, pos, samples):
        p = self.pos_embedding(pos)
        c = self.coeff_embedding(samples * self.envelope[None, :, None])
        x = torch.cat([p, c], dim=-1)
        x = self.comb(x)

        # shuffle randomly
        indices = torch.randperm(n_samples)
        x = x.permute(1, 0, 2)
        x = x[indices]
        x = x.permute(1, 0, 2)

        # reduce
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)

        x = torch.sigmoid(self.final(x))
        return x


def transform(x):
    return dct(x, axis=-1, norm='ortho')


def inverse_transform(x):
    return idct(x, axis=-1, norm='ortho')


def real():
    _, samples = get_batch(random_indices=False, batch_size=1)
    return zounds.AudioSamples(
        inverse_transform(samples.data.cpu().numpy().reshape(-1)),
        sr).pad_with_silence()


def fake():
    with torch.no_grad():
        pos, _ = get_batch(random_indices=False, batch_size=1)
        z = latent()
        samples = gen(pos, z)
    return zounds.AudioSamples(
        inverse_transform(samples.data.cpu().numpy().reshape(-1)),
        sr).pad_with_silence()


def latent():
    return torch.FloatTensor(batch_size, 1, 128).normal_(0, 1).to(device)


gen = Gen(network_channels).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
disc = Disc(network_channels).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


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
    return fake, samples


def get_batch(batch_size=batch_size):
    sig = next(batch_stream(path, '*.wav', batch_size, n_samples))
    sig /= (sig.max(axis=-1, keepdims=True) + 1e-12)
    sig = transform(sig)

    samples = torch.from_numpy(sig).to(device).float()
    pos = pos_encode_feature(torch.linspace(-1, 1, n_samples), 1, n_samples, n_embedding_freqs)\
        .view(1, n_samples, -1).repeat(batch_size, 1, 1)
    return samples[..., None], pos


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    while True:
        samples, pos = get_batch()
        train_gen(pos, samples)

        samples, pos = get_batch()
        fake, orig = train_disc(pos, samples)

        f = fake.data.cpu().numpy()[0].squeeze()
        r = orig.data.cpu().numpy()[0].squeeze()
