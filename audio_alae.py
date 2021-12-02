

import torch
from torch.nn.modules.linear import Linear
from torch.optim.adam import Adam
import zounds
from torch import nn

from datastore import batch_stream
from modules import pos_encode_feature
from modules3 import LinearOutputStack
from torch.nn import functional as F
import numpy as np
from scipy.fftpack import dct, idct
from itertools import chain
import math

sr = zounds.SR22050()
batch_size = 8
n_samples = 2**14
n_embedding_freqs = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_channels = 64
latent_dim = 128

nyquist_cycles_per_sample = 0.5
nyquist_radians_per_sample = nyquist_cycles_per_sample * (2 * np.pi)
freq_bands = torch.from_numpy(np.geomspace(0.001, 1, 64)).to(device).float()

path = '/hdd/musicnet/train_data'


torch.backends.cudnn.benchmark = True


def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.1, 0.1)
        except AttributeError:
            pass


def activation(x):
    return F.leaky_relu(x, 0.2)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return activation(x)


def latent(batch_size):
    return torch.FloatTensor(batch_size, latent_dim).normal_(0, 1).to(device)

class Generator(nn.Module):
    def __init__(self, latent_dim, channels):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.transform = LinearOutputStack(
            channels, 
            4, 
            in_channels=latent_dim,
            out_channels=latent_dim)
        
        self.expand = nn.Linear(latent_dim, channels * 4)
        
        layers = int(math.log(n_samples, 4)) - int(math.log(4, 4))

        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose1d(channels, channels, 26, 4, 11),
                nn.LeakyReLU(0.2)
            )
             for _ in range(layers)
        ])
        self.final = nn.Conv1d(channels, 1, 25, 1, 12)
        # self.amp = nn.Conv1d(channels, 1, 1, 1, 0)
        # self.freq = nn.Conv1d(channels, 1, 1, 1, 0)

        self.embed = nn.Conv1d(33 + latent_dim, channels, 1, 1, 0)

        self.full = nn.Sequential(*[
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
        ])
        

        self.apply(init_weights)
    
    def transform_latent(self, z):
        z = z.view(-1, self.latent_dim)
        encoded = x = self.transform(z)
        return encoded
    
    def generate(self, x):

        pos = pos_encode_feature(torch.linspace(-1, 1, n_samples), 1, n_samples, 16)
        pos = pos.view(1, -1, n_samples).repeat(batch_size, 1, 1)

        x = torch.cat([pos, x], dim=1)
        x = self.embed(x)
        x = self.full(x)
        x = self.final(x)

        # x = self.expand(x)
        # x = x.view(-1, self.channels, 4)
        # x = self.net(x)
        # x = self.final(x)

        # amp = torch.relu(self.amp(x))
        # amp = F.avg_pool1d(amp, 255, 1, 127)

        # freq = torch.sigmoid((self.freq(x))) * nyquist_radians_per_sample * freq_bands[None, :, None]

        # x = amp * torch.sin(torch.cumsum(freq, dim=-1))
        # x = x.sum(dim=1, keepdim=True)
        return x


    def forward(self, z):
        encoded = self.transform_latent(z)
        x = self.generate(encoded)
        return encoded, x


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = int(math.log(n_samples, 4))

        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, 25, 1, 12),
            nn.Sequential(*[
                nn.Sequential(
                    nn.Conv1d(channels, channels, 25, 4, 12),
                    nn.LeakyReLU(0.2)
                )
             for _ in range(layers)]),
            nn.Conv1d(channels, latent_dim, 1, 1, 0)
        )

        self.disc = LinearOutputStack(channels, 4, in_channels=latent_dim, out_channels=1)
    
        self.apply(init_weights)
    
    def encode(self, x):
        encoded = self.encoder(x)
        return encoded
    
    def forward(self, x):
        encoded = self.encode(x)
        encoded = encoded.permute(0, 2, 1)
        disc = self.disc(encoded)
        return encoded, disc



gen = Generator(latent_dim, network_channels).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator(network_channels).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

real_target = 0.9
fake_target = 0


def train_gen(samples):
    gen_optim.zero_grad()
    z = latent(batch_size=samples.shape[0])
    fe, fake = gen(z)
    de, fj = disc(fake)
    loss = torch.abs(real_target - fj).mean()
    loss.backward()
    gen_optim.step()
    print('G', loss.item())


def train_disc(samples):
    disc_optim.zero_grad()
    z = latent(batch_size=samples.shape[0])
    fe, fake = gen(z)
    _, rj = disc(samples)
    _, fj = disc(fake)
    loss = (torch.abs(rj - real_target).mean() + torch.abs(fj - fake_target).mean()) * 0.5
    loss.backward()
    disc_optim.step()
    print('D', loss.item())
    return fake, samples


def train_latent(samples):
    gen_optim.zero_grad()
    disc_optim.zero_grad()
    z = latent(batch_size=samples.shape[0])

    encoded = gen.transform_latent(z)
    generated = gen.generate(encoded)
    de = disc.encode(generated)


    loss = F.mse_loss(de.view(batch_size, latent_dim), encoded.view(batch_size, latent_dim))
    loss.backward()
    gen_optim.step()
    disc_optim.step()
    print('L', loss.item())

    return encoded, de


def get_samples(batch_size=batch_size):
    sig = next(batch_stream(path, '*.wav', batch_size, n_samples))
    sig /= (sig.max(axis=-1, keepdims=True) + 1e-12)
    return sig


def get_batch(batch_size=batch_size):
    sig = get_samples(batch_size)
    samples = torch.from_numpy(sig).to(device).float()
    return samples.view(batch_size, 1, n_samples)


def fake():
    return zounds.AudioSamples(f, sr).pad_with_silence()

def real():
    return zounds.AudioSamples(r, sr).pad_with_silence()

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    print('testing...')
    z = latent(batch_size=4)
    e, f = gen(z)
    print(e.shape, f.shape)
    de, j = disc(f)
    print(de.shape, j.shape)

    while True:
        samples = get_batch()
        train_gen(samples)

        samples = get_batch()
        fake_sample, orig_sample = train_disc(samples)

        samples = get_batch()
        gen_encoded, disc_encoded = train_latent(samples)

        gz = gen_encoded.data.cpu().numpy().squeeze()
        dz = disc_encoded.data.cpu().numpy().squeeze()
        f = fake_sample.data.cpu().numpy()[0].squeeze()
        r = orig_sample.data.cpu().numpy()[0].squeeze()

