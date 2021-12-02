

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
import math
from random import choice

"""
Train three networks in lock-step:

1) an embedding network that puts adjacent audio clips nearby in latent space
2) a generator that can produce realistic audio from the embeddings
3) a discriminator that can tell the difference between real and fake embeddings
"""
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sr = zounds.SR22050()


n_filters = 128
filter_size = 512

scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), n_filters)
fb = zounds.learn.FilterBank(
    sr,
    filter_size,
    scale,
    0.01,
    normalize_filters=True,
    a_weighting=False).to(device)
fb.filter_bank = fb.filter_bank * 0.01

overfit = False
batch_size = 1 if overfit else 8
n_samples = 2**14

network_channels = 64
latent_dim = 128


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
        
        self.final = nn.Conv1d(channels, n_filters, 7, 1, 3)
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
        x = self.expand(x)
        x = x.view(-1, self.channels, 4)
        x = self.net(x)
        x = self.final(x)
        return x

    def forward(self, z):
        encoded = self.transform_latent(z)
        encoded = encoded.view(batch_size, latent_dim, 1).repeat(1, 1, n_samples)
        
        pos = pos_encode_feature(torch.linspace(-1, 1, n_samples).to(device), 1, n_samples, 16)
        pos = pos.view(1, -1, n_samples).repeat(batch_size, 1, 1)

        x = torch.cat([pos, encoded], dim=1)
        x = self.embed(x)
        x = self.full(x)
        x = self.final(x)

        x = F.pad(x, (0, 1))
        x = fb.transposed_convolve(x)

        return encoded, x


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        layers = int(math.log(n_samples, 4))

        self.encoder = nn.Sequential(
            nn.Conv1d(n_filters, channels, 25, 1, 12),
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
        x = fb.convolve(x)[..., :-1]
        features = []
        for layer in self.encoder:
            if layer.__class__ == nn.Sequential:
                for sub_layer in layer:
                    x = sub_layer(x)
                    features.append(x.view(batch_size, -1))
            else:
                x = layer(x)
                features.append(x.view(batch_size, -1))

        encoded = x        
        encoded = encoded.view(batch_size, latent_dim)
        features = torch.cat(features, dim=1)
        return features, encoded
    
    def forward(self, x, return_features=False):
        features, encoded = self.encode(x)
        encoded = encoded.view(batch_size, 1, latent_dim)
        disc = self.disc(encoded)
        if return_features:
            return features, encoded, disc
        else:
            return encoded, disc



gen = Generator(latent_dim, network_channels).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator(network_channels).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

real_target = 0.9
fake_target = 0


def train_gen(samples):
    gen_optim.zero_grad()

    # embed
    features, encoded = disc.encode(samples)

    # generate from embedding
    _, fake = gen(encoded.clone().detach())

    # judge generation
    fake_features, fe, fj = disc(fake, return_features=True)

    # minimize distance between disc feature spaces
    feature_loss = torch.abs(features - fake_features).sum()

    loss = feature_loss
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake


def train_disc(samples):
    disc_optim.zero_grad()
    re, rj = disc(samples)

    _, fake = gen(re.clone().detach())
    fe, fj = disc(fake)


    loss = (torch.abs(rj - real_target).mean() + torch.abs(fj - fake_target).mean()) * 0.5
    loss.backward()
    disc_optim.step()
    print('D', loss.item())
    return fake, samples


def covariance(x):
    m = x.mean(dim=0, keepdim=True)
    x = x - m
    cov = torch.matmul(x.T, x.clone().detach()) * (1 / x.shape[1])
    return cov

def train_embedder(samples):
    disc.zero_grad()

    a, b = samples[:, :, :n_samples], samples[:, :, n_samples:]

    _, a_embed = disc.encode(a)
    _, b_embed = disc.encode(b)

    dist_loss = torch.norm(a_embed - b_embed, dim=-1).mean() * 0.1

    embedded = choice([a_embed, b_embed])
    
    mean_loss = torch.abs(0 - embedded.mean(dim=0)).mean()
    std_loss = torch.abs(1 - embedded.std(dim=0)).mean()

    cov = covariance(embedded)
    d = torch.sqrt(torch.diag(cov))
    cov = cov / d[None, :]
    cov = cov / d[:, None]
    cov = torch.abs(cov)

    full_cov = cov
    cov = cov.mean()

    loss = dist_loss + mean_loss + std_loss + cov
    loss.backward()
    disc_optim.step()
    print('E', loss.item())
    return embedded, full_cov


streams = {}

def get_batch(batch_size=batch_size, n_samples=n_samples):
    try:
        sig = next(streams[(batch_size, n_samples)])
    except KeyError:
        stream = batch_stream(path, '*.wav', batch_size, n_samples)
        streams[(batch_size, n_samples)] = stream
        sig = next(stream)
    sig /= (sig.max(axis=-1, keepdims=True) + 1e-12)
    samples = torch.from_numpy(sig).to(device).float()
    return samples.view(batch_size, 1, n_samples)


def fake():
    return zounds.AudioSamples(f, sr).pad_with_silence()

def real():
    return zounds.AudioSamples(r, sr).pad_with_silence()

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    samples = get_batch(batch_size, n_samples)

    while True:
        samples = get_batch(batch_size, n_samples * 2)
        emb, fcov = train_embedder(samples)

        samples = get_batch(batch_size, n_samples)
        fk, orig = train_disc(samples)

        if not overfit:
            samples = get_batch(batch_size, n_samples)        
        fk = train_gen(samples)

        e = emb.data.cpu().numpy()
        r = samples.data.cpu().numpy()[0].squeeze()
        f = fk.data.cpu().numpy()[0].squeeze()
        fc = fcov.data.cpu().numpy()



    

