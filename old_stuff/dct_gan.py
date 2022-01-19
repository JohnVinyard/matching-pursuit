

import torch
from torch.optim.adam import Adam
import zounds
from torch import nn

from data.datastore import batch_stream
from modules import pos_encode_feature
from modules3 import LinearOutputStack
from torch.nn import functional as F
import numpy as np
from scipy.fftpack import dct, idct
from itertools import chain

sr = zounds.SR22050()
batch_size = 2
n_samples = 8192
n_embedding_freqs = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network_channels = 64
latent_dim = 128
ae_mode = True

nyquist_cycles_per_sample = 0.5
nyquist_radians_per_sample = nyquist_cycles_per_sample * (2 * np.pi)
freq_bands = torch.from_numpy(np.geomspace(0.001, 1, 64)).to(device).float()


path = '/hdd/musicnet/train_data'

# stats = next(batch_stream(path, '*.wav', 64, n_samples))
# stats /= (stats.max(axis=-1, keepdims=True) + 1e-8)
# stats = dct(stats, axis=-1, norm='ortho')
# stats_batch = stats
# stats = stats.std(axis=0, keepdims=True)

torch.backends.cudnn.benchmark = True


def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.2, 0.2)
        except AttributeError:
            pass


def activation(x):
    # return F.leaky_relu(x, 0.2)
    return torch.sin(x)


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
            in_channels=latent_dim,
            activation=activation)

        # self.comb = LinearOutputStack(
        #     channels, 3, in_channels=channels * 2, activation=activation)
        self.final = LinearOutputStack(
            channels, 6, activation=activation)

        self.pos = nn.Parameter(torch.FloatTensor(1, n_samples, n_embedding_freqs * 2 + 1).normal_(0, 1))

        self.amp = LinearOutputStack(channels, 3)
        self.freq = LinearOutputStack(channels, 3)

        self.apply(init_weights)

    def forward(self, latent, pos):
        p = self.pos_embedding(self.pos.repeat(latent.shape[0], 1, 1))
        l = self.latent_embedding(latent).repeat(1, n_samples, 1)

        # x = torch.cat([p, l], dim=-1)
        x = p + l
        # x = self.comb(x)
        x = self.final(x)

        amp = torch.relu(self.amp(x))
        amp = amp.permute(0, 2, 1)
        amp = F.avg_pool1d(amp, 255, 1, 127)
        amp = amp.permute(0, 2, 1)

        freq = torch.sigmoid((self.freq(x))) * nyquist_radians_per_sample * freq_bands[None, None, :]

        x = amp * torch.sin(torch.cumsum(freq, dim=1))
        x = x.mean(dim=-1, keepdim=True)
        return x


class Disc(nn.Module):
    def __init__(self, channels, is_encoder=False):
        super().__init__()

        self.is_encoder = is_encoder

        self.pos_embedding = LinearOutputStack(
            channels,
            3,
            in_channels=n_embedding_freqs * 2 + 1,
            activation=activation, bias=False)

        self.coeff_embedding = LinearOutputStack(
            channels, 3, in_channels=1, activation=activation, bias=False)
        self.comb = LinearOutputStack(
            channels, 3, activation=activation, bias=False)

        n_reduction_layers = int(np.log2(n_samples))
        self.net = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(channels, channels, 2, 2, 0, bias=False),
                    Activation()
                )
                for _ in range(n_reduction_layers)
            ])
        self.final = LinearOutputStack(
            channels, 3, out_channels=latent_dim if is_encoder else 1, activation=activation, bias=False)
        
        self.pos = nn.Parameter(torch.FloatTensor(1, n_samples, n_embedding_freqs * 2 + 1).normal_(0, 1))

        self.apply(init_weights)

    def forward(self, pos, samples):


        p = self.pos_embedding(self.pos.repeat(samples.shape[0], 1, 1))
        c = self.coeff_embedding(samples)
        x = p + c
        x = self.comb(x)

        # shuffle randomly (TODO: utility func)
        # indices = torch.randperm(n_samples)
        # x = x.permute(1, 0, 2)
        # x = x[indices]
        # x = x.permute(1, 0, 2)

        # reduce
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.final(x)

        if self.is_encoder:
            return x
        else:
            return torch.sigmoid(x)


def transform(x):
    # return dct(x, axis=-1, norm='ortho') / stats.reshape(1, n_samples)
    return x


def inverse_transform(x):
    # return idct(x * stats.reshape(1, n_samples), axis=-1, norm='ortho')
    return x


def real():
    return zounds.AudioSamples(inverse_transform(r), sr).pad_with_silence()


def fake():
    return zounds.AudioSamples(inverse_transform(f), sr).pad_with_silence()


def latent(batch_size=batch_size):
    return torch.FloatTensor(batch_size, 1, latent_dim).normal_(0, 1).to(device)


encoder = Disc(network_channels, is_encoder=True).to(device)
gen = Gen(network_channels).to(device)
ae_optim = Adam(chain(encoder.parameters(), gen.parameters()), lr=1e-4, betas=(0, 0.9))

gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))
disc = Disc(network_channels).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

real_target = 0.9
fake_target = 0

def train_ae(pos, samples):
    ae_optim.zero_grad()
    encoded = encoder(pos, samples)
    decoded = gen(encoded, pos)
    loss = F.mse_loss(decoded, samples)
    loss.backward()
    ae_optim.step()
    print('AE', loss.item())
    return decoded, samples, encoded


def train_gen(pos, samples):
    gen_optim.zero_grad()
    z = latent(batch_size=samples.shape[0])
    fake = gen(z, pos)
    fj = disc(pos, fake)
    loss = torch.abs(real_target - fj).mean()
    loss.backward()
    gen_optim.step()
    print('G', loss.item())


def train_disc(pos, samples):
    disc_optim.zero_grad()
    z = latent(batch_size=samples.shape[0])
    fake = gen(z, pos)
    rj = disc(pos, samples)
    fj = disc(pos, fake)
    loss = (torch.abs(rj - real_target).mean() + torch.abs(fj - fake_target).mean()) * 0.5
    loss.backward()
    disc_optim.step()
    print('D', loss.item())
    return fake, samples


def get_samples(batch_size=batch_size):
    sig = next(batch_stream(path, '*.wav', batch_size, n_samples))
    sig /= (sig.max(axis=-1, keepdims=True) + 1e-12)
    sig = transform(sig)
    return sig


def get_batch(batch_size=batch_size):
    sig = get_samples(batch_size)

    pos = pos_encode_feature(
        torch.linspace(-1, 1, n_samples).view(-1, 1).to(device),
        domain=1,
        n_samples=n_samples,
        n_freqs=n_embedding_freqs) \
        .view(1, n_samples, -1) \
        .repeat(batch_size, 1, 1)

    samples = torch.from_numpy(sig).to(device).float()
    return samples[..., None], pos


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    while True:

        if ae_mode:
            samples, pos = get_batch()
            fake_sample, orig_sample, enc = train_ae(pos, samples)
            e = enc.data.cpu().numpy().squeeze()
        else:
            samples, pos = get_batch()
            train_gen(pos, samples)

            samples, pos = get_batch()
            fake_sample, orig_sample = train_disc(pos, samples)

        # monitoring
        # p = pos.data.cpu().numpy()[0]
        f = fake_sample.data.cpu().numpy()[0].squeeze()
        r = orig_sample.data.cpu().numpy()[0].squeeze()

