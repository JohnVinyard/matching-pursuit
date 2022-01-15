from tkinter import E
from torch import nn
import zounds
import numpy as np
from torch.nn import functional as F
from datastore import batch_stream
from torch.optim import Adam
import torch

from modules import pos_encode_feature

samplerate = zounds.SR22050()
path = '/home/john/workspace/audio-data/musicnet/train_data'
n_samples = 2**15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 128
n_channels = 16
init_value = 0.1


def do_fft(x):
    return torch.fft.rfft(x).real


def stft(x):
    x = x.unfold(-1, 512, 256)
    x = x * torch.hamming_window(512)[None, None, None, :].to(device)
    x = torch.fft.rfft(x)
    x = torch.abs(x)
    return x


def mse_loss(generated, orig):
    g = stft(generated)
    o = stft(orig)
    return F.mse_loss(g, o), o


def fft_loss(generated, orig, means=None, stds=None):
    g = do_fft(generated)
    o = do_fft(orig)

    if means is not None:
        g = g - means
        o = o - means

    if stds is not None:
        g = g / stds
        o = o / stds

    return F.mse_loss(g, o)


def init_weights(p):

    with torch.no_grad():
        try:
            p.weight.uniform_(-init_value, init_value)
        except AttributeError:
            pass


class ConvGenerator(nn.Module):
    def __init__(self, size, latent_dim, n_channels):
        super().__init__()
        self.size = size
        self.latent_dim = latent_dim
        self.n_channels = n_channels

        n_layers = int(np.log2(self.size) - np.log2(4))
        self.initial = nn.Conv1d(self.latent_dim, self.n_channels * 4, 1, 1, 0)

        self.net = nn.Sequential(*[
            nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.ConvTranspose1d(self.n_channels, self.n_channels, 4, 2, 1),
                nn.Conv1d(self.n_channels, self.n_channels, 7, 1, 3),
                nn.LeakyReLU(0.2)
            )
            for _ in range(n_layers)])

        self.to_samples = nn.Conv1d(self.n_channels, 1, 7, 1, 3)
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1)
        x = self.initial(x)
        x = x.view(-1, self.n_channels, 4)
        x = self.net(x)
        x = self.to_samples(x)
        return x


class ForwardBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.ln = nn.Linear(self.n_channels, self.n_channels)
        self.gate = nn.Linear(self.n_channels, self.n_channels)
        self.apply(init_weights)

    def forward(self, x):
        shortcut = x
        g = self.gate(x)
        x = self.ln(x)
        x = F.leaky_relu(x + shortcut, 0.2) * \
            (torch.sigmoid(shortcut + g) ** 2)
        return x


class FourierMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(x, dim=-1, norm='ortho')
        x = torch.fft.fft(x, dim=-2, norm='ortho')
        x = x.real
        return x


class PosEncodedGenerator(nn.Module):
    def __init__(self, size, latent_dim, n_channels):
        super().__init__()
        self.size = size
        self.latent_dim = latent_dim
        self.n_channels = n_channels

        self.pos_embed = nn.Conv1d(33, self.n_channels, 1, 1, 0)
        self.latent_embed = nn.Conv1d(latent_dim, self.n_channels, 1, 1, 0)

        self.transform = nn.Sequential(*[
            nn.Sequential(
                ForwardBlock(self.n_channels),
                FourierMixer()
            ) for _ in range(8)])

        self.to_samples = nn.Conv1d(self.n_channels, 1, 1, 1, 0)
        self.apply(init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        pos = pos_encode_feature(torch.linspace(-1, 1, self.size).view(-1, 1), 1, self.size, 16)\
            .view(1, self.size, 33)\
            .repeat(batch_size, 1, 1)\
            .permute(0, 2, 1).to(device)

        pos = self.pos_embed(pos)

        x = x.repeat(1, 1, self.size)
        x = self.latent_embed(x)
        x = x * pos

        # permute for MLP and then back for final conv
        x = x.permute(0, 2, 1)
        x = self.transform(x)
        x = x.permute(0, 2, 1)

        x = self.to_samples(x)
        return x


def fake():
    return zounds.AudioSamples(
        recon.data.cpu().numpy().reshape(-1), samplerate).pad_with_silence()


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    model = PosEncodedGenerator

    gen = model(n_samples, latent_dim, n_channels)
    optim = Adam(gen.parameters(), lr=1e-3, betas=(0, 0.9))

    stream = batch_stream(path, '*.wav', 1, n_samples)

    samples = next(stream)
    orig = zounds.AudioSamples(
        samples.reshape(-1), samplerate).pad_with_silence()

    samples = torch.from_numpy(samples).float().to(
        device).view(1, 1, n_samples)

    latent = torch.FloatTensor(1, latent_dim, 1).normal_(0, 1).to(device)

    while True:
        optim.zero_grad()

        recon = gen.forward(latent)
        l, spec = mse_loss(recon, samples)
        ospec = spec.data.cpu().numpy().squeeze()
        l.backward()
        optim.step()
        print('TIME', l.item())
