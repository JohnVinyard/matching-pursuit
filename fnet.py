from tkinter import E
from torch import nn
import zounds
import numpy as np
from torch.nn import functional as F
from datastore import batch_stream
from torch.optim import Adam
import torch
from scipy.signal import stft, istft, hann
from dct_idea import least_squares_disc_loss, least_squares_generator_loss
from modules import pos_encode_feature
import lws
from itertools import cycle

from modules.transformer import Transformer


samplerate = zounds.SR22050()
path = '/hdd/musicnet/train_data'
n_samples = 2**15
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
latent_dim = 128
n_channels = 256
init_value = 0.01
batch_size = 8
ws = 512
step = 256
window = np.sqrt(hann(ws))
n_coeffs = (ws // 2) + 1
op = lws.lws(ws, step, mode='speech', perfectrec=True)

time_dim = (n_samples // step) + 1

def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-init_value, init_value)
        except AttributeError:
            pass

        try:
            p.bias.fill_(0)
        except AttributeError:
            pass



class Generator(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.embed_latent = nn.Linear(latent_dim, self.n_channels)
        self.embed_pos = nn.Linear(33, self.n_channels)

        self.transformer = Transformer(self.n_channels, 8)
        self.to_spec = nn.Linear(self.n_channels, n_coeffs)
        self.apply(init_weights)
    
    def forward(self, x):
        x = x.view(-1, 1, latent_dim)

        batch_size = x.shape[0]
        pos = pos_encode_feature(torch.linspace(-1, 1, time_dim).view(-1, 1), 1, time_dim, 16)\
            .view(1, time_dim, 33)\
            .repeat(batch_size, 1, 1)\
            .view(batch_size, time_dim, 33)\
            .to(device)

        pos = self.embed_pos(pos)
        x = self.embed_latent(x).repeat(1, time_dim, 1)

        x = pos + x
        x = self.transformer(x)
        x = self.to_spec(x)
        return torch.abs(x)

class Discriminator(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.embed_spec = nn.Linear(n_coeffs, n_channels)
        self.embed_pos = nn.Linear(33, n_channels)

        self.transformer = Transformer(self.n_channels, 8)
        self.judge = nn.Linear(self.n_channels, 1)
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.embed_spec(x)

        pos = pos_encode_feature(torch.linspace(-1, 1, time_dim).view(-1, 1), 1, time_dim, 16)\
            .view(1, time_dim, 33)\
            .repeat(batch_size, 1, 1)\
            .view(batch_size, time_dim, 33)\
            .to(device)
        pos = self.embed_pos(pos)

        x = pos + x
        x = self.transformer(x)
        x = x[:, -1:, :]
        x = self.judge(x)
        return torch.sigmoid(x)

def get_latent():
    return torch.FloatTensor(batch_size, 1, latent_dim).normal_(0, 1).to(device)

def real():
    coeffs = batch[0]
    coeffs = op.run_lws(coeffs)
    coeffs = coeffs.T
    _, recon = istft(coeffs, nperseg=ws, noverlap=step, window=window)
    return zounds.AudioSamples(recon, samplerate).pad_with_silence()

def real_spec():
    return np.log(0.0001 + batch[0])

def fake():
    coeffs = r[0]
    coeffs = op.run_lws(coeffs)
    coeffs = coeffs.T
    _, recon = istft(coeffs, nperseg=ws, noverlap=step, window=window)
    return zounds.AudioSamples(recon, samplerate).pad_with_silence()

def fake_spec():
    return np.log(0.0001 + r[0])

def to_spectral(samples):
    max = samples.max(axis=-1, keepdims=True)
    samples /= (max + 1e-12)
    _, _, spec = stft(samples, nperseg=ws, noverlap=step, window=window)
    spec = np.abs(spec).transpose(0, 2, 1)
    return spec


def spec_batch_stream():
    stream = batch_stream(path, '*.wav', batch_size, n_samples)
    for samples in stream:
        spec = to_spectral(samples)
        yield spec


gen = Generator(n_channels).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator(n_channels).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

def train_gen(batch):
    gen_optim.zero_grad()
    latent = get_latent()
    recon = gen.forward(latent)
    j = disc.forward(recon)
    # loss = torch.abs(1 - j).mean()
    loss = least_squares_generator_loss(j)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return recon

def train_disc(batch):
    disc_optim.zero_grad()
    latent = get_latent()
    recon = gen.forward(latent)
    fj = disc.forward(recon)
    rj = disc.forward(batch)
    loss = torch.abs(0 - fj).mean() + torch.abs(1 - rj).mean()
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())

steps = cycle(['gen', 'disc'])

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    for i, batch in enumerate(spec_batch_stream()):
        current_step = next(steps)

        b = torch.from_numpy(batch).float().to(device)
        if current_step == 'gen':
            rec = train_gen(b)
            r = np.abs(rec.data.cpu().numpy())
        else:
            train_disc(b)
        

