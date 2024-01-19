import numpy as np
import scipy as sp
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from modules.ddsp import NoiseModel, OscillatorBank
from modules.pif import AuditoryImage
from train.optim import optimizer
from upsample import ConvUpsample

from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
from modules.latent_loss import latent_loss

n_samples = 2 ** 15
samplerate = zounds.SR22050()
n_frames = n_samples // 256
n_noise_frames = n_frames * 8
model_dim = 128
aim_window_size = 256
aim_coeffs = (aim_window_size // 2) + 1
filter_kernel_size = 128
n_freqs = 128

n_atoms = 16

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, n_freqs)

init_weights = make_initializer(0.1)

time_dim = torch.linspace(-np.pi, np.pi, n_frames)
freq_dim = torch.linspace(-np.pi, np.pi, n_freqs)

x, y = torch.meshgrid(time_dim, freq_dim)
grid = torch.cat([x[None, ...], y[None, ...]], dim=0)

n_pos_freqs = 8

pos = [grid]

for i in range(n_pos_freqs):
    pos.append(torch.sin(grid * (2 ** i)))
    pos.append(torch.cos(grid * (2 ** i)))

pos = torch.cat(pos, dim=0).to(device)
norms = torch.norm(pos, dim=0, keepdim=True)
pos = pos / (norms + 1e-8)

fb = zounds.learn.FilterBank(
    samplerate,
    filter_kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)

aim = AuditoryImage(
    aim_window_size,
    n_frames,
    do_windowing=False,
    check_cola=False).to(device)


def perceptual_feature(x):
    x = torch.abs(fb.convolve(x))
    x = F.avg_pool1d(x, 512, 256, 256)
    return x[:, None, :, :n_frames]


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)


class AudioModel(nn.Module):
    def __init__(self, n_samples=8192):
        super().__init__()
        self.n_samples = n_samples

        self.up = ConvUpsample(
            model_dim,
            model_dim,
            4,
            32,
            mode='nearest',
            out_channels=model_dim)

        self.osc = OscillatorBank(
            model_dim,
            model_dim,
            n_samples,
            constrain=True,
            lowest_freq=30 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)

        self.noise = NoiseModel(
            model_dim,
            n_samples // 256,
            (n_samples // 256) * 8,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)

    def forward(self, x):
        x = self.up(x)
        harm = self.osc.forward(x)
        noise = self.noise(x)
        signal = harm + noise
        return signal


class SparseBlock(nn.Module):
    def __init__(
            self,
            channels,
            time,
            freq,
            sparse_ratio=None,
            sparse_count=None):

        super().__init__()

        if sparse_ratio is not None:
            self.k = int((time * freq) * sparse_ratio)
        elif sparse_count is not None:
            self.k = sparse_count
        else:
            raise ValueError('Please specfity either a ratio or a count')

        self.time = time
        self.freq = freq
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)
        self.values = nn.Conv2d(channels, channels, 1, 1, 0)
        self.select = nn.Conv2d(channels, 1, 1, 1, 0)

    def forward(self, x):
        batch = x.shape[0]

        x = self.conv(x)

        norms = torch.norm(x, dim=1, keepdim=True)
        normed = x / (norms + 1e-8)
        s = self.select(normed)
        s = s.view(x.shape[0], self.time * self.freq)
        s = torch.softmax(s, dim=-1)
        values, indices = torch.topk(s, k=self.k, dim=-1)
        s = s.reshape(x.shape[0], self.time, self.freq)

        values = self.values(x)

        freq = indices // n_freqs
        time = indices % n_frames

        # out = torch.zeros(
        #     x.shape[0], self.channels, self.freq, self.time).to(x.device)
        
        latents = []

        for b in range(batch):
            for a in range(self.k):
                f = freq[b, a]
                t = time[b, a]

                val = s[b, f, t]
                val = val + (1 - val)

                latent = values[b, :, f, t] * val
                # out[b, :, f, t] = latent * val
                latents.append(latent[None, ...])
        
        latents = torch.cat(latents, dim=0)

        return None, indices, latents, s.view(batch, n_freqs, n_frames)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial = nn.Conv2d(
            1, model_dim - pos.shape[0], 3, 1, 1)
        
        self.with_pos = nn.Conv2d(model_dim, model_dim, 3, 1, 1)

        self.sparse = nn.Sequential(
            nn.Conv2d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

            nn.Conv2d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(model_dim, model_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(model_dim, model_dim, 3, 1, 1),
            nn.LeakyReLU(0.2),
            # nn.Dropout(0.2),

            SparseBlock(model_dim, n_frames, n_freqs, sparse_count=16),
        )

        self.audio = AudioModel(n_samples=8192)

        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]
        x = self.initial(x)

        p = pos[None, ...].repeat(x.shape[0], 1, 1, 1)
        x = torch.cat([x, p], dim=1)
        x = self.with_pos(x)

        _, indices, latents, feat_map = self.sparse(x)

        latents = latents.view(batch, n_atoms, model_dim).reshape(-1, model_dim)

        atoms = self.audio.forward(latents)
        atoms = atoms.view(batch, n_atoms, 8192)

        output = torch.zeros(batch, 1, n_samples + 8192).to(device)

        for b in range(batch):
            for a in range(n_atoms):
                atom = atoms[b, a]
                ps = indices[b, a]
                output[b, :, ps: ps + 8192] += atom

        output = output[:, :, :n_samples]
        return output, latents.view(batch, n_atoms, model_dim), feat_map


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    x = perceptual_feature(batch)
    recon, latents, feat_map = model.forward(x)
    recon_feat = perceptual_feature(recon)

    loss = F.mse_loss(recon_feat, x.clone()) #+ latent_loss(latents.view(-1, model_dim))

    loss.backward()
    optim.step()
    return recon, loss, latents, feat_map


@readme
class DeepSparse(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.latents = None
        self.feat_map = None

    def orig(self):
        return playable(self.real, samplerate)

    def listen(self):
        return playable(self.fake, samplerate)
    
    def z(self):
        return self.latents.data.cpu().numpy()[0].squeeze()
    
    def map(self):
        return self.feat_map.data.cpu().numpy()[0].squeeze()

    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))

    def positions(self):
        return pos.data.cpu().numpy()
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)

            self.real = item
            self.fake, loss, self.latents, self.feat_map = train(item)

            if i % 10 == 0:
                print(loss.item())
