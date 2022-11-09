import torch
from torch import nn
from torch.distributions import Normal
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm
from modules.pos_encode import pos_encoded
from modules.stft import stft
from train.optim import optimizer
from util import playable
from util.readmedocs import readme
import numpy as np
from torch.nn import functional as F

from util.weight_init import make_initializer

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

init_weights = make_initializer(0.02)

n_events = 32
min_center_freq = 20 / exp.samplerate.nyquist
max_center_freq = 4000 / exp.samplerate.nyquist


class Window(nn.Module):
    def __init__(self, n_samples, mn, mx):
        super().__init__()
        self.n_samples = n_samples
        self.mn = mn
        self.mx = mx
        self.scale = self.mx - self.mn

    def forward(self, means, stds):
        dist = Normal(self.mn + (means * self.scale), stds)
        rng = torch.linspace(0, 1, self.n_samples)[None, None, :]
        windows = torch.exp(dist.log_prob(rng))
        return windows


class BandLimitedNoise(nn.Module):
    def __init__(self, n_samples, samplerate, min_center_freq_hz=20, max_center_freq_hz=4000):
        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.nyquist = self.samplerate // 2
        self.min_center_freq = min_center_freq_hz / self.nyquist
        self.max_center_freq = max_center_freq_hz / self.nyquist
        self.window = Window(n_samples // 2 + 1, self.min_center_freq, self.max_center_freq)


    def forward(self, center_frequencies, bandwidths):
        windows = self.window.forward(center_frequencies, bandwidths)
        n = torch.zeros(1, 1, self.n_samples).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(n, dim=-1, norm='ortho')
        # filter noise in the frequency domain
        filtered = noise_spec * windows
        # invert
        band_limited_noise = torch.fft.irfft(filtered, dim=-1, norm='ortho')
        return band_limited_noise


class Model(nn.Module):
    def __init__(self, n_samples, n_events):
        super().__init__()
        self.n_samples = n_samples
        self.n_events = n_events
        self.impulses = Window(n_samples, 0, 1)
        self.noise = BandLimitedNoise(n_samples, int(exp.samplerate))
        self.ln = nn.Linear(10, 11)

        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.to_freq_means = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events)
        self.to_freq_stds = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events)

        self.to_time_means = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events)
        self.to_time_stds = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events)

        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]
        target = x = x.view(-1, 1, exp.n_samples)
        x = exp.fb.forward(x, normalize=False)
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        x = self.norm(x)
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)
        x = self.context(x)
        x = self.norm(x)

        x, _ = torch.max(x, dim=-1)

        freq_means = torch.sigmoid(self.to_freq_means(x)).view(
            batch, self.n_events, 1)
        freq_stds = torch.sigmoid(self.to_freq_stds(x)).view(
            batch, self.n_events, 1) * 0.1

        time_means = torch.sigmoid(self.to_time_means(x)).view(
            batch, self.n_events, 1)
        time_stds = torch.sigmoid(self.to_time_stds(x)).view(
            batch, self.n_events, 1)

        windows = self.noise.forward(freq_means, freq_stds)
        events = self.impulses.forward(time_means, time_stds)

        located = windows * events
        located = torch.mean(located, dim=1, keepdim=True)
        return located


model = Model(exp.n_samples, n_events)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    # loss = exp.perceptual_loss(recon, batch)

    rspec = stft(batch)
    fspec = stft(recon)
    loss = F.mse_loss(rspec, fspec)

    loss.backward()
    optim.step()
    return recon, loss


@readme
class ResonantAtomsExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.win = None
        self.real = None

    def orig(self):
        return playable(self.real, exp.samplerate)

    def orig_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def noise(self):
        return playable(self.win[0, 0], exp.samplerate)

    def noise_spec(self):
        return np.abs(zounds.spectral.stft(self.noise()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            w, loss = train(item)
            self.win = w
            print(loss.item())
