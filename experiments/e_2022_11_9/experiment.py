import torch
from torch import nn
from torch.distributions import Normal
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm
from modules.pos_encode import pos_encoded
from modules.stft import stft
from train.optim import optimizer
from util import playable
from util.readmedocs import readme
import numpy as np
from torch.nn import functional as F
from modules.reverb import NeuralReverb
from config import Config

from util.weight_init import make_initializer

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

init_weights = make_initializer(0.02)

n_events = 32
n_harmonics = 16
samples_per_frame = 128
min_center_freq = 20
max_center_freq = 4000


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


class Resonance(nn.Module):
    def __init__(self, n_samples, samples_per_frame, factors, samplerate, min_freq_hz, max_freq_hz):
        super().__init__()
        self.n_samples = n_samples
        self.register_buffer('factors', factors)
        self.n_freqs = factors.shape[0]
        self.samples_per_frame = samples_per_frame
        self.n_frames = n_samples // samples_per_frame
        
        self.samplerate = samplerate
        self.nyquist = self.samplerate // 2
        self.min_freq = min_freq_hz / self.nyquist
        self.max_freq = max_freq_hz / self.nyquist
        self.freq_scale = self.max_freq - self.min_freq

    def forward(self, f0, res):
        batch, n_events, _ = f0.shape
        batch, n_events, n_freqs = res.shape

        # first, we need freq values for all harmonics
        f0 = self.min_freq + (self.freq_scale * f0) 
        freqs = f0 * self.factors[None, None, :]
        freqs = freqs[..., None].repeat(1, 1, 1, self.n_frames).view(-1, 1, self.n_frames)
        freqs = F.interpolate(freqs, size=self.n_samples, mode='linear')

        # we also need resonance values for each harmonic        
        res = res[..., None].repeat(1, 1, 1, self.n_frames)
        res = torch.cumprod(res, dim=-1).view(-1, 1, self.n_frames)
        res = F.interpolate(res, size=self.n_samples, mode='linear')

        # generate resonances
        final = res * torch.sin(torch.cumsum(freqs * 2 * np.pi, dim=-1))
        final = final.view(batch, n_events, n_freqs, self.n_samples)
        final = torch.mean(final, dim=2)
        return final


class BandLimitedNoise(nn.Module):
    def __init__(self, n_samples, samplerate, min_center_freq_hz=20, max_center_freq_hz=4000):
        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.nyquist = self.samplerate // 2
        self.min_center_freq = min_center_freq_hz / self.nyquist
        self.max_center_freq = max_center_freq_hz / self.nyquist
        self.window = Window(n_samples // 2 + 1,
                             self.min_center_freq, self.max_center_freq)

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
    def __init__(self, n_samples, n_events, n_harmonics, samples_per_frame):
        super().__init__()
        self.n_samples = n_samples
        self.n_events = n_events
        self.n_harmonics = n_harmonics
        self.samples_per_frame = samples_per_frame
        self.impulses = Window(n_samples, 0, 1)
        self.noise = BandLimitedNoise(n_samples, int(exp.samplerate))

        self.resonance = Resonance(
            n_samples, 
            samples_per_frame=samples_per_frame, 
            factors=torch.arange(1, n_harmonics + 1),
            samplerate=int(exp.samplerate),
            min_freq_hz=min_center_freq,
            max_freq_hz=max_center_freq)
        
        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)
        self.n_rooms = self.verb.n_rooms
        
        self.ln = nn.Linear(10, 11)

        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(exp.model_dim, 2, out_channels=self.n_rooms)

        self.to_f0 = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events)
        self.to_harmonics = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events * n_harmonics)

        self.to_amplitudes = LinearOutputStack(
            exp.model_dim, 2, out_channels=n_events)

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

        mx = torch.sigmoid(self.to_mix(x))
        rooms = torch.softmax(self.to_room(x), dim=-1)

        f0 = torch.sigmoid(self.to_f0.forward(x)).view(
            batch, self.n_events, 1)
        res = torch.sigmoid(self.to_harmonics(x)).view(
            batch, self.n_events, self.n_harmonics)

        freq_means = torch.sigmoid(self.to_freq_means(x)).view(
            batch, self.n_events, 1)
        freq_stds = torch.sigmoid(self.to_freq_stds(x)).view(
            batch, self.n_events, 1) * 0.1

        time_means = torch.sigmoid(self.to_time_means(x)).view(
            batch, self.n_events, 1)
        time_stds = torch.sigmoid(self.to_time_stds(x)).view(
            batch, self.n_events, 1) * 0.1

        amps = torch.sigmoid(self.to_amplitudes(x)).view(
            batch, self.n_events, 1)

        windows = self.noise.forward(freq_means, freq_stds)
        events = self.impulses.forward(time_means, time_stds)

        resonances = self.resonance.forward(f0, res)

        # locate events in time and scale by amplitude
        located = windows * events * amps

        # convolve impulses with resonances
        located = located + fft_convolve(located, resonances)

        located = torch.mean(located, dim=1, keepdim=True)

        dry = located
        wet = self.verb.forward(located, rooms)

        located = (dry * (1 - mx)) + (wet * mx)

        return located


model = Model(
    exp.n_samples, 
    n_events, 
    n_harmonics, 
    samples_per_frame)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
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
