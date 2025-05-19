from typing import Literal
from torch import nn
import torch
import numpy as np

from modules import LinearOutputStack
from modules.normal_pdf import gamma_pdf
from modules.reds import interpolate_last_axis
from util import make_initializer
from torch.nn import functional as F


import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

BandType = Literal['linear', 'geometric']

initializer = make_initializer(0.02)

def generate_envelope(n_envelopes: int, resolution: int, device: torch.device):
    a = 1e-12 + torch.zeros(n_envelopes, 1, device=device).uniform_(0, 10)
    b = 1e-12 + torch.zeros(n_envelopes, 1, device=device).uniform_(0, 10)
    env = gamma_pdf(a, b, resolution, normalize=True)
    return env

def generate_training_batch(n_examples: int, resolution: int, envelope_resolution: int, device: torch.device):
    """
    Each entry in the infinite/synthetic dataset will consist of the following
    pairs:

        inputs:  event_time, event_duration, event_envelope, timespan
        targets:
    """
    start_times = torch.zeros(n_examples, device=device).uniform_(0, 1)
    durations = torch.zeros(n_examples, device=device).uniform_(1e-3, 1)
    envelopes = generate_envelope(n_examples, envelope_resolution, device)

    start_samples = torch.floor(start_times * resolution).int()
    duration_samples = torch.floor(durations * resolution).int()
    end_samples = start_samples + duration_samples

    target = torch.zeros(n_examples, 1, resolution, device=device)

    for i in range(n_examples):
        rasterized_envelope = interpolate_last_axis(envelopes[i: i + 1, ...], desired_size=duration_samples[i])
        last_sample = end_samples[i]
        diff = max(0, last_sample.item() - resolution)
        target_size = target[i, :, start_samples[i].item(): end_samples[i].item() - diff].shape[-1]

        # print(
        #     target[i, :, start_samples[i].item(): end_samples[i].item() - diff].shape,
        #     rasterized_envelope[..., :-diff].shape
        # )
        target[i, :, start_samples[i].item(): end_samples[i].item() - diff] = rasterized_envelope[..., :target_size]

    return target



class PosEncoder(nn.Module):

    def __init__(
            self,
            n_bands: int,
            band_type: BandType = 'linear',
            max_freq: float = 128,
            min_freq: float = 0.01):

        super().__init__()
        self.n_bands = n_bands

        if band_type == 'linear':
            freqs = np.linspace(min_freq, max_freq, num=n_bands)
        else:
            freqs = np.geomspace(max_freq, max_freq, num=n_bands)

        freqs = torch.from_numpy(freqs).float()
        self.register_buffer('freqs', freqs.view(1, 1, n_bands, 1))

    @property
    def total_bands(self):
        return self.n_bands * 2

    def get_range(self, start: float, end: float, steps: int) -> torch.Tensor:
        t = torch.linspace(0, 1, 128)[None, None, :]
        x = pe.forward(t)
        return x
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_events, time = x.shape
        s = torch.sin(x[:, :, None, :] * self.freqs)
        c = torch.cos(x[:, :, None, :] * self.freqs)
        x = torch.zeros(batch, n_events, self.total_bands, time, device=x.device)
        x[:, :, ::2, :] = s
        x[:, :, 1::2, :] = c
        return x

class Model(nn.Module):

    def __init__(
            self, envelope_resolution: int,
            pos_encoding_dim: int,
            model_dim: int):

        super().__init__()
        self.envelope_resolution = envelope_resolution
        self.pos_encoding_dim = pos_encoding_dim
        self.model_dim = model_dim

        self.embed_envelope = nn.Linear(envelope_resolution, model_dim)
        self.embed_start = nn.Linear(pos_encoding_dim, model_dim)
        self.embed_duration = nn.Linear(pos_encoding_dim, model_dim)

        self.weighting = nn.Parameter(torch.zeros(1).fill_(1))

        self.network = LinearOutputStack(
            channels=model_dim,
            layers=5,
            out_channels=1,
            in_channels=model_dim,
            activation=lambda x: torch.sin(x * self.weighting)
        )

        self.apply(initializer)

    def forward(
            self,
            start: torch.Tensor,
            duration: torch.Tensor,
            envelope: torch.Tensor,
            resolution: int) -> torch.Tensor:

        batch, n_events, _ = start.shape

        start = self.embed_start(start)
        duration = self.embed_duration(duration)
        envelope = self.embed_envelope(envelope)

        x = start + duration + envelope
        x = self.network(x)
        x = x.view(batch, n_events, resolution)
        return x



if __name__ == '__main__':
    batch_size = 16
    resolution = 128
    envelope_resolution = 32

    b = generate_training_batch(
        batch_size, resolution, envelope_resolution, device=torch.device('cpu'))

    plt.matshow(b[:, 0, :].data.cpu().numpy())
    plt.show()
