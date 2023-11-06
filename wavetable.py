import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal
import numpy as np
from matplotlib import pyplot as plt


def max_norm(x, dim=-1, epsilon=1e-8, return_value=False):
    n, _ = torch.max(torch.abs(x), dim=dim, keepdim=True)
    normed = x / (n + epsilon)
    if return_value:
        return normed, n
    else:
        return normed


class WavetableSynth(nn.Module):
    def __init__(self, table_size: int, n_tables: int, total_samples: int, samplerate: int):
        super().__init__()
        self.samplerate = samplerate
        self.total_samples = total_samples
        self.n_tables = n_tables
        self.table_size = table_size

        sp = torch.sin(torch.linspace(-np.pi, np.pi, table_size)) * 0.9
        noise = torch.zeros(self.n_tables, self.table_size).uniform_(-1, 1)
        self.wavetables = nn.Parameter()
        nyquist = self.samplerate // 2
        self.min_freq = 40 / nyquist
        self.max_freq = 4000 / nyquist
        self.freq_range = self.max_freq - self.min_freq

        self.max_smoothness = 0.05

    def get_tables(self):
        return max_norm(self.wavetables)

    def forward(self, env: torch.Tensor, table_selection: torch.Tensor, freq: torch.Tensor, smoothness: torch.Tensor):
        batch, _, n_frames = env.shape
        assert env.shape[1] == 1

        # env must be positive, it represents overall energy over time
        env = torch.abs(env)
        env = F.interpolate(env, size=self.total_samples, mode='linear')

        batch, n_tables, n_frames = table_selection.shape
        assert n_tables == self.n_tables

        # ts represents the mixture of tables
        ts = torch.softmax(table_selection, dim=1)
        ts = F.interpolate(ts, size=self.total_samples, mode='linear')

        batch, _, n_frames = freq.shape
        assert freq.shape[1] == 1

        # freq is a scalar representing how quickly we loop through the wavetables
        # concretely it represents the mean of the gaussian used as a "read head"
        freq = self.min_freq + (torch.sigmoid(freq) * self.freq_range)
        freq = F.interpolate(freq, size=self.total_samples, mode='linear')
        freq = torch.cumsum(freq, dim=-1) % 1

        batch, _, n_frames = smoothness.shape
        assert smoothness.shape[1] == 1

        # smoothness is a scalar that represents a low-pass filter
        # concretely, it represents the std of the gaussian used as a "read head"
        smoothness = torch.sigmoid(smoothness) * self.max_smoothness
        smoothness = F.interpolate(smoothness, size=self.total_samples, mode='linear')

        tables = self.get_tables()

        x = tables.T @ ts
        assert x.shape[1:] == (self.table_size, self.total_samples)

        dist = Normal(freq, smoothness)
        rng = torch.linspace(0, self.table_size, self.table_size, device=x.device)[None, :, None]
        read = torch.exp(dist.log_prob(rng))

        samples = (read * x).sum(dim=1, keepdim=True)
        samples = samples * env
        return samples


if __name__ == '__main__':
    table_size = 512
    n_tables = 16
    total_samples = 2 ** 15
    samplerate = 22050

    batch_size = 4
    n_frames = 128

    model = WavetableSynth(table_size, n_tables, total_samples, samplerate)
    env = torch.zeros(batch_size, 1, n_frames).uniform_(-1, 1)
    sel = torch.zeros(batch_size, n_tables, n_frames).uniform_(-1, 1)
    freq = torch.zeros(batch_size, 1, n_frames).uniform_(-1, 1)
    smoothness = torch.zeros(batch_size, 1, n_frames).uniform_(-1, 1)

    result = model.forward(env, sel, freq, smoothness)
    print(result.shape)