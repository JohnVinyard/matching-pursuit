import torch
from torch.distributions import Normal
from torch import nn
from modules.normalization import max_norm


class Window(nn.Module):
    def __init__(self, n_samples, mn, mx, epsilon=1e-8, padding=0):
        super().__init__()
        self.n_samples = n_samples
        self.mn = mn
        self.mx = mx
        self.scale = self.mx - self.mn
        self.epsilon = epsilon
        self.padding = padding

    def forward(self, means, stds):
        dist = Normal(self.mn + (means * self.scale), self.epsilon + stds)
        rng = torch.linspace(0, 1, self.n_samples, device=means.device)[
            None, None, :]
        windows = torch.exp(dist.log_prob(rng))
        windows = max_norm(windows)
        return windows


def harmonics(n_octaves, waveform, device):

    rng = torch.arange(1, n_octaves + 1, device=device)

    if waveform == 'sawtooth':
        return 1 / rng
    elif waveform == 'square':
        rng =  1 / rng
        rng[::2] = 0
        return rng
    elif waveform == 'triangle':
        rng = 1 / (rng ** 2)
        rng[::2] = 0
        return rng