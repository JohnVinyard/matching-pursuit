import torch
from torch import nn
import numpy as np

from modules import sparse_softmax
from modules.transfer import hierarchical_dirac, fft_convolve
from modules.upsample import upsample_with_holes
from util import device


def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # this is here to make the shift value interpretable
    shift = (1 - shift)

    n_samples = a.shape[-1]

    shift_samples = (shift * n_samples * 0.5)

    # a = F.pad(a, (0, n_samples * 2))

    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs

    shift = torch.exp(shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    # samples = torch.relu(samples)
    return samples


class DiracScheduler(nn.Module):
    def __init__(self, n_events: int, start_size: int, n_samples: int, pre_sparse: bool = False):
        super().__init__()
        self.n_events = n_events
        self.start_size = start_size
        self.n_samples = n_samples
        self.pos = nn.Parameter(
            torch.zeros(1, n_events, start_size).uniform_(-0.02, 0.02)
        )
        self.pre_sparse = pre_sparse

    def random_params(self):
        pos = torch.zeros(1, self.n_events, self.start_size, device=device).uniform_(-0.02, 0.02)
        if self.pre_sparse:
            pos = sparse_softmax(pos, normalize=True, dim=-1)
        return pos

    @property
    def params(self):
        return self.pos

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        if not self.pre_sparse:
            pos = sparse_softmax(pos, normalize=True, dim=-1)
        pos = upsample_with_holes(pos, desired_size=self.n_samples)
        final = fft_convolve(events, pos)
        return final


class FFTShiftScheduler(nn.Module):
    def __init__(self, n_events):
        super().__init__()
        self.n_events = n_events
        self.pos = nn.Parameter(torch.zeros(1, n_events, 1).uniform_(0, 1))

    def random_params(self):
        return torch.zeros(1, self.n_events, 1, device=device).uniform_(0, 1)

    @property
    def params(self):
        return self.pos

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        final = fft_shift(events, pos)
        return final


class HierarchicalDiracModel(nn.Module):
    def __init__(self, n_events: int, signal_size: int):
        super().__init__()
        self.n_events = n_events
        self.signal_size = signal_size
        n_elements = int(np.log2(signal_size))

        self.elements = nn.Parameter(
            torch.zeros(1, n_events, n_elements, 2).uniform_(-0.02, 0.02))

        self.n_elements = n_elements

    def random_params(self):
        return torch.zeros(1, self.n_events, self.n_elements, 2, device=device).uniform_(-0.02, 0.02)

    @property
    def params(self):
        return self.elements

    def schedule(self, pos: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
        x = hierarchical_dirac(pos)
        x = fft_convolve(x, events)
        return x
