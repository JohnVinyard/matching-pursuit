import torch
from torch import nn

from modules.softmax import sparse_softmax
from modules.transfer import make_waves
from util.music import musical_scale_hz
from torch.nn import functional as F
from typing import Literal

SelectionType = Literal['sparse_softmax', 'gumbel_softmax', 'softmax', 'relu', 'identity']


def hard_choice(
        selections: torch.Tensor,
        selection_type: SelectionType = 'sparse_softmax') -> torch.Tensor:

    if selection_type == 'sparse_softmax':
        selections = sparse_softmax(selections, normalize=True, dim=-1)
    elif selection_type == 'identity':
        return selections
    elif selection_type == 'gumbel_softmax':
        selections = F.gumbel_softmax(selections, tau=0.1, hard=True, dim=-1)
    elif selection_type == 'softmax':
        selections = torch.softmax(selections, dim=-1)
    elif selection_type == 'relu':
        selections = torch.relu(selections)
    else:
        raise ValueError(f'{selection_type} is an unknown selection type')

    return selections


def select_items(
        selections: torch.Tensor,
        items: torch.Tensor,
        selection_type: SelectionType = 'sparse_softmax'):
    selections = hard_choice(selections, selection_type)
    selected = selections @ items
    return selected


class QuantizedResonanceMixture(nn.Module):
    def __init__(
            self,
            n_resonances: int,
            quantize_dim: int,
            n_samples: int,
            samplerate: int,
            hard_func=lambda x: sparse_softmax(x, normalize=True, dim=-1)):

        super().__init__()
        self.n_resonances = n_resonances
        self.quantize_dim = quantize_dim
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.hard_func = hard_func

        self.to_quantized_dim = nn.Linear(self.n_resonances, self.quantize_dim)
        self.to_resonance_choice = nn.Linear(self.quantize_dim, self.n_resonances)

        f0s = musical_scale_hz(start_midi=21, stop_midi=106, n_steps=n_resonances // 4)
        waves = make_waves(n_samples, f0s, samplerate)
        self.register_buffer('waves', waves.view(1, n_resonances, n_samples))

    def forward(self, x: torch.Tensor, return_code: bool = False):
        batch, n_events, dim = x.shape
        # assert dim == self.quantize_dim

        x = self.to_quantized_dim.forward(x)
        quantized = self.hard_func(x)
        choice = self.to_resonance_choice(quantized)
        choice = torch.relu(choice)
        resonances = choice @ self.waves

        assert resonances.shape == (batch, n_events, self.n_samples)

        if return_code:
            return quantized, resonances
        else:
            return resonances
