from typing import Tuple
import numpy as np
import torch
from torch import nn

from freqdomain import fft_shift
from modules import unit_norm, stft
from modules.upsample import ensure_last_axis_length
from util import make_initializer
from util.overfit import overfit_model


class Splitter(nn.Module):
    def __init__(self, latent_dim: int, branching_factor: int = 2, scale: float = 1):
        super().__init__()
        self.latent_dim = latent_dim
        self.branching_factor = branching_factor
        self.scale = scale

        self.to_time_offset = nn.Linear(latent_dim, self.branching_factor, bias=False)
        self.split = nn.Linear(latent_dim, latent_dim * branching_factor)

    def forward(self, x: torch.Tensor, base_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, n_events, latent_dim = x.shape

        to = torch.sigmoid(self.to_time_offset(x))
        offsets = base_time + (to * self.scale)
        offsets = offsets.view(batch, n_events * self.branching_factor, 1)

        split = self.split(x)
        split = split.view(batch, n_events * self.branching_factor, latent_dim)

        return offsets, split


class Model(nn.Module):

    def __init__(
            self,
            n_samples: int = 2 ** 17,
            n_events: int = 128,
            n_atoms: int = 32,
            atom_size: int = 512,
            latent_dim: int = 16):
        super().__init__()
        self.n_samples = n_samples
        self.n_events = n_events
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.latent_dim = latent_dim

        self.total_layers = int(np.log2(n_events))

        self.base_latent = nn.Parameter(torch.zeros(1, latent_dim).uniform_(-1, 1))

        self.layers = nn.ModuleList([
            Splitter(self.latent_dim, branching_factor=2, scale=1 / 2 ** i) for i in range(self.total_layers)
        ])

        self.atoms = nn.Parameter(torch.zeros(n_atoms, atom_size).uniform_(-1, 1))
        self.to_atoms = nn.Linear(latent_dim, n_atoms)
        self.to_amp = nn.Linear(latent_dim, 1)


    def _to_atoms(self, latents: torch.Tensor) -> torch.Tensor:
        atoms = self.to_atoms(latents)
        atoms = atoms @ self.atoms
        window = torch.hamming_window(self.atom_size, device=latents.device)
        atoms = atoms * window
        atoms = ensure_last_axis_length(atoms, self.n_samples)
        atoms = unit_norm(atoms)
        return atoms

    def forward(self) -> torch.Tensor:
        x = self.base_latent
        batch, latent_dim = x.shape

        x = x.view(batch, 1, latent_dim)
        base_times = torch.zeros(batch, 1, 1, device=x.device)

        for layer in self.layers:
            base_times, x = layer.forward(x, base_times)

        atoms = self._to_atoms(x)
        amps = self.to_amp(x)
        atoms = atoms * amps

        scheduled = fft_shift(atoms, base_times)
        scheduled = scheduled.view(batch, self.n_events, self.n_samples)
        scheduled = torch.sum(scheduled, dim=1, keepdim=True)
        return scheduled


def loss_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = stft(a, 2048, 256, pad=True)
    b = stft(b, 2048, 256, pad=True)
    return torch.abs(a - b).sum()

def overfit():
    n_samples = 2 ** 16

    model = Model(
        n_samples=n_samples,
        n_events=128,
        n_atoms=128,
        atom_size=512,
        latent_dim=16
    )
    overfit_model(
        n_samples=n_samples,
        model=model,
        loss_func=loss_func,
        collection_name='textural',
        learning_rate=1e-4
    )

if __name__ == '__main__':
    overfit()
