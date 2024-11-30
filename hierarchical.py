import torch
import numpy as np
from conjure import LmdbCollection, loggers, serve_conjure
from torch import nn
from torch.nn import functional as F

from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import unit_norm, iterative_loss, flattened_multiband_spectrogram, sparse_softmax, max_norm, \
    gammatone_filter_bank
from itertools import count
from torch.optim import Adam
from typing import Tuple
from modules.transfer import hierarchical_dirac, fft_convolve, make_waves
from util.music import musical_scale_hz


class OverfitHierarchicalEvents(nn.Module):
    def __init__(
            self,
            n_samples: int,
            samplerate: int,
            atom_samples: int,
            n_atoms: int,
            n_events: int,
            soft: bool = False):

        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.atom_samples = atom_samples
        self.n_atoms = n_atoms
        self.n_events = n_events
        self.soft = soft

        self.event_time_dim = int(np.log2(self.n_samples))

        self.atoms = nn.Parameter(torch.zeros(n_atoms, atom_samples).uniform_(-0.01, 0.01))

        self.amplitudes = nn.Parameter(torch.zeros(1, n_events, 1).uniform_(-1, 1))
        self.atom_choice = nn.Parameter(torch.zeros(n_events, n_atoms).uniform_(-1, 1))
        self.times = nn.Parameter(torch.zeros(1, n_events, self.event_time_dim, 2).uniform_(-1, 1))

    @property
    def normalized_atoms(self):
        return unit_norm(self.atoms, dim=-1)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:

        # first, a hard choice of atom for this event, and pad to the correct size
        if self.soft:
            ac = torch.softmax(self.atom_choice, dim=-1) @ self.normalized_atoms
        else:
            ac = sparse_softmax(self.atom_choice, dim=-1, normalize=True) @ self.normalized_atoms

        ac = ac.view(-1, self.n_events, self.atom_samples)
        ac = F.pad(ac, (0, self.n_samples - self.atom_samples))

        # apply amplitudes to the unit norm atoms
        amps = self.amplitudes ** 2
        with_amps = ac * amps

        # produce the times from the hierarchical time vectors
        times = hierarchical_dirac(self.times, soft=self.soft)
        scheduled = fft_convolve(with_amps, times)
        return scheduled, amps



def loss_transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def reconstruction_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    target_spec = loss_transform(target)
    recon_spec = loss_transform(recon)
    loss = torch.abs(target_spec - recon_spec).sum()
    return loss


def overfit():
    n_samples = 2 ** 15
    samplerate = 22050
    atom_samples = 512
    n_atoms = 16
    n_events = 64


    # Begin: this would be a nice little helper to wrap up
    collection = LmdbCollection(path='hierarchical')
    collection.destroy()
    collection = LmdbCollection(path='hierarchical')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)


    audio = get_one_audio_segment(n_samples, samplerate, device='cpu')
    target = audio.view(1, 1, n_samples)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=8888, n_workers=1)
    # end proposed helper function



    model = OverfitHierarchicalEvents(
        n_samples, samplerate, atom_samples, n_atoms, n_events, soft=False)
    optim = Adam(model.parameters(), lr=1e-3)

    for i in count():
        optim.zero_grad()
        recon, amps = model.forward()

        recon_summed = torch.sum(recon, dim=1, keepdim=True)
        recon_audio(max_norm(recon_summed))

        loss = iterative_loss(target, recon, loss_transform)
        # loss = reconstruction_loss(target, recon)
        sparsity_loss = torch.abs(amps).sum()

        loss = loss + sparsity_loss

        loss.backward()
        optim.step()
        print(i, loss.item())


if __name__ == '__main__':
    overfit()