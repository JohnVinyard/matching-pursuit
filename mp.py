from typing import Union, Literal

import torch
from torch.optim import Adam
from matplotlib import pyplot as plt
from conjure import LmdbCollection, serve_conjure, loggers
from torch import nn

from conjure.logger import encode_audio
from data import AudioIterator, get_one_audio_segment
from modules import unit_norm, sparsify2, stft, iterative_loss, fft_frequency_decompose, gammatone_filter_bank, \
    max_norm, HyperNetworkLayer
from modules.latent_loss import covariance
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio, morlet_filter_bank
from modules.transfer import fft_convolve, fft_shift, freq_domain_transfer_function_to_resonance
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from torch.nn import functional as F

from modules.upsample import upsample_with_holes
from util import device, playable
import numpy as np

import matplotlib

from util.music import musical_scale
from util.playable import listen_to_sound

matplotlib.use('Qt5Agg')


class MatchingPursuit(nn.Module):
    def __init__(self, n_atoms: int, atom_samples: int, n_samples: int, n_iterations: int):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_samples = atom_samples
        self.n_samples = n_samples
        self.n_iterations = n_iterations

        self.atoms = nn.Parameter(torch.zeros(1, self.n_atoms, self.atom_samples).uniform_(-0.01, 0.01))

    @property
    def normalized_atoms(self):
        a = torch.cat(
            [self.atoms, torch.zeros(1, self.n_atoms, self.n_samples - self.atom_samples, device=self.atoms.device)],
            dim=-1)

        return a

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, _, time = audio.shape

        residual = audio

        na = self.normalized_atoms

        channels = torch.zeros(batch, self.n_iterations, self.n_samples, device=audio.device)

        for i in range(self.n_iterations):
            spec = fft_convolve(residual, na)
            sparse, time, atom = sparsify2(spec, n_to_keep=1)
            a = atom @ na
            b = fft_convolve(a, time)
            residual = residual - b
            channels[:, i: i + 1, :] = b

        return channels


def transform(x: torch.Tensor) -> torch.Tensor:
    spec = stft(x, 2048, 256, pad=True)
    return spec


def train():
    collection = LmdbCollection(path='mp')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=9999, n_workers=1)

    n_samples = 2 ** 15
    ai = AudioIterator(batch_size=1, n_samples=n_samples, samplerate=22050, normalize=True)

    model = MatchingPursuit(n_atoms=128, atom_samples=1024, n_samples=n_samples, n_iterations=25).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for i, target in enumerate(iter(ai)):
        optim.zero_grad()
        target = target.view(-1, 1, n_samples).to(device)
        orig_audio(target)
        recon = model.forward(target)

        rs = torch.sum(recon, dim=1, keepdim=True)
        recon_audio(rs)

        loss = iterative_loss(target, recon, transform)
        loss.backward()
        print(i, loss.item())
        optim.step()



def positional_encoding(
        sequence_length: int,
        n_freqs: int,
        geometric_freq_spacing: bool = False,
        geometric_freq_decay: bool = False):

    time = torch.linspace(-np.pi, np.pi, sequence_length)
    freqs = torch.linspace(1, sequence_length // 2, n_freqs)
    if geometric_freq_spacing:
        freqs = freqs ** 2

    scaling = torch.linspace(1, 1e-8, n_freqs)
    if geometric_freq_decay:
        scaling = scaling ** 2

    x = torch.sin(time[None, :] * freqs[:, None]) * scaling[:, None]
    return x


if __name__ == '__main__':

    window_size = 2048
    n_coeffs = window_size // 2 + 1
    n_samples = 2 ** 16
    step_size = 256
    n_frames = n_samples // step_size
    control_plane_dim = 16
    n_resonances = 16
    n_deformations = 4
    batch_size = 2
    n_events = 3

    resonances = torch.zeros(1, n_resonances, n_coeffs).uniform_(0.2, 0.9) * torch.zeros(1, 1, n_coeffs).bernoulli_(p=0.01)
    resonance = freq_domain_transfer_function_to_resonance(window_size, resonances, n_samples // (window_size // 2), apply_decay=True)

    cp = torch.zeros(batch_size, n_events, control_plane_dim, n_frames).bernoulli_(p=0.005)

    routing = torch.zeros(n_deformations, control_plane_dim, n_resonances).uniform_(-1, 1)



    x = routing[None, None, :, :, :].permute(0, 1, 2, 4, 3) @ cp[:, :, None, :, :]
    x = upsample_with_holes(x, desired_size=n_samples)
    r = resonance.view(1, 1, 1, n_resonances, n_samples)
    x = fft_convolve(x, r)

    x = torch.sum(x, dim=3)


    mixture = torch.zeros(1, n_deformations, n_samples).uniform_(-1, 1)

    mixture = F.avg_pool1d(mixture, 4096, stride=1, padding=2048)[:, None, :, :-1]
    mixture = torch.softmax(mixture, dim=2)
    x = x * mixture



    x = torch.sum(x, dim=2)


    x = torch.tanh(x * 3)


    x = max_norm(x)

    p = playable(x[0, 0, :], 22050, normalize=True)
    listen_to_sound(p)
