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


def fft_shift(a, shift):
    n_samples = a.shape[-1]
    shift_samples = shift * n_samples
    spec = torch.fft.rfft(a, dim=-1)

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1)
    return samples


if __name__ == '__main__':
    window_size = 1024
    step_size = window_size // 2
    n_coeffs = window_size // 2 + 1
    n_samples = 2 ** 16
    n_frames = n_samples // step_size

    batch_size = 3
    control_plane_dim = 16

    latent_space_dim = 128

    cp_to_latent = torch.zeros(control_plane_dim, latent_space_dim).uniform_(-1, 1)

    latent_to_samples = torch.zeros(latent_space_dim, window_size).uniform_(-1, 1)
    latent_to_shift = torch.zeros(latent_space_dim, 1).uniform_(-1, 1)

    cp = torch.zeros(batch_size, n_frames, control_plane_dim).bernoulli_(p=0.01) * torch.zeros(batch_size, n_frames, control_plane_dim).uniform_(0, 2)

    decays = torch.zeros(batch_size, latent_space_dim).uniform_(0.2, 0.9)
    latent = decays.view(batch_size, 1, latent_space_dim).repeat(1, n_frames, 1)
    latent = torch.log(latent)

    latent = latent + torch.relu(cp @ cp_to_latent)

    # cumulative product in log-space
    latent = torch.cumsum(latent, dim=1)
    latent = torch.exp(latent)

    samples = latent @ latent_to_samples
    shifts = latent @ latent_to_shift

    samples = fft_shift(samples, shifts)

    final = overlap_add(samples[:, None, :, :], apply_window=True)

    samples = playable(final, 22050, normalize=True)
    listen_to_sound(samples)
