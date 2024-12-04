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
from modules.transfer import fft_convolve, fft_shift
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from torch.nn import functional as F
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
        a = torch.cat([self.atoms, torch.zeros(1, self.n_atoms, self.n_samples - self.atom_samples, device=self.atoms.device)], dim=-1)

        return a

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, _ , time = audio.shape

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


if __name__ == '__main__':
    # train()

    n_frames = 256
    control_dim = 16

    latent_dim = 32

    fade_template = torch.linspace(0, 1, n_frames)

    event_latent = torch.zeros(1, 1, latent_dim).uniform_(-0.01, 0.01)


    to_fade = torch.zeros(latent_dim, control_dim).uniform_(-1, 1)
    hyper = HyperNetworkLayer(latent_dim, 64, layer_in_channels=16, layer_out_channels=512)


    control_plane = torch.zeros(1, n_frames, control_dim).bernoulli_(0.001)
    fade = torch.softmax(event_latent @ to_fade, dim=-1) * 40

    fade = fade_template.view(1, 1, 1, n_frames) ** fade.view(1, 1, control_dim, 1)
    plt.matshow(fade.view(control_dim, n_frames))
    plt.show()

    delays = torch.eye(n_frames)
    fades = fft_convolve(fade[..., None].repeat(1, 1, 1, 1, n_frames), delays.view(1, 1, 1, n_frames, n_frames))
    plt.matshow(fades[0, 0, 0, :, :])
    plt.show()

    w, forward = hyper.forward(event_latent)
    sig = forward(control_plane)

    windowed = sig.view(1, 1, n_frames, 512)
    signal = overlap_add(windowed, apply_window=True)
    p = playable(signal, 22050, normalize=True)
    listen_to_sound(p)
