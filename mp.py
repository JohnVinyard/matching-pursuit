import torch
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, loggers
from torch import nn

from conjure.logger import encode_audio
from data import AudioIterator
from modules import unit_norm, sparsify2, stft, iterative_loss
from modules.transfer import fft_convolve

from util import device


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
    train()