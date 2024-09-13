"""[markdown]

This work takes a slightly different approach to the problem of decomposing
audio into a sparse and interpretable representation.  It models the physical
system that produced a segment of "natural"
(i.e., produced by acoustic instruments) musical audio as a state-space-model and
attempts to find a sparse control signal for the system.  The control system
can be thought of as the energy injected into the system by a human musician,
corresponding roughly to a score, and the state-space model can be thought of
as the dynamics and resonances of the musical instrument and room in which it
was played.

"""

from typing import Dict

import torch
from torch import nn
from itertools import count

from avenues.conjurearticle import conjure_article
from data import AudioIterator
from modules import max_norm, stft, fft_frequency_decompose, flattened_multiband_spectrogram
from modules.overlap_add import overlap_add
from torch.optim import Adam
from util import device, encode_audio

from conjure import logger, LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer, audio_conjure, Conjure
from torch.nn.utils.clip_grad import clip_grad_value_
from argparse import ArgumentParser

"""[markdown]
# The Model

blah blah

## Something

blah

### Nested

## Something Else

"""

class SSM(nn.Module):
    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int):
        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim

        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        self.state_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        self.input_matrix = nn.Parameter(
            torch.zeros(input_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        self.output_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, input_dim).uniform_(-0.01, 0.01)
        )
        self.direct_matrix = nn.Parameter(
            torch.zeros(input_dim, input_dim).uniform_(-0.01, 0.01)
        )

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)

        proj = control @ self.proj
        assert proj.shape == (batch, frames, self.input_dim)

        results = []
        state_vec = torch.zeros(batch, self.state_matrix_dim, device=control.device)

        for i in range(frames):
            inp = proj[:, i, :]
            state_vec = (state_vec @ self.state_matrix) + (inp @ self.input_matrix)
            output = (state_vec @ self.output_matrix) + (inp @ self.direct_matrix)
            results.append(output.view(batch, 1, self.input_dim))

        result = torch.cat(results, dim=1)
        result = result[:, None, :, :]

        result = overlap_add(result)
        return result[..., :frames * (self.input_dim // 2)]


class OverfitControlPlane(nn.Module):

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, n_samples: int):
        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim)
        self.n_samples = n_samples
        self.n_frames = int(n_samples / (input_dim // 2))

        self.control = nn.Parameter(
            torch.zeros(1, control_plane_dim, self.n_frames).uniform_(-0.01, 0.01))

    @property
    def control_signal(self):
        return torch.relu(self.control)

    def random(self):
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=0.001)
        return self.forward(sig=cp)

    def forward(self, sig=None):
        return self.ssm.forward(sig if sig is not None else self.control_signal)


collection = LmdbCollection(path='ssm')

def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


recon_audio, orig_audio, random_audio = loggers(
    ['recon', 'orig', 'random'],
    'audio/wav',
    encode_audio,
    collection)

envelopes, state_space = loggers(
    ['envelopes', 'statespace'],
    SupportedContentType.Spectrogram.value,
    to_numpy,
    collection,
    serializer=NumpySerializer(),
    deserializer=NumpyDeserializer())


"""[markdown]
# The training process

"""


def train(
        target: torch.Tensor,
        control_plane_dim: int,
        window_size: int,
        state_dim: int,
        device):
    model = OverfitControlPlane(
        control_plane_dim, window_size, state_dim, n_samples=target.shape[-1]).to(device)

    optim = Adam(model.parameters(), lr=1e-2)

    def perceptual_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        fake_spec = transform(recon)
        real_spec = transform(target)
        return torch.abs(fake_spec - real_spec).sum()

    def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
        return torch.abs(c).sum() * 1e-5

    for iteration in count():
        optim.zero_grad()
        recon = model.forward()
        recon_audio(max_norm(recon))
        loss = perceptual_loss(recon, target) + sparsity_loss(model.control_signal)

        non_zero = (model.control_signal > 0).sum()
        sparsity = (non_zero / model.control_signal.numel()).item()

        state_space(model.ssm.state_matrix)
        envelopes(model.control_signal.view(control_plane_dim, -1))
        loss.backward()

        clip_grad_value_(model.parameters(), 0.5)

        optim.step()
        print(iteration, loss.item(), sparsity)

        with torch.no_grad():
            rnd = model.random()
            rnd = max_norm(rnd)
            random_audio(rnd)


"""[markdown]

# Sparsity Loss

blah

## More about the sparsity loss 

"""


def transform(x: torch.Tensor):
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


n_samples = 2 ** 18
samplerate = 22050
window_size = 512
control_plane_dim = 32
state_dim = 128


def demo_page_dict() -> Dict[str, any]:
    return dict()


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(__file__, 'html', **display)


def train_and_monitor():

    # TODO: get_one convenience function

    ai = AudioIterator(
        batch_size=1,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True,
        overfit=True, )
    target: torch.Tensor = next(iter(ai)).to(device).view(-1, 1, n_samples)
    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
        envelopes,
        random_audio,
        state_space
    ], port=9999, n_workers=1)

    train(
        target,
        control_plane_dim=control_plane_dim,
        window_size=window_size,
        state_dim=state_dim,
        device=device)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str)
    args = parser.parse_args()
    if args.mode == 'train':
        train_and_monitor()
    elif args.mode == 'demo':
        generate_demo_page()
