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
from modules import max_norm, stft, fft_frequency_decompose
from modules.overlap_add import overlap_add
from torch.optim import Adam
import numpy as np
from scipy.signal import gammatone
from torch.nn import functional as F
from functools import reduce
from util import device

from conjure import LmdbCollection, audio_conjure, serve_conjure, numpy_conjure, SupportedContentType
from io import BytesIO
from soundfile import SoundFile
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

def gammatone_filter_bank(n_filters: int, size: int, device) -> torch.Tensor:
    bank = np.zeros((n_filters, size))

    frequencies = np.linspace(
        20,
        11000,
        num=n_filters)

    for i, freq in enumerate(frequencies):
        b, a = gammatone(
            freq=freq,
            ftype='fir',
            order=4,
            numtaps=size,
            fs=22050)

        bank[i] = b

    bank = bank / np.abs(bank).max(axis=-1, keepdims=True)
    bank = torch.from_numpy(bank).to(device).float()
    return bank


def fft_convolve(*args, norm=None) -> torch.Tensor:
    n_samples = args[0].shape[-1]

    # pad to avoid wraparound artifacts
    padded = [F.pad(x, (0, x.shape[-1])) for x in args]

    specs = [torch.fft.rfft(x, dim=-1, norm=norm) for x in padded]
    spec = reduce(lambda accum, current: accum * current, specs[1:], specs[0])
    final = torch.fft.irfft(spec, dim=-1, norm=norm)

    # remove padding
    return final[..., :n_samples]


def n_fft_coeffs(x: int):
    return x // 2 + 1


def auditory_image_model(
        signal: torch.Tensor,
        filters: torch.Tensor,
        aim_window_size:
        int, aim_step_size) -> torch.Tensor:
    n_samples = signal.shape[-1]

    n_filters, n_taps = filters.shape

    filters = filters.view(1, n_filters, n_taps)
    padded_filters = F.pad(filters, (0, n_samples - n_taps))
    spec = fft_convolve(signal, padded_filters)

    # half-wave rectification
    spec = torch.relu(spec)
    spec = spec.unfold(-1, aim_window_size, aim_step_size)
    aim = torch.abs(torch.fft.rfft(spec, dim=-1))
    return aim



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


def audio(x: torch.Tensor):
    x = x.data.cpu().numpy()[0].reshape((-1,))
    io = BytesIO()

    with SoundFile(
            file=io,
            mode='w',
            samplerate=samplerate,
            channels=1,
            format='WAV',
            subtype='PCM_16') as sf:
        sf.write(x)

    io.seek(0)
    return io.read()


collection = LmdbCollection(path='ssm')

@audio_conjure(storage=collection)
def recon_audio(x: torch.Tensor):
    return audio(x)


@audio_conjure(storage=collection)
def orig_audio(x: torch.Tensor):
    return audio(x)

@numpy_conjure(storage=collection, content_type=SupportedContentType.Spectrogram.value)
def envelopes(x: torch.Tensor):
    return x.data.cpu().numpy()

@numpy_conjure(storage=collection, content_type=SupportedContentType.Spectrogram.value)
def state_space(x: torch.Tensor):
    return x.data.cpu().numpy()

@audio_conjure(storage=collection)
def random_audio(x: torch.Tensor):
    return audio(x)

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
        control_plane_dim, window_size, state_dim, n_samples = target.shape[-1]).to(device)

    optim = Adam(model.parameters(), lr=1e-2)
    gammatone_bank = gammatone_filter_bank(n_filters=512, size=512, device=device)

    def perceptual_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # fake_spec = auditory_image_model(
        #     recon, gammatone_bank, aim_window_size=512, aim_step_size=128)
        # real_spec = auditory_image_model(
        #     target, gammatone_bank, aim_window_size=512, aim_step_size=128)

        # fake_spec = stft(recon, ws=2048, step=256, pad=True)
        # real_spec = stft(target, ws=2048, step=256, pad=True)
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

def transform(x: torch.Tensor) -> torch.Tensor:
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.reshape(batch_size, channels, -1) for b in bands.values()], dim=-1)


def multiband_transform(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    bands = fft_frequency_decompose(x, 512)
    # TODO: each band should have 256 frequency bins and also 256 time bins
    # this requires a window size of (n_samples // 256) * 2
    # and a window size of 512, 256

    window_size = 512

    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}

    normal = stft(x, 2048, 256, pad=True).reshape(-1, 128, 1025).permute(0, 2, 1)

    return dict(
        normal=normal,
        **d1,
        **d3,
        **d4
    )


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