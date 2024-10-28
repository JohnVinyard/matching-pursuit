# the size, in samples of the audio segment we'll overfit
from modules.atoms import unit_norm

n_samples = 2 ** 18

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

# the size of each, half-lapped audio "frame"
window_size = 512

# the dimensionality of the control plane or control signal
control_plane_dim = 32

# the dimensionality of the state vector, or hidden state
state_dim = 128

from itertools import count
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer
from data import get_one_audio_segment
from modules import max_norm, flattened_multiband_spectrogram, limit_norm
from modules.overlap_add import overlap_add
from util import device, encode_audio


'''
def limit_norm(x, dim=2, max_norm=0.9999):
    norm = torch.norm(x, dim=dim, keepdim=True)
    unit_norm = x / (norm + 1e-8)
    clamped_norm = torch.clamp(norm, 0, max_norm)
    x = unit_norm * clamped_norm
    return x
'''

def project_and_limit_norm(vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    original_norm = torch.norm(vector, dim=-1, keepdim=True)
    x = vector @ matrix
    new_norm = torch.norm(x, dim=-1, keepdim=True)
    clamped_norm = torch.clamp(new_norm, min=None, max=original_norm)
    normalized = unit_norm(x, axis=-1)
    x = normalized * clamped_norm
    return x



class SSM(nn.Module):
    """
    A state-space model-like module, with one additional matrix, used to project the control
    signal into the shape of each audio frame.

    The final output is produced by overlap-adding the windows/frames of audio into a single
    1D signal.
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int):
        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        # state matrix mapping previous state vector to next state vector
        self.state_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        # matrix mapping audio frame to hidden/state vector dimension
        self.input_matrix = nn.Parameter(
            torch.zeros(input_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        # matrix mapping hidden/state vector to audio frame dimension
        self.output_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, input_dim).uniform_(-0.01, 0.01)
        )

        # skip-connection-like matrix mapping input audio frame to next
        # output audio frame
        self.direct_matrix = nn.Parameter(
            torch.zeros(input_dim, input_dim).uniform_(-0.01, 0.01)
        )

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)


        # proj = control @ self.proj
        proj = project_and_limit_norm(control, self.proj)
        assert proj.shape == (batch, frames, self.input_dim)

        results = []
        state_vec = torch.zeros(batch, self.state_matrix_dim, device=control.device)

        for i in range(frames):
            inp = proj[:, i, :]
            state_vec = project_and_limit_norm(state_vec, self.state_matrix)
            b = project_and_limit_norm(inp, self.input_matrix)
            c = project_and_limit_norm(state_vec, self.output_matrix)
            d = project_and_limit_norm(inp, self.direct_matrix)

            # state_vec = (state_vec @ self.state_matrix) + (inp @ self.input_matrix)
            # output = (state_vec @ self.output_matrix) + (inp @ self.direct_matrix)

            state_vec = state_vec + b
            output = c + d
            results.append(output.view(batch, 1, self.input_dim))

        result = torch.cat(results, dim=1)
        result = result[:, None, :, :]

        result = overlap_add(result)
        return result[..., :frames * (self.input_dim // 2)]


class OverfitControlPlane(nn.Module):
    """
    Encapsulates parameters for control signal and state-space model
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, n_samples: int):
        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim)
        self.n_samples = n_samples
        self.n_frames = int(n_samples / (input_dim // 2))

        self.control = nn.Parameter(
            torch.zeros(1, control_plane_dim, self.n_frames).uniform_(-0.01, 0.01))

    @property
    def control_signal_display(self) -> np.ndarray:
        return self.control_signal.data.cpu().numpy().reshape((-1, self.n_frames))

    @property
    def control_signal(self) -> torch.Tensor:
        return torch.relu(self.control)

    def random(self, p=0.001):
        """
        Produces a random, sparse control signal, emulating short, transient bursts
        of energy into the system modelled by the `SSM`
        """
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=p)
        audio = self.forward(sig=cp)
        return max_norm(audio)

    def forward(self, sig=None):
        """
        Inject energy defined by `sig` (or by the `control` parameters encapsulated by this class)
        into the system modelled by `SSM`
        """
        return self.ssm.forward(sig if sig is not None else self.control_signal)


def transform(x: torch.Tensor):
    """
    Decompose audio into sub-bands of varying sample rate, and compute spectrogram with
    varying time-frequency tradeoffs on each band.
    """
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the difference between the `recon` and `target`
    representations
    """
    fake_spec = transform(recon)
    real_spec = transform(target)
    return torch.abs(fake_spec - real_spec).sum()


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the control signal
    """
    return torch.abs(c).sum() * 1e-5


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def construct_experiment_model(state_dict: Union[None, dict] = None) -> OverfitControlPlane:
    """
    Construct a randomly initialized `OverfitControlPlane` instance, ready for training/overfitting
    """
    model = OverfitControlPlane(
        control_plane_dim=control_plane_dim,
        input_dim=window_size,
        state_matrix_dim=state_dim,
        n_samples=n_samples
    )
    model = model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def train_and_monitor():
    target = get_one_audio_segment(n_samples=n_samples, samplerate=samplerate)
    collection = LmdbCollection(path='ssm')

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

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
        envelopes,
        random_audio,
        state_space
    ], port=9999, n_workers=1)

    def train(target: torch.Tensor):
        model = construct_experiment_model()

        optim = Adam(model.parameters(), lr=1e-3)

        for iteration in count():
            optim.zero_grad()
            recon = model.forward()
            recon_audio(max_norm(recon))
            loss = reconstruction_loss(recon, target) + sparsity_loss(model.control_signal)

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
                random_audio(rnd)

    train(target)


if __name__ == '__main__':
    train_and_monitor()
