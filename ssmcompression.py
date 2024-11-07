# the size, in samples of the audio segment we'll overfit
from modules.atoms import unit_norm

n_samples = 2 ** 17

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

# the size of each, half-lapped audio "frame"
window_size = 2048

# the dimensionality of the control plane or control signal
control_plane_dim = 32

# the dimensionality of the state vector, or hidden state
state_dim = 128

is_complex = False

max_efficiency = 0.99

windowing = True

from itertools import count
from typing import Union

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer
from data import get_one_audio_segment
from modules import max_norm, flattened_multiband_spectrogram, limit_norm
from modules.overlap_add import overlap_add
from util import device, encode_audio



def project_and_limit_norm(
        vector: torch.Tensor,
        matrix: torch.Tensor,
        max_efficiency: float = max_efficiency) -> torch.Tensor:
    # get the original norm, this is the absolute max norm/energy we should arrive at,
    # given a perfectly efficient physical system
    original_norm = torch.norm(vector, dim=-1, keepdim=True)
    # project
    x = vector @ matrix

    # TODO: clamp norm should be a utility that lives in normalization
    # find the norm of the projection
    new_norm = torch.norm(x, dim=-1, keepdim=True)
    # clamp the norm between the allowed values
    clamped_norm = torch.clamp(new_norm, min=None, max=original_norm * max_efficiency)

    # give the projected vector the clamped norm, such that it
    # can have lost some or all energy, but not _gained_ any
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

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, complex: bool = False):
        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim
        self.complex = complex


        control_plane_dim = control_plane_dim  // 2 + 1 if complex else control_plane_dim
        state_matrix_dim = state_matrix_dim // 2 + 1 if complex else state_matrix_dim
        input_dim = input_dim // 2 + 1 if complex else input_dim


        self.input_dim = input_dim
        self.state_matrix_dim = state_matrix_dim

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim, dtype=torch.complex64 if complex else torch.float32).uniform_(-0.01, 0.01)
        )

        # state matrix mapping previous state vector to next state vector
        self.state_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, state_matrix_dim, dtype=torch.complex64 if complex else torch.float32).uniform_(-0.01, 0.01))

        # matrix mapping audio frame to hidden/state vector dimension
        self.input_matrix = nn.Parameter(
            torch.zeros(input_dim, state_matrix_dim, dtype=torch.complex64 if complex else torch.float32).uniform_(-0.01, 0.01))

        # matrix mapping hidden/state vector to audio frame dimension
        self.output_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, input_dim, dtype=torch.complex64 if complex else torch.float32).uniform_(-0.01, 0.01)
        )

        # skip-connection-like matrix mapping input audio frame to next
        # output audio frame
        self.direct_matrix = nn.Parameter(
            torch.zeros(input_dim, input_dim, dtype=torch.complex64 if complex else torch.float32).uniform_(-0.01, 0.01)
        )

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)


        # proj = control @ self.proj
        if self.complex:
            control = torch.fft.rfft(control, dim=-1)

        # print(control.shape, self.proj.shape)

        proj = project_and_limit_norm(control, self.proj)
        assert proj.shape == (batch, frames, self.input_dim)

        results = []
        state_vec = torch.zeros(
            batch,
            self.state_matrix_dim,
            device=control.device,
            dtype=torch.complex64 if self.complex else torch.float32)

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

        if self.complex:
            print(result.shape)
            result = torch.fft.irfft(result, dim=-1)
            print(result.shape)

        result = overlap_add(result, apply_window=windowing)
        return result[..., :n_samples]


class OverfitControlPlane(nn.Module):
    """
    Encapsulates parameters for control signal and state-space model
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, n_samples: int, complex: bool = False):
        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim, complex)
        self.n_samples = n_samples
        self.n_frames = int(n_samples / (input_dim // 2))
        self.control_plane_dim = control_plane_dim
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
        # cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=p)
        # audio = self.forward(sig=cp)
        # return max_norm(audio)

        indices = torch.randperm(self.control_plane_dim)
        cp = self.control_signal[:, indices, :]
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
    return torch.abs(c).sum() * 1e-2


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
        n_samples=n_samples,
        complex=is_complex
    )
    model = model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def train_and_monitor():
    target = get_one_audio_segment(n_samples=n_samples, samplerate=samplerate)
    collection = LmdbCollection(path='ssmcompression')

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

        optim = Adam(model.parameters(), lr=1e-2)

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

            # clip_grad_value_(model.parameters(), 0.5)

            optim.step()
            print(iteration, loss.item(), sparsity)

            with torch.no_grad():
                rnd = model.random()
                random_audio(rnd)

    train(target)


if __name__ == '__main__':
    train_and_monitor()
