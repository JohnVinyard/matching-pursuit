# the size, in samples of the audio segment we'll overfit
from ssm import n_active_sites

n_samples = 2 ** 18

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

# the size of each, half-lapped audio "frame"
window_size = 128

# the dimensionality of the control plane or control signal
control_plane_dim = 64

# the dimensionality of the state vector, or hidden state for the RNN
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
from modules import max_norm, flattened_multiband_spectrogram, sparsify, sparse_softmax
from modules.infoloss import CorrelationLoss
from util import device, encode_audio, make_initializer
from torch.nn import functional as F


init_weights = make_initializer(0.05)


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

        self.net = nn.RNN(
            input_size=input_dim,
            hidden_size=state_matrix_dim,
            num_layers=1,
            nonlinearity='tanh',
            bias=False,
            batch_first=True)

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        self.out_proj = nn.Linear(state_dim, window_size, bias=False)

        self.apply(init_weights)

    def forward(self, control: torch.Tensor) -> torch.Tensor:
        """
        (batch, control_plane, time) -> (batch, window_size, time)
        """

        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)

        # try to ensure that the input signal only includes low-frequency info
        proj = control @ self.proj
        # proj = F.interpolate(proj, size=self.input_dim, mode='linear')
        # proj = proj * torch.zeros_like(proj).uniform_(-1, 1)
        # proj = proj * torch.hann_window(self.input_dim, device=proj.device)

        assert proj.shape == (batch, frames, self.input_dim)
        result, hidden = self.net.forward(proj)

        result = self.out_proj(result)

        result = result.view(batch, 1, -1)
        result = torch.sin(result)
        return result


class OverfitControlPlane(nn.Module):
    """
    Encapsulates parameters for control signal and state-space model
    """

    def __init__(
            self,
            control_plane_dim: int,
            input_dim: int,
            state_matrix_dim: int,
            n_samples: int,
            n_events: int):

        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim)
        self.n_samples = n_samples
        self.n_frames = n_samples // input_dim
        self.n_events = n_events

        self.control = nn.Parameter(
            torch.zeros(1, control_plane_dim, self.n_frames).uniform_(0, 0.1))


    @property
    def control_signal_display(self) -> np.ndarray:
        return self.control_signal.data.cpu().numpy().reshape((-1, self.n_frames))

    @property
    def control_signal(self) -> torch.Tensor:
        s = sparsify(self.control, n_to_keep=256)
        return torch.relu(s)

    # TODO: probability should scale with time _only_, so control plane size does not matter
    def random(self, p=0.0001):
        """
        Produces a random, sparse control signal, emulating short, transient bursts
        of energy into the system modelled by the `SSM`
        """
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=p)
        audio = self.forward(sig=cp)
        return max_norm(audio)

    def rolled_control_plane(self):
        indices = torch.randperm(control_plane_dim)
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
        n_events=n_active_sites
    )
    model = model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()

    forward = mask
    backward = x

    y = backward + (forward - backward).detach()

    return y.sum()


# def sparsity_loss(x: torch.Tensor) -> torch.Tensor:
#     return l0_norm(x)


def train_and_monitor():
    target = get_one_audio_segment(n_samples=n_samples, samplerate=samplerate)
    collection = LmdbCollection(path='ssm')

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random'],
        'audio/wav',
        encode_audio,
        collection)

    envelopes, = loggers(
        ['envelopes'],
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
    ], port=9999, n_workers=1)

    loss_model = CorrelationLoss().to(device)

    def train(target: torch.Tensor):
        model = construct_experiment_model()

        optim = Adam(model.parameters(), lr=1e-3)

        for iteration in count():
            optim.zero_grad()
            recon = model.forward()
            recon_audio(max_norm(recon))
            loss = \
                reconstruction_loss(recon, target) \
                + loss_model.multiband_noise_loss(target, recon, window_size=32, step=16) \
                # + sparsity_loss(model.control_signal)

            envelopes(model.control_signal.view(control_plane_dim, -1) / (model.control_signal.max() + 1e-8))
            loss.backward()

            active = (model.control_signal > 0).sum().item()

            clip_grad_value_(model.parameters(), 0.5)

            optim.step()
            print(iteration, loss.item(), active)

            with torch.no_grad():
                rnd = model.random()
                random_audio(rnd)

    train(target)


if __name__ == '__main__':
    train_and_monitor()
