from subprocess import Popen, PIPE

import torch
from attr import dataclass
from matplotlib import pyplot as plt
from soundfile import SoundFile
from torch import nn
from torch.nn import functional as F
from typing import Union, Tuple
import numpy as np
from io import BytesIO
from time import time

from modules.decompose import fft_resample
from modules.reds import interpolate_last_axis
from modules.transfer import fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import device
from util.overfit import overfit_model


def encode_audio(
        x: Union[torch.Tensor, np.ndarray],
        samplerate: int = 22050,
        format='WAV',
        subtype='PCM_16') -> bytes:
    if isinstance(x, torch.Tensor):
        x = x.data.cpu().numpy()

    if x.ndim > 1:
        x = x[0]

    x = x.reshape((-1,))
    io = BytesIO()

    with SoundFile(
            file=io,
            mode='w',
            samplerate=samplerate,
            channels=1,
            format=format,
            subtype=subtype) as sf:
        sf.write(x)

    io.seek(0)
    return io.read()


def listen_to_sound(samples: bytes, wait_for_user_input: bool = True) -> None:
    proc = Popen(f'aplay', shell=True, stdin=PIPE)
    print('PROC', proc.stdin)
    if proc.stdin is not None:
        proc.stdin.write(samples)
        proc.communicate()

    if wait_for_user_input:
        input('Next')


def stft(
        x: torch.Tensor,
        ws: int = 2048,
        step: int = 256,
        pad: bool = True):
    frames = x.shape[-1] // step

    if pad:
        x = F.pad(x, (0, ws))

    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.abs(torch.fft.rfft(x, norm='ortho'))

    x = x[:, :, :frames, :]
    return x


@torch.jit.script
def sim(
        home: torch.Tensor,
        tensions: torch.Tensor,
        masses: torch.Tensor,
        damping: float,
        gains: torch.Tensor,
        mics: torch.Tensor,
        forces: torch.Tensor,
        home_modifier: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    batch, n_masses, dim, n_samples = forces.shape
    # batch, n_masses, dim, _ = mic.shape
    batch, n_masses, dim, n_samples = home_modifier.shape

    # NOTE: home is defined as zero for each node/mass, so this
    # could simply be home_modifier directly
    h = home + home_modifier

    position = torch.zeros(batch, n_masses, dim, 1, device=forces.device)
    velocity = torch.zeros(batch, n_masses, dim, 1, device=forces.device)

    recording = torch.zeros(batch, n_masses, n_samples, 1, device=forces.device)
    displacement = torch.zeros(batch, n_masses, dim, n_samples, device=forces.device)

    for i in range(n_samples):
        direction = h[..., i: i + 1] - position

        displacement[..., i: i + 1] = direction

        acc = forces[..., i: i + 1] + ((tensions * direction) / masses)
        velocity = velocity + acc
        velocity = velocity * damping
        position = position + velocity

        r = torch.tanh(velocity * gains)
        r = r.permute(0, 1, 3, 2) @ mics

        recording[..., i: i + 1, :] = r

    # recording = torch.einsum('abcd,abcd->acd', recording, layer_mics)
    # recording = recording.permute(0, 2, 1)

    return recording, displacement


def unit_norm(x: torch.Tensor, dim: int = -1, epsilon: float = 1e-8):
    n = torch.norm(x, dim=dim, keepdim=True)
    return x / (n + epsilon)



class BetterGooLayer(nn.Module):
    def __init__(
            self,
            n_samples: int,
            n_filters: int,
            dimension: int,
            n_masses: int,
            damping: float = 0.9998):

        super().__init__()
        self.n_samples = n_samples
        self.n_filters = n_filters
        self.dimension = dimension
        self.n_masses = n_masses
        self.damping = damping

        home = torch.zeros(1, 1, dimension, 1)
        self.register_buffer('home', home)

        self.masses = nn.Parameter(torch.zeros(1, n_masses, 1, 1).uniform_(1000, 10000))
        self.tensions = nn.Parameter(torch.zeros(1, n_masses, dimension, 1).uniform_(20, 30))
        self.gains = nn.Parameter(torch.zeros(1, n_masses, 1, 1).uniform_(10, 100))
        self.mics = nn.Parameter(torch.zeros(1, n_masses, dimension, 1).uniform_(-1, 1))
        self.to_filter_mixture = nn.Parameter(torch.zeros(self.dimension, self.n_filters).uniform_(-1, 1))

    def forward(
            self,
            forces: torch.Tensor,
            home_modifier: torch.Tensor,
            filters: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch = forces.shape[0]

        # run physical simulation
        recording, displacement = sim(
            self.home,
            self.tensions,
            self.masses,
            self.damping,
            self.gains,
            self.mics,
            forces,
            home_modifier)

        # compute the filture mixture over time
        mixture = torch.einsum('abcd,ce->abed', displacement, self.to_filter_mixture)
        mixture = interpolate_last_axis(mixture, self.n_samples)

        # upsample the *envelope* of the recording and multiply with noise
        upsampled = fft_resample(
            recording.view(batch, self.n_masses, -1), desired_size=self.n_samples, is_lowest_band=True)
        upsampled  = torch.abs(upsampled) * torch.zeros_like(upsampled).uniform_(-1, 1)

        # convolve noise with the filters
        f = ensure_last_axis_length(filters, self.n_samples)[:, None, :, :]
        f = unit_norm(f)
        filtered = fft_convolve(f, upsampled[:, :, None, :])


        hf = torch.einsum('abcd,abcd->abd', mixture, filtered) + upsampled


        return recording, displacement, hf





class GooStack(nn.Module):

    def __init__(
            self, n_samples: int,
            n_filters: int,
            dimension: int,
            n_masses: int,
            n_layers: int,
            damping: float = 0.9998):

        super().__init__()
        self.n_samples = n_samples
        self.dimension = dimension
        self.n_masses = n_masses
        self.n_layers = n_layers
        self.n_filters = n_filters

        self.network = nn.ModuleList([
            BetterGooLayer(n_samples, n_filters,dimension, n_masses, damping) for _ in range(n_layers)])

        self.mix = nn.Parameter(torch.zeros(n_layers).uniform_(-1, 1))

    def forward(self, filters: torch.Tensor, forces: torch.Tensor, home_modifier: torch.Tensor) -> torch.Tensor:
        recordings = []

        for layer in self.network:
            r, h, hf = layer(forces, home_modifier, filters)
            r = fft_resample(r, self.n_samples, is_lowest_band=True) + hf
            recordings.append(r)
            home_modifier = h

        recordings = torch.stack(recordings, dim=-1)
        mx = torch.softmax(self.mix, dim=-1)

        r = recordings @ mx
        return r


class GooPerformance(nn.Module):

    def __init__(
            self,
            dimension: int,
            n_masses: int,
            damping: float = 0.9998,
            n_layers: int = 3,
            n_samples: int = 2 ** 16,
            simulation_block_size: int = 4,
            n_filters: int = 6,
            filter_size: int = 256):

        super().__init__()
        self.dimension = dimension
        self.n_masses = n_masses
        self.damping = damping
        self.n_filters = n_filters
        self.filter_size = filter_size

        self.n_layers = n_layers
        self.n_samples = n_samples
        self.simulation_block_size = simulation_block_size

        self.n_simulation_steps = n_samples // simulation_block_size

        forces = torch.zeros(1, n_masses, dimension, self.n_simulation_steps // 32).bernoulli_(p=1e-4)
        forces = forces * torch.zeros_like(forces).uniform_(-0.01, 0.01)
        self.forces = nn.Parameter(forces)

        home_modifier = torch.zeros(1, n_masses, dimension, self.n_simulation_steps)
        self.register_buffer('home_modifier', home_modifier)

        self.network = GooStack(n_samples, n_filters, dimension, n_masses, n_layers, damping)

        self.hf_filters = nn.Parameter(torch.zeros(1, self.n_filters, self.filter_size).uniform_(-0.001, 0.001))



    def forward(self):
        f = upsample_with_holes(self.forces, self.n_simulation_steps)
        recording = self.network.forward(self.hf_filters, f, self.home_modifier)
        recording = recording / torch.abs(recording.max()) + 1e-8
        return recording


def loss_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = stft(a, 2048, 256, pad=True)
    b = stft(b, 2048, 256, pad=True)
    return torch.abs(a - b).sum()


def exercise_goo_stack(device: torch.device = None):
    n_samples = 2 ** 18
    simulation_block_size = 4
    n_simulation_steps = n_samples // simulation_block_size

    n_masses = 8
    dim = 8
    batch_size = 1
    n_layers = 3

    forces = torch.zeros(batch_size, n_masses, dim, n_simulation_steps, device=device).bernoulli_(p=1e-6)
    forces = forces * torch.zeros_like(forces, device=device).uniform_(-0.01, 0.01)

    home_modifier = torch.zeros(batch_size, n_masses, dim, n_simulation_steps, device=device)
    network = GooStack(dim, n_masses, n_layers=n_layers, damping=0.9993).to(device)
    recording = network.forward(forces, home_modifier)

    recording = fft_resample(recording, n_samples, is_lowest_band=True)
    recording = recording / torch.abs(recording.max()) + 1e-8

    # layer = GooLayer(dimension=dim, n_masses=n_masses)

    # recording, displacement = layer.forward(forces, home_modifier)

    return recording


# def exercise_goo_layer():
#     n_samples = 2 ** 17
#     n_masses = 8
#     dim = 16
#     batch_size = 1
#
#     forces = torch.zeros(batch_size, n_masses, dim, n_samples).bernoulli_(p=1e-6)
#     forces = forces * torch.zeros_like(forces).uniform_(-0.01, 0.01)
#     home_modifier = torch.zeros(batch_size, n_masses, dim, n_samples)
#     layer = GooLayer(dimension=dim, n_masses=n_masses)
#
#     recording, displacement = layer.forward(forces, home_modifier)
#
#     return recording


# def goo():
#     dimension = 3
#     home = torch.zeros(dimension)
#     mass = torch.zeros(1).uniform_(1000, 10000)
#     tension = torch.zeros(dimension).uniform_(20, 30)
#     position = torch.zeros(dimension)
#
#     n_samples = 2 ** 18
#
#     forces = torch.zeros(n_samples, dimension)
#     forces[8192, :] = torch.zeros(dimension).uniform_(-0.02, 0.02)
#     forces[8192 * 5, :] = torch.zeros(dimension).uniform_(-0.03, 0.03)
#
#     mic = unit_norm(torch.zeros(dimension).uniform_(-1, 1))
#     recording = torch.zeros(n_samples, 1)
#     damping = 0.9998
#     velocity = torch.zeros(dimension)
#
#     gain = torch.zeros(dimension).uniform_(10, 1000)
#
#     # gain = torch.zeros(dimension).uniform_(0.01, 0.1)
#
#     for i in range(n_samples):
#         direction = home - position
#         acc = forces[i, :] + ((tension * direction) / mass)
#         velocity += acc
#         velocity *= damping
#         position += velocity
#
#         # TODO: This needs to be applied as part of the simulation
#         recording[i] = torch.tanh(velocity * gain) @ mic
#
#     return recording


def generate():
    start = time()

    with torch.no_grad():
        x = exercise_goo_stack(device=None)

    seconds = x.numel() / 22050
    stop = time()
    print(f'Generated {seconds:.2f} seconds of audio in {(stop - start):.2f} seconds')
    listen_to_sound(encode_audio(x[0]), wait_for_user_input=True)


def check_model():
    n_samples = 2 ** 16

    device = torch.device('cuda')

    model = GooPerformance(
        dimension=4, n_masses=8, n_layers=2, n_samples=2 ** 16, simulation_block_size=8).to(device)
    overfit_model(
        n_samples=n_samples,
        model=model,
        loss_func=loss_func,
        collection_name='goo',
        learning_rate=1e-3,
        device=device,
    )


if __name__ == '__main__':
    check_model()
