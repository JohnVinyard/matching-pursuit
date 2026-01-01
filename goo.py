from subprocess import Popen, PIPE

import torch
from matplotlib import pyplot as plt
from soundfile import SoundFile
from torch import nn
from torch.nn import functional as F
from typing import Union
import numpy as np
from io import BytesIO


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


def unit_norm(x: torch.Tensor, dim: int = -1, epsilon: float = 1e-8):
    n = torch.norm(x, dim=dim, keepdim=True)
    return x / (n + epsilon)


class GooLayer(nn.Module):

    def __init__(self, dimension: int, n_masses: int, damping: float = 0.9998):
        super().__init__()
        self.dimension = dimension
        self.n_masses = n_masses
        self.damping = damping

        home = torch.zeros(1, 1, dimension, 1)
        self.register_buffer('home', home)

        self.masses = nn.Parameter(torch.zeros(1, n_masses, 1, 1).uniform_(1000, 10000))
        self.tensions = nn.Parameter(torch.zeros(1, n_masses, dimension, 1).uniform_(20, 30))
        self.gains = nn.Parameter(torch.zeros(1, n_masses, 1, 1).uniform_(10, 100))

    def forward(
            self,
            forces: torch.Tensor,
            home_modifier: torch.Tensor,
            mic: torch.Tensor) -> torch.Tensor:
        batch, n_masses, dim, n_samples = forces.shape
        batch, n_masses, dim, _ = mic.shape
        batch, n_masses, dim, n_samples = home_modifier.shape

        # NOTE: home is defined as zero for each node/mass, so this
        # could simply be home_modifier directly
        h = self.home + home_modifier

        position = torch.zeros(batch, n_masses, dim, 1, device=forces.device)
        velocity = torch.zeros(batch, n_masses, dim, 1, device=forces.device)
        recording = torch.zeros(batch, n_masses, n_samples, 1, device=forces.device)

        for i in range(n_samples):
            direction = h[..., i: i + 1] - position
            acc = forces[..., i: i + 1] + ((self.tensions * direction) / self.masses)
            velocity += acc
            velocity *= self.damping
            position += velocity

            r = torch.tanh(velocity * self.gains)
            r = r.permute(0, 1, 3, 2) @ mic

            recording[..., i: i + 1, :] = r

        return recording


def exercise_goo_layer():
    n_samples = 2 ** 17
    n_masses = 8
    dim = 16
    batch_size = 1

    forces = torch.zeros(batch_size, n_masses, dim, n_samples).bernoulli_(p=0.00002)
    forces = forces * torch.zeros_like(forces).uniform_(-0.01, 0.01)

    home_modifier = torch.zeros(batch_size, n_masses, dim, n_samples)
    mics = torch.zeros(batch_size, n_masses, dim, 1).uniform_(-1, 1)

    layer_mics = torch.zeros(batch_size, n_masses, 1, 1).uniform_(-1, 1)

    layer = GooLayer(dimension=dim, n_masses=n_masses, damping=0.9991)

    recording = layer.forward(forces, home_modifier, mics)

    final = torch.einsum('abcd,abcd->acd', recording, layer_mics)

    return final.permute(0, 2, 1)


def goo():
    dimension = 3
    home = torch.zeros(dimension)
    mass = torch.zeros(1).uniform_(1000, 10000)
    tension = torch.zeros(dimension).uniform_(20, 30)
    position = torch.zeros(dimension)

    n_samples = 2 ** 18

    forces = torch.zeros(n_samples, dimension)
    forces[8192, :] = torch.zeros(dimension).uniform_(-0.02, 0.02)
    forces[8192 * 5, :] = torch.zeros(dimension).uniform_(-0.03, 0.03)

    mic = unit_norm(torch.zeros(dimension).uniform_(-1, 1))
    recording = torch.zeros(n_samples, 1)
    damping = 0.9998
    velocity = torch.zeros(dimension)

    gain = torch.zeros(dimension).uniform_(10, 1000)

    # gain = torch.zeros(dimension).uniform_(0.01, 0.1)

    for i in range(n_samples):
        direction = home - position
        acc = forces[i, :] + ((tension * direction) / mass)
        velocity += acc
        velocity *= damping
        position += velocity

        # TODO: This needs to be applied as part of the simulation
        recording[i] = torch.tanh(velocity * gain) @ mic

    return recording


if __name__ == '__main__':
    with torch.no_grad():
        x = exercise_goo_layer()
        # x /= torch.abs(x).max()
        plt.plot(x.view(-1))
        plt.show()

    # x = goo().view(-1)
    #
    # x /= torch.abs(x).max()
    # plt.plot(x)
    # plt.show()
    #

    listen_to_sound(encode_audio(x), wait_for_user_input=True)

    # x = stft(x.view(1, 1, -1))[0, 0, ...]
    #
    # plt.matshow(x.T)
    # plt.show()
    # plt.plot(x.data.cpu().numpy())
    # plt.show()
