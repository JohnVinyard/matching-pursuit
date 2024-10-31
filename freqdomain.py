from typing import Union

import torch
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio
from util import playable
from util.playable import listen_to_sound
from torch import nn


class Block(nn.Module):

    def __init__(
            self,
            window_size: int,
            control_plane_dim: int,
            transfer: torch.Tensor,
            gain: Union[float, torch.Tensor]):

        super().__init__()
        self.window_size = window_size
        self.n_coeffs = self.window_size // 2 + 1
        self.control_plane_dim = control_plane_dim

        self.transfer = nn.Parameter(torch.zeros((self.control_plane_dim, self.n_coeffs,)))
        self.mixer_matrix = nn.Parameter(torch.eye(self.control_plane_dim))

        self.transfer.data[:] = transfer

        self.gain = nn.Parameter(torch.ones((1,)).fill_(gain))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = (x.permute(0, 2, 1) @ self.mixer_matrix).permute(0, 2, 1)

        batch, channels, time = x.shape

        assert channels == self.control_plane_dim

        windowed = windowed_audio(x, self.window_size, self.window_size // 2)
        print('windowed', windowed.shape)

        spec = torch.fft.rfft(windowed, dim=-1)
        print('windowed spec', spec.shape)

        print('transfer', self.transfer.shape)

        batch, channels, frames, coeffs = spec.shape

        output_frames = []

        for i in range(frames):
            current = spec[:, :, i: i + 1, :]

            if i > 0:
                current = current + output_frames[i - 1]

            # TODO: transform the transform function given a deformation matrix
            filtered = current * self.transfer[None, :, None, :]

            output_frames.append(filtered)

        output = torch.cat(output_frames, dim=2)
        audio_windows = torch.fft.irfft(output, dim=-1)
        samples = overlap_add(audio_windows, apply_window=True)
        samples = samples * self.gain
        samples = torch.tanh(samples)
        x = samples[..., :time]
        return x


class AudioNetwork(nn.Module):
    def __init__(self, control_plane_dim: int, window_size: int, n_blocks: int):
        super().__init__()
        self.window_size = window_size
        self.n_blocks = n_blocks
        self.mixer = nn.Parameter(torch.zeros((n_blocks + 1,)))
        self.control_plane_dim = control_plane_dim
        self.n_coeffs = window_size // 2 + 1

        self.blocks = nn.ModuleList([
            Block(window_size, control_plane_dim, self.init_transfer(), torch.zeros(1).uniform_(1, 50).item())
            for _ in range(self.n_blocks)
        ])

    def init_transfer(self):
        resonances = torch.zeros(self.control_plane_dim, self.n_coeffs).uniform_(0.9, 0.9998)
        sparse = torch.zeros_like(resonances).bernoulli_(p=0.01)
        resonances = resonances * sparse
        scaling = torch.linspace(1, 0, self.n_coeffs) ** 2
        scaled_resonances = resonances * scaling[None, :]
        return scaled_resonances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = [x[..., None]]
        inp = x

        for block in self.blocks:
            inp = block(inp)
            outputs.append(inp[..., None])

        result = torch.cat(outputs, dim=-1)
        mixer_values = torch.softmax(self.mixer, dim=-1)
        mixed = (result * mixer_values[None, None, None, :]).sum(dim=-1)
        mixed = torch.sum(mixed, dim=1, keepdim=True)
        return mixed


if __name__ == '__main__':
    batch_size = 1
    window_size = 2048
    n_frames = 128
    control_plane_dim = 16

    impulse_size = 16
    impulse = torch.zeros(impulse_size).uniform_(-1, 1)
    impulse = impulse * torch.hann_window(impulse_size)

    inp = torch.zeros(1, control_plane_dim, 2 ** 18)

    inp[:, 0, 1024: 1024 + impulse_size] += impulse * 0.1
    inp[:, 1, 8192: 8192 + impulse_size] += impulse * 0.5
    inp[:, 2, 16384: 16384 + impulse_size] += impulse

    network = AudioNetwork(control_plane_dim, window_size, n_blocks=3)
    samples = network.forward(inp)

    print(samples.min(), samples.max())
    samples = playable(samples, samplerate=22050, normalize=True)
    listen_to_sound(samples)
