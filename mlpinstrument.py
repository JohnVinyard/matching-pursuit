from typing import Tuple

import torch
import numpy as np
from torch import nn
from matplotlib import pyplot as plt

from modules import max_norm
from modules.transfer import fft_convolve
from util.playable import listen_to_sound, playable
import zounds


def run_layer(
        control_plane: torch.Tensor,
        mapping: torch.Tensor,
        decays: torch.Tensor,
        out_mapping: torch.Tensor,
        audio_mapping: torch.Tensor,
        gains: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, control_plane_dim, frames = control_plane.shape

    x = mapping @ control_plane
    orig = x
    decays = decays.view(batch_size, control_plane_dim, 1).repeat(1, 1, frames)
    decays = decays.cumprod(dim=-1)
    # decays = torch.flip(decays, dims=[-1])
    x = fft_convolve(x, decays)
    x = (out_mapping @ x) + orig

    cp = torch.tanh(x * gains.view(batch_size, control_plane_dim, 1))

    audio = audio_mapping @ cp

    # TODO: This should be mapped to audio outside of this layer, probably
    # each layer by a single mapping network
    audio = audio.permute(0, 2, 1).reshape(batch_size, 1, -1)
    return audio, cp


class Block(nn.Module):
    def __init__(
            self,
            block_size,
            base_resonance: float = 0.5,
            max_gain: float = 5):
        super().__init__()
        self.block_size = block_size
        self.base_resonance = base_resonance
        self.resonance_span = 1 - base_resonance
        self.max_gain = max_gain

        self.w1 = torch.zeros(block_size, block_size).uniform_(-1, 1)
        self.w2 = torch.zeros(block_size, block_size).uniform_(-1, 1)
        self.audio = torch.zeros(block_size, block_size).uniform_(-1, 1)

        self.decays = torch.zeros(block_size).uniform_(0.001, 0.99)
        self.gains = torch.zeros(block_size).uniform_(0, 1)

    def forward(self, cp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, cp = run_layer(
            cp,
            self.w1,
            self.base_resonance + torch.sigmoid(self.decays) * self.resonance_span,
            self.s2,
            self.audio,
            torch.sigmoid(self.gains) * self.max_gain)
        return output, cp


class Stack(nn.Module):
    def __init__(
            self,
            n_blocks,
            block_size,
            base_resonance:
            float = 0.5,
            max_gain: float = 5):
        super().__init__()

        self.mix = nn.Parameter(torch.zeros(n_blocks).uniform_(-1, 1))
        self.blocks = nn.ModuleList([
            Block(
                block_size,
                base_resonance,
                max_gain
            ) for _ in range(n_blocks)
        ])

    def forward(self, cp):
        batch_size, channels, frames = cp.shape

        working_control_plane = cp

        total_samples = frames * self.block_size

        channels = torch.zeros(
            batch_size, self.n_blocks, total_samples, device=cp.device)

        for i, block in enumerate(self.blocks):
            working_control_plane, output = block(working_control_plane)
            channels[:, i: i + 1, :] = output

        mix = torch.softmax(self.mix, dim=-1)
        mixed = channels.permute(0, 2, 1) @ mix
        mixed = mixed.view(batch_size, 1, total_samples)
        return mixed


if __name__ == '__main__':
    n_samples = 2 ** 18
    block_size = 128
    n_frames = n_samples // block_size
    control_plane_dim = block_size

    control_plane = torch.zeros(1, control_plane_dim, n_frames).bernoulli_(p=0.0001)
    w1 = torch.zeros(control_plane_dim, control_plane_dim).uniform_(-1, 1)
    w2 = torch.zeros(control_plane_dim, control_plane_dim).uniform_(-1, 1)
    w3 = torch.zeros(control_plane_dim, control_plane_dim).uniform_(-1, 1)

    decays = torch.zeros(control_plane_dim).uniform_(0.001, 0.99)
    gains = torch.zeros(control_plane_dim).uniform_(0.1, 10)

    output, cp = run_layer(control_plane, w1, decays, w2, w3, gains)
    print(output.shape, cp.shape)
    output = playable(output, 22050, normalize=True)
    listen_to_sound(output)
