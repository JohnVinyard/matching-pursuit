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
        gains: torch.Tensor) -> torch.Tensor:

    batch_size, control_plane_dim, frames = control_plane.shape

    x = mapping @ control_plane
    orig = x
    decays = decays.view(batch_size, control_plane_dim, 1).repeat(1, 1, frames)
    decays = decays.cumprod(dim=-1)
    # decays = torch.flip(decays, dims=[-1])
    x = fft_convolve(x, decays)
    x = (out_mapping @ x) + orig
    x = torch.tanh(x * gains.view(batch_size, control_plane_dim, 1))
    x = x.permute(0, 2, 1).reshape(batch_size, 1, -1)
    return x

if __name__ == '__main__':
    n_samples = 2 ** 18
    block_size = 128
    n_frames = n_samples // block_size
    control_plane_dim = block_size

    control_plane = torch.zeros(1, control_plane_dim, n_frames).bernoulli_(p=0.0001)
    w1 = torch.zeros(control_plane_dim, control_plane_dim).uniform_(-1, 1)
    w2 = torch.zeros(control_plane_dim, control_plane_dim).uniform_(-1, 1)
    decays = torch.zeros(control_plane_dim).uniform_(0.001, 0.99)
    gains = torch.zeros(control_plane_dim).uniform_(0.1, 10)


    output = run_layer(control_plane, w1, decays, w2, gains)
    output = playable(output, 22050, normalize=True)
    listen_to_sound(output)
