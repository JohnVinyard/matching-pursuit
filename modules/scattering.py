import torch
import numpy as np
from torch.nn import functional as F


def batch_fft_convolve(signal, d):
    signal_size = signal.shape[1]
    atom_size = d.shape[1]
    diff = signal_size - atom_size

    half_width = atom_size // 2

    px = F.pad(signal, (half_width, half_width))
    py = F.pad(d, (0, px.shape[1] - atom_size))


    fpx = torch.fft.rfft(px, dim=-1, norm='ortho')[:, None, ...]

    fpy = torch.fft.rfft(
        torch.flip(py, dims=(-1,)),
        dim=-1)[None, ...]

    c = fpx * fpy

    new_size = (c.shape[-2] - 1) * 2
    c = torch.fft.irfft(c, dim=-1, norm='ortho')

    return c[..., :signal.shape[-1]]


def scattering_transform(signal, d, window_size=512, step_size=256):
    batch, samples = signal.shape

    s1 = torch.abs(batch_fft_convolve(signal, d))
    s1 = s1.view(batch, -1, samples)
    pooled = F.avg_pool1d(s1, kernel_size=window_size, stride=1, padding=step_size)[..., :samples]
    
    c1 = F.avg_pool1d(pooled, kernel_size=step_size, stride=step_size, padding=step_size // 2)

    s2 = s1 - pooled

    s2 = s2.view(-1, samples)
    s2 = torch.abs(batch_fft_convolve(s2, d))
    s2 = s2.view(batch, -1, samples)
    c2 = F.avg_pool1d(s2, kernel_size=window_size, stride=step_size, padding=step_size)

    return c1, c2


