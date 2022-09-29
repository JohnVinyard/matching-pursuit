import torch
from torch.nn import functional as F
from functools import reduce
import numpy as np

def fft_convolve(*args):
    n_samples = args[0].shape[-1]

    # pad to avoid wraparound artifacts
    padded = [F.pad(x, (0, x.shape[-1])) for x in args]
    
    specs = [torch.fft.rfft(x, dim=-1, norm='ortho') for x in padded]
    spec = reduce(lambda accum, current: accum * current, specs[1:], specs[0])
    final = torch.fft.irfft(spec, dim=-1, norm='ortho')

    # remove padding
    return final[..., :n_samples]


def fft_shift(a, shift):
    n_samples = a.shape[-1]
    shift_samples = shift * n_samples
    a = F.pad(a, (0, n_samples))
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs) * 2j * np.pi) / n_coeffs
    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    return samples
