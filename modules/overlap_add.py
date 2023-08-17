import torch
import numpy as np
from torch.nn import functional as F
from scipy.signal import hann

def _torch_overlap_add(x, apply_window=True, flip=False):
    batch, channels, frames, samples = x.shape

    if apply_window:
        window = torch.from_numpy(hann(samples, False)).to(x.device).float()
        # window = torch.hamming_window(samples, periodic=False).to(x.device)
        # window = torch.hann_window(samples, periodic=False).to(x.device)
        x = x * window[None, None, None, :]

    hop_size = samples // 2
    first_half = x[:, :, :, :hop_size].contiguous().view(batch, channels, -1)
    second_half = x[:, :, :, hop_size:].contiguous().view(batch, channels, -1)
    first_half = F.pad(first_half, (0, hop_size))
    second_half = F.pad(second_half, (hop_size, 0))

    if flip:
        first_half = first_half[:, :, ::-1]

    output = first_half + second_half
    return output


def _np_overlap_add(x, apply_window=True, flip=False):
    batch, channels, frames, samples = x.shape

    if apply_window:
        window = hann(samples)
        x = x * window[None, None, None, :]

    hop_size = samples // 2
    first_half = x[:, :, :, :hop_size].reshape((batch, channels, -1))
    second_half = x[:, :, :, hop_size:].reshape((batch, channels, -1))

    first_half = np.pad(first_half, [(0, 0), (0, 0), (0, hop_size)])
    second_half = np.pad(second_half, [(0, 0), (0, 0), (hop_size, 0)])

    if flip:
        first_half = first_half[:, :, ::-1]

    output = first_half + second_half
    return output


def overlap_add(x, apply_window=True, flip=False, trim=None):

    if isinstance(x, np.ndarray):
        result = _np_overlap_add(x, apply_window, flip)
    else:
        result = _torch_overlap_add(x, apply_window, flip)
    
    if trim is not None:
        result = result[..., :trim]
    
    return result
