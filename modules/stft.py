import torch
from torch.nn import functional as F
import numpy as np
from scipy.signal import morlet
import zounds

def stft(x, ws=512, step=256, pad=False, log_amplitude=False, log_epsilon=1e-4):
    if pad:
        x = F.pad(x, (0, step))
    
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = torch.abs(x)

    if log_amplitude:
        x = torch.log(x + log_epsilon)
    
    return x


def stft_relative_phase(x, ws=512, step=256, pad=False):

    if pad:
        x = F.pad(x, (0, step))

    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = x.view(x.shape[0], -1, ws // 2 + 1)
    
    # get the magnitude
    mag = torch.abs(x)

    # compute the angle in radians
    phase = torch.angle(x)

    # get instantaneous frequency.  We should now be absolute phase
    # agnostic, while still emphasizing periodicity
    padding = torch.zeros(phase.shape[1]).to(x.device)[None, :, None]
    phase = torch.diff(phase, axis=-1, prepend=padding)

    return mag, phase


def log_stft(x, ws=512, step=256, a=0.001):
    x = stft(x, ws, step)
    return torch.log(a + x)


def morlet_filter_bank(
        samplerate,
        kernel_size,
        scale,
        scaling_factor,
        normalize=True):
    
    basis_size = len(scale)
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    try:
        if len(scaling_factor) != len(scale):
            raise ValueError('scaling factor must have same length as scale')
    except TypeError:
        scaling_factor = np.repeat(float(scaling_factor), len(scale))

    sr = int(samplerate)

    for i, band in enumerate(scale):
        scaling = scaling_factor[i]
        w = band.center_frequency / (scaling * 2 * sr / kernel_size)
        basis[i] = morlet(
            M=kernel_size,
            w=w,
            s=scaling)

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8

    return basis


def geom_basis(fft_size, sr):
    return morlet_filter_bank(
        sr, 
        fft_size, 
        zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), fft_size), 
        np.linspace(0.05, 0.99, fft_size))


def short_time_transform(x, basis, pad=True):

    ws = basis.shape[1]
    ss = ws // 2

    if pad:
        x = F.pad(x, (0, ss))
    
    windowed = x.unfold(-1, ws, ss)

    windowed = windowed * torch.hamming_window(ws)[None, None, None, :].to(x.device)

    freq_domain = windowed @ basis.T

    return freq_domain[..., :(ws // 2 + 1)]