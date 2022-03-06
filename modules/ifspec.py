import torch
from torch.nn import functional as F
from modules.ddsp import overlap_add
import numpy as np


def to_spectrogram(audio_batch, window_size, step_size, samplerate):
    batch_size = audio_batch.shape[0]

    audio_batch = F.pad(audio_batch, (0, step_size))
    windowed = audio_batch.unfold(-1, window_size, step_size)
    window = torch.hann_window(window_size).to(audio_batch.device)
    spec = torch.fft.rfft(windowed * window, dim=-1, norm='ortho')
    n_coeffs = (window_size // 2) + 1
    spec = spec.reshape(batch_size, -1, n_coeffs)

    mag = torch.abs(spec) + 1e-12
    phase = torch.angle(spec)
    phase = torch.diff(
        phase, 
        dim=1, 
        prepend=torch.zeros(batch_size, 1, n_coeffs).to(audio_batch.device))
    
    return torch.cat([mag[..., None], phase[..., None]], dim=-1)


def from_spectrogram(spec, window_size, step_size, samplerate):
    print(spec.shape)
    batch_size, time, n_coeffs, _ = spec.shape
    mag = spec[..., 0]
    phase = spec[..., 1]

    real = mag
    imag = torch.cumsum(phase, dim=1)
    imag = (imag + np.pi) % (2 * np.pi) - np.pi

    # spec = torch.complex(real, imag)
    spec = real * torch.exp(1j * imag)
    windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
    signal = overlap_add(windowed[:, None, :, :], apply_window=False)
    return signal
