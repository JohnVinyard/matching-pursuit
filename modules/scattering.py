from curses import window
import torch
import numpy as np
from torch.nn import functional as F
from torch.jit._script import ScriptModule, script_method

from fft_basis import morlet_filter_bank
from torch import nn
import zounds

from scratch import filter_bank


class MoreCorrectScattering(ScriptModule):
    def __init__(
            self,
            samplerate,
            scale,
            kernel_size,
            scaling_factors=0.1,
            non_linearity=torch.abs):

        super().__init__()
        filter_bank = zounds.learn.FilterBank(
            samplerate,
            kernel_size,
            scale,
            scaling_factors,
            normalize_filters=True,
            a_weighting=False)

        self.filter_bank = filter_bank.filter_bank
        self.window_size = kernel_size
        self.step_size = kernel_size // 2
        self.scale = scale
        self.non_linearity = non_linearity
        self.n_bands = len(scale)

    @script_method
    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, 1, -1)
        n_samples = x.shape[-1]
        n_frames = n_samples // self.step_size

        # convolve the signal with our filterbank and apply a non-linearity (abs())
        orig_spec = F.conv1d(
            x, self.filter_bank, padding=self.step_size)[..., :n_samples]
        orig_spec = self.non_linearity(orig_spec)

        # smooth the spectrogram at the original sampling rate
        avg = F.avg_pool1d(orig_spec, self.window_size, 1, padding=self.step_size)[..., :n_samples]

        first_order = F.avg_pool1d(orig_spec, self.window_size, self.step_size, padding=self.step_size)[..., :n_frames]

        # remove the local averages
        fine = orig_spec - avg
        fine = fine.view(batch, self.n_bands, n_samples)

        # apply each filter lower than the center frequency to each channel
        output = []
        for i in range(2, self.n_bands):
            bank = self.filter_bank[:i, ...]
            signal = fine[:, i:i + 1, :]
            spec = F.conv1d(signal, bank, padding=self.step_size)[..., :n_samples]
            spec = self.non_linearity(spec)
            spec = F.avg_pool1d(spec, self.window_size, self.step_size, padding=self.step_size)[..., :n_frames]
            output.append(spec)
        
        second_order = torch.cat(output, dim=1)

        full = torch.cat([first_order, second_order], dim=1)
        return full


def batch_fft_convolve(signal, d):
    signal_size = signal.shape[1]
    atom_size = d.shape[1]
    diff = signal_size - atom_size

    half_width = atom_size // 2

    px = F.pad(signal, (half_width, half_width))
    py = F.pad(d, (0, px.shape[1] - atom_size))

    # signal
    fpx = torch.fft.rfft(px, dim=-1, norm='ortho')[:, None, ...]

    # filter bank
    fpy = torch.fft.rfft(
        torch.flip(py, dims=(-1,)),
        dim=-1)[None, ...]

    c = fpx * fpy

    c = torch.fft.irfft(c, dim=-1, norm='ortho')

    return c[..., :signal.shape[-1]]


def scattering_transform(signal, d, window_size=512, step_size=256):
    n_filters = d.shape[0]

    batch, samples = signal.shape

    s1 = torch.abs(batch_fft_convolve(signal, d))
    s1 = s1.view(batch, -1, samples)
    pooled = F.avg_pool1d(s1, kernel_size=window_size,
                          stride=1, padding=step_size)[..., :samples]

    c1 = F.avg_pool1d(pooled, kernel_size=step_size,
                      stride=step_size, padding=step_size // 2)

    s2 = s1 - pooled

    s2 = s2.view(-1, samples)

    s2 = torch.abs(batch_fft_convolve(s2, d))
    s2 = s2.view(batch, -1, samples)
    c2 = F.avg_pool1d(s2, kernel_size=window_size,
                      stride=step_size, padding=step_size)

    return c1, c2


class ScatteringTransform(ScriptModule):
    def __init__(self, signal_size, scale, samplerate, kernel_size=512):
        super().__init__()
        bank = torch.from_numpy(
            morlet_filter_bank(samplerate, kernel_size, scale, 0.1).real)
        py = F.pad(bank, (0, signal_size))
        # filter bank
        fpy = torch.fft.rfft(
            torch.flip(py, dims=(-1,)),
            dim=-1)[None, ...]
        self.register_buffer('fpy', fpy)
        self.kernel_size = kernel_size
        self.signal_size = signal_size

    @script_method
    def _inner_forward(self, signal):
        # signal_size = signal.shape[1]
        atom_size = self.kernel_size
        # diff = self.signal_size - self.kernel_size

        half_width = atom_size // 2

        px = F.pad(signal, (half_width, half_width))

        # signal
        fpx = torch.fft.rfft(px, dim=-1, norm='ortho')[:, None, ...]

        c = fpx * self.fpy

        c = torch.fft.irfft(c, dim=-1, norm='ortho')

        return c[..., :signal.shape[-1]]

    @script_method
    def forward(self, signal):

        batch, samples = signal.shape

        s1 = torch.abs(self._inner_forward(signal))
        s1 = s1.view(batch, -1, samples)
        step_size = self.kernel_size // 2

        pooled = F.avg_pool1d(s1, kernel_size=self.kernel_size,
                              stride=1, padding=step_size)[..., :samples]

        c1 = F.avg_pool1d(pooled, kernel_size=step_size,
                          stride=step_size, padding=step_size // 2)

        s2 = s1 - pooled

        s2 = s2.view(-1, samples)

        s2 = torch.abs(self._inner_forward(s2))
        s2 = s2.view(batch, -1, samples)
        c2 = F.avg_pool1d(s2, kernel_size=self.kernel_size,
                          stride=step_size, padding=step_size)

        return c1, c2
