from typing import Tuple
import zounds
from torch import Tensor, nn
import numpy as np
import torch
from torch.nn import functional as F

from modules.stft import morlet_filter_bank


class PerceptualAudioModel(nn.Module):
    def __init__(self, exp, norm_second_order: bool = True):
        super().__init__()
        self.exp = exp
        self.norm_second_order = norm_second_order

        scale = zounds.MelScale(zounds.FrequencyBand(
            1, exp.samplerate.nyquist), 128)

        orig_filters = filters = morlet_filter_bank(
            exp.samplerate, exp.kernel_size, scale, scaling_factor=0.25, normalize=True)
        filters = np.fft.rfft(filters, axis=-1, norm='ortho')

        self.register_buffer('orig', torch.from_numpy(orig_filters))

        padded = np.pad(
            orig_filters, [(0, 0), (0, exp.n_samples - exp.kernel_size)])
        full_size_filters = np.fft.rfft(
            padded, axis=-1, norm='ortho')

        self.register_buffer('filters', torch.from_numpy(filters))
        self.register_buffer('full_size_filters',
                             torch.from_numpy(full_size_filters))
    

    def loss(self, a, b):
        a1, a2 = self.forward(a)
        b1, b2 = self.forward(b)
        return F.mse_loss(a1, b1) + F.mse_loss(a2, b2)

    def forward(self, x) -> Tuple[Tensor, Tensor]:

        x = x.view(-1, 1, self.exp.n_samples)

        spec = torch.fft.rfft(x, dim=-1, norm='ortho')

        conv = spec * self.full_size_filters[None, ...]

        spec = torch.fft.irfft(conv, dim=-1, norm='ortho')

        # half-wave rectification
        spec = torch.relu(spec)

        # compression
        spec = torch.sqrt(spec)

        # loss of phase locking above 5khz (TODO: make this independent of sample rate)
        spec = F.avg_pool1d(spec, kernel_size=3, stride=1, padding=1)

        # compute within-band periodicity
        spec = F.pad(spec, (0, 256)).unfold(-1, 512, 256)
        spec = spec * \
            torch.hamming_window(512, device=spec.device)[None, None, None, :]

        real = spec @ self.orig.real.T
        imag = spec @ self.orig.imag.T

        spec = torch.complex(real, imag)
        spec = torch.abs(spec)

        pooled = spec[..., 0]

        # only frequencies below the current band matter
        spec = torch.tril(spec)

        # we care about the *shape* and not the magnitude here
        if self.norm_second_order:
            norms = torch.norm(spec, dim=-1, keepdim=True)
            spec = spec / (norms + 1e-8)

        return pooled.float(), spec.float()
