from torch import nn
import torch
from modules.atoms import unit_norm


class MFCC(nn.Module):
    def __init__(self, n_coeffs=12):
        super().__init__()
        self.n_coeffs = n_coeffs

    def mfcc(self, x):
        batch, freq_bins, time = x.shape
        # assume that the time-frequency representation already has
        # somewhat log-spaced frequency bins
        cepstrum = torch.fft.rfft(x, dim=1, norm='ortho')
        mag = torch.log(torch.abs(cepstrum) + 1e-12)
        coeffs = mag[:, 1:self.n_coeffs + 1, :]
        # coeffs = unit_norm(coeffs, axis=1)
        return coeffs

    def forward(self, x):
        mfcc = self.mfcc(x)
        return mfcc


class Chroma(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.register_buffer('basis', torch.from_numpy(basis).float())

    def chroma(self, x):
        batch, freq_bins, time = x.shape
        # x = x.T @ self.basis.T
        x = x.permute(0, 2, 1) @ self.basis.T
        x = x.permute(0, 2, 1)
        # x = unit_norm(x, axis=1)
        return x

    def forward(self, x):
        chroma = self.chroma(x)
        return chroma
