from typing import Any
import torch
from torch import nn
import zounds

from modules.ddsp import overlap_add
from .phase import morlet_filter_bank
import numpy as np
from torch.nn import functional as F

class STFTTransferFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 512
        self.n_coeffs = self.window_size // 2 + 1
        self.n_samples = 2 ** 15
        self.step_size = self.window_size // 2
        self.n_frames = self.n_samples // self.step_size

        self.dim = self.n_coeffs * 2
    
    def forward(self, tf):
        batch, n_coeffs = tf.shape
        if n_coeffs != self.dim:
            raise ValueError(f'Expected (*, {self.dim}) but got {tf.shape}')
        
        tf = tf.view(-1, self.n_coeffs * 2, 1)
        tf = tf.repeat(1, 1, self.n_frames)
        tf = tf.view(-1, self.n_coeffs, 2, self.n_frames)

        tf = tf.view(-1, self.n_coeffs * 2, self.n_frames)

        real = torch.clamp(tf[:, :self.n_coeffs, :], 0, 1) * 0.9999
        imag = torch.clamp(tf[:, self.n_coeffs:, :], -1, 1) * np.pi


        real = real * torch.cos(imag)
        imag = real * torch.sin(imag)
        tf = torch.complex(real, imag)
        tf = torch.cumprod(tf, dim=-1)

        tf = tf.view(-1, self.n_coeffs, self.n_frames)
        tf = torch.fft.irfft(tf, dim=1, norm='ortho')\
            .permute(0, 2, 1)\
            .view(batch, 1, self.n_frames, self.window_size)
        tf = overlap_add(tf, trim=self.n_samples)
        return tf



class TransferFunction(nn.Module):

    def __init__(
            self, 
            samplerate: zounds.SampleRate, 
            scale: zounds.FrequencyScale, 
            n_frames: int, 
            resolution: int,
            n_samples: int,
            softmax_func: Any):

        super().__init__()
        self.samplerate = samplerate
        self.scale = scale
        self.n_frames = n_frames
        self.resolution = resolution
        self.n_samples = n_samples
        self.softmax_func = softmax_func

        bank = morlet_filter_bank(
            samplerate, n_samples, scale, 0.1, normalize=False)\
            .real.astype(np.float32)
        
        self.register_buffer('filter_bank', torch.from_numpy(bank)[None, :, :])

        resonances = torch.linspace(0, 0.999, resolution)\
            .view(resolution, 1).repeat(1, n_frames)
        resonances = torch.cumprod(resonances, dim=-1)
        self.register_buffer('resonance', resonances)
    
    @property
    def n_bands(self):
        return self.scale.n_bands
    
    def forward(self, x: torch.Tensor):
        batch, bands, resolution = x.shape
        if bands != self.n_bands or resolution != self.resolution:
            raise ValueError(
                f'Expecting tensor with shape (*, {self.n_bands}, {self.resolution})')
        
        # x = F.gumbel_softmax(x, dim=-1, hard=True)
        # x = F.softmax(x, dim=-1)
        x = self.softmax_func(x)
    
        x = x @ self.resonance
        x = F.interpolate(x, size=self.n_samples, mode='linear')
        x = x * self.filter_bank
        x = torch.mean(x, dim=1, keepdim=True)
        return x


    
