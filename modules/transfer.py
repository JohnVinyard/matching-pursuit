import torch
from torch import nn
import zounds
from .phase import morlet_filter_bank
import numpy as np
from torch.nn import functional as F

class TransferFunction(nn.Module):

    def __init__(
            self, 
            samplerate: zounds.SampleRate, 
            scale: zounds.FrequencyScale, 
            n_frames: int, 
            resolution: int,
            n_samples: int):

        super().__init__()
        self.samplerate = samplerate
        self.scale = scale
        self.n_frames = n_frames
        self.resolution = resolution
        self.n_samples = n_samples

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
        x = F.gumbel_softmax(x, dim=-1)
        x = x @ self.resonance
        x = F.interpolate(x, size=self.n_samples, mode='linear')
        x = x * self.filter_bank
        x = torch.mean(x, dim=1, keepdim=True)
        return x


    
