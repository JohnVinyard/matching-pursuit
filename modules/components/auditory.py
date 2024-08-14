"""
Auditory-inspired transforms that result in non-negative representations
that lend themselves to iterative decomposition.

Ideally, they are robust to:
    - perceptually irrelevant phase shifts
    - noise

so that perceptually-irrelevant details can be ignored
by the analysis network
"""

from torch import nn
import torch

from modules.stft import stft


class STFTTransform(nn.Module):
    def __init__(
            self, 
            window_size: int = 2048, 
            step_size: int = 256,
            frequency_plane_slice: slice = slice(None)):
        
        super().__init__()
        self.window_size = window_size
        self.n_coeffs = window_size // 2 + 1
        self.step_size = step_size
        self.frequency_plane_slice = frequency_plane_slice
    
    def forward(self, samples: torch.Tensor):
        batch, channels, time = samples.shape
        
        if channels != 1:
            raise ValueError(f'channels must equal 1, but was {channels}')
        
        spec = stft(
            samples, 
            ws=self.window_size, 
            step=self.step_size, 
            pad=True)
        
        spec = spec.view(batch, -1, self.n_coeffs)
        spec = spec.permute(0, 2, 1)
        spec = spec[:, self.frequency_plane_slice, :]
        return spec
        