from typing import Callable
import torch
import numpy as np
from modules.fft import fft_convolve, fft_shift

def matching_pursuit(input: torch.Tensor, target: torch.Tensor):
    batch, n_events, n_samples = input.shape

    norms = torch.norm(input, dim=-1, keepdim=True)
    input = input / (norms + 1e-8)
    
    recon = torch.zeros_like(target)

    for i in range(input.shape[1]):
        atom = input[:, i: i + 1, :]
        feature_map = fft_convolve(atom, target)
        values, maxes = torch.max(feature_map, dim=-1)
        scalar = maxes / n_samples
        shifted = fft_shift(atom, scalar)

        shifted = shifted * values

        
        recon = recon + shifted
        target = target - shifted
    
    return target, recon
    

def serial_loss(
    input: torch.Tensor, 
    target: torch.Tensor, 
    transform: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:

    t = transform(target)

    batch, n_events, n_samples = input.shape
    input = input.view(-1, 1, n_samples)
    input = transform(input)
    input = input.view(batch, n_events, *input.shape[1:])

    for i in range(input.shape[1]):
        x = input[:, i: i + 1, ...]
        t = t - x
    

    return torch.sum(torch.abs(t))