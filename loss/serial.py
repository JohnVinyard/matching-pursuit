from typing import Callable
import torch
import numpy as np

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