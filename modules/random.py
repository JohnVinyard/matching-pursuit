from typing import Callable, Optional, Union
import torch
from torch import nn

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class RandomProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: Optional[TensorTransform] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        
        self.register_buffer(
            'projection_matrix', 
            torch.zeros(in_channels, out_channels).uniform_(-1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = x @ self.projection_matrix
        if self.norm is not None:
            result = self.norm(result)
        return result