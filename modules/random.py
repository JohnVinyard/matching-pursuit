from typing import Callable, Optional, Union
import torch
from torch import nn
from torch.nn.init import orthogonal_

TensorTransform = Callable[[torch.Tensor], torch.Tensor]


class RandomProjection(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: Optional[TensorTransform] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm = norm
        
        pm = torch.zeros(in_channels, out_channels).uniform_(-1, 1)
        
        self.register_buffer('projection_matrix', pm)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print(x.shape, self.projection_matrix.shape)
        
        result = x @ self.projection_matrix
        if self.norm is not None:
            result = self.norm(result)
        return result