import torch
from torch import nn

class UnitNorm(nn.Module):
    def __init__(self, axis=-1, epsilon=1e-8):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
    
    def forward(self, x):
        norms = torch.norm(x, dim=self.axis, keepdim=True)
        x = x / (norms + self.epsilon)
        return x