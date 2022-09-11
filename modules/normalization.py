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



class ExampleNorm(nn.Module):
    def __init__(self, axis=(1, 2), epsilon=1e-8):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
    
    def forward(self, x):
        stds = torch.std(x, dim=self.axis, keepdim=True)
        return x / (stds + self.epsilon)


def limit_norm(x, dim=2):
    norm = torch.norm(x, dim=dim, keepdim=True)
    unit_norm = x / (norm + 1e-8)
    clamped_norm = torch.clamp(norm, 0, 0.9999)
    x = unit_norm * clamped_norm
    return x
