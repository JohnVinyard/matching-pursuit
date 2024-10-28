import torch
from torch import nn

def unit_norm(x: torch.Tensor, dim: int=-1, epsilon: float=1e-8):
    n = torch.norm(x, dim=dim, keepdim=True)
    return x / (n + epsilon)


def max_norm(x: torch.Tensor, dim=-1, epsilon=1e-8, return_value=False):
    n, _ = torch.max(torch.abs(x), dim=dim, keepdim=True)
    normed = x / (n + epsilon) 
    if return_value:
        return normed, n
    else:
        return normed


class UnitNorm(nn.Module):
    def __init__(self, axis=-1, epsilon=1e-8):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
    
    def forward(self, x):
        return unit_norm(x, dim=self.axis, epsilon=self.epsilon)


class ExampleNorm(nn.Module):
    def __init__(self, axis=(1, 2), epsilon=1e-8):
        super().__init__()
        self.axis = axis
        self.epsilon = epsilon
    
    def forward(self, x):
        stds = torch.std(x, dim=self.axis, keepdim=True)
        return x / (stds + self.epsilon)


class MaxNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, x):
        orig_shape = x.shape
        x = x.view(x.shape[0], -1)
        mx, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
        x = x / (mx + self.epsilon)
        x = x.view(*orig_shape)
        return x
    
def limit_norm(x, dim=2, max_norm=0.9999):
    norm = torch.norm(x, dim=dim, keepdim=True)
    unit_norm = x / (norm + 1e-8)
    clamped_norm = torch.clamp(norm, 0, max_norm)
    x = unit_norm * clamped_norm
    return x

