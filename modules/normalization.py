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


"""
def project_and_limit_norm(
        vector: torch.Tensor,
        matrix: torch.Tensor,
        max_efficiency: float = max_efficiency) -> torch.Tensor:
    # get the original norm, this is the absolute max norm/energy we should arrive at,
    # given a perfectly efficient physical system
    original_norm = torch.norm(vector, dim=-1, keepdim=True)
    # project
    x = vector @ matrix

    # TODO: clamp norm should be a utility that lives in normalization
    # find the norm of the projection
    new_norm = torch.norm(x, dim=-1, keepdim=True)
    # clamp the norm between the allowed values
    clamped_norm = torch.clamp(new_norm, min=None, max=original_norm * max_efficiency)

    # give the projected vector the clamped norm, such that it
    # can have lost some or all energy, but not _gained_ any
    normalized = unit_norm(x, axis=-1)
    x = normalized * clamped_norm
    return x

"""
    
def limit_norm(x, dim=2, max_norm=0.9999):
    # give x unit norm
    norm = torch.norm(x, dim=dim, keepdim=True)
    unit_norm = x / (norm + 1e-8)

    clamped_norm = torch.clamp(norm, min=None, max=max_norm)
    x = unit_norm * clamped_norm
    return x

