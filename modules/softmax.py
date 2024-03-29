import torch
from torch.nn import functional as F

from modules.normalization import max_norm


def hard_softmax(x, dim=-1, invert=False, tau=1):
    return F.gumbel_softmax(
        torch.exp(max_norm(x)) if invert else x, 
        tau=tau, 
        dim=dim, 
        hard=True)


def sparse_softmax(x, normalize=False, dim=-1):
    x_backward = torch.softmax(x, dim=dim)
    values, indices = torch.max(x_backward, dim=dim, keepdim=True)
    if normalize:
        values = values + (1 - values)
    x_forward = torch.zeros_like(x_backward)
    x_forward = torch.scatter(x_forward, dim=dim, index=indices, src=values)
    y = x_backward + (x_forward - x_backward).detach()
    return y

def soft_clamp(x):
    x_backward = x
    x_forward = torch.clamp(x_backward, 0, 1)
    y = x_backward + (x_forward - x_backward).detach()
    return y


def step_func(x: torch.Tensor):
    forward = torch.sign(x)
    backward = x
    y = backward + (forward - backward).detach()
    return y
