import torch
from torch.nn import functional as F

def hard_softmax(x, dim=-1):
    return F.gumbel_softmax(x, tau=1, dim=dim, hard=True)


def sparse_softmax(x):
    x_backward = torch.softmax(x, dim=-1)
    values, indices = torch.max(x_backward, dim=-1, keepdim=True)
    # values = values + (1 - values)
    x_forward = torch.zeros_like(x_backward)
    x_forward = torch.scatter(x_forward, dim=-1, index=indices, src=values)
    y = x_backward + (x_forward - x_backward).detach()
    return y
