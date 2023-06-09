import torch
from torch import nn


def unit_sine(x):
    return (torch.sin(x) + 1) * 0.5

class Sine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)
