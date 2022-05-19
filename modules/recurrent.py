import torch
from torch import nn


from modules.linear import LinearOutputStack


class RecurrentSynth(nn.Module):
    def __init__(self, layers, channels):
        super().__init__()
        self.net = LinearOutputStack(channels, layers)
    
    def forward(self, x):
        pass
