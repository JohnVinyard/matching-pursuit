from torch import nn
import torch
from torch.nn import functional as F

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
    def forward(self, x):
        batch = x.shape[0]
        skip = x
        scale = self.scale(x)
        gate = self.gate(x)
        x = torch.tanh(scale) * torch.sigmoid(gate)
        out = self.out(x)
        next = self.next(x) + skip
        return next, out




class DilatedStack(nn.Module):
    def __init__(self, channels, dilations):
        super().__init__()
        self.stack = nn.Sequential(*[DilatedBlock(channels, d) for d in dilations])
        self.channels = channels
    
    def forward(self, x, return_features=False):
        batch = x.shape[0]
        n = x
        outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)
        features = []

        for layer in self.stack:
            n, o = layer.forward(n)
            features.append(n)
            outputs = outputs + o
        if return_features:
            return outputs, torch.cat([x.view(-1) for x in features])
        else:
            return outputs