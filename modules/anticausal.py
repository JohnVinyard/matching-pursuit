import torch
from torch import nn
from torch.nn import functional as F


class AntiCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation)
        self.kernel_size = kernel_size
        self.dilation = dilation
    
    def forward(self, x: torch.Tensor):
        x = F.pad(x, (0, ((self.kernel_size * self.dilation) // 2)))
        x = self.conv.forward(x)
        return x


class AntiCausalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.dilation = dilation
        self.conv = AntiCausalConv(channels, channels, kernel_size, dilation)
        self.gate = AntiCausalConv(channels, channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        skip = x
        a = torch.tanh(self.conv(x))
        b = torch.sigmoid(self.gate(x))
        x = a * b
        x = x + skip
        x = self.norm(x)
        return x


class AntiCausalStack(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.blocks = nn.ModuleList([AntiCausalBlock(channels, kernel_size, d) for d in dilations])
        self.norm = nn.BatchNorm1d(channels)
        self.ff = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        output = torch.zeros_like(x)
        for block in self.blocks:
            x = block.forward(x)
            output = output + x
        output = self.norm(output)
        output = self.ff(output)
        return output
    