from typing import List
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
    def __init__(self, channels, kernel_size, dilation, do_norm: bool = False):
        super().__init__()
        self.dilation = dilation
        self.conv = AntiCausalConv(channels, channels, kernel_size, dilation)
        self.gate = AntiCausalConv(channels, channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(channels)
        self.do_norm = do_norm
    
    def forward(self, x):
        skip = x
        a = torch.tanh(self.conv(x))
        b = torch.sigmoid(self.gate(x))
        x = a * b
        x = x + skip
        
        if self.do_norm:
            x = self.norm(x)
        return x


class AntiCausalStack(nn.Module):
    def __init__(self, channels, kernel_size, dilations, do_norm: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            AntiCausalBlock(channels, kernel_size, d, do_norm=do_norm) for d in dilations])
        self.ff = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        output = torch.zeros_like(x)
        for block in self.blocks:
            x = block.forward(x)
            output = output + x
        output = self.ff(output)
        return output


class AntiCausalAnalysis(nn.Module):
    """Wrapper around AntiCausalStack that first projects
    from time-frequency transform channels to internal channels
    """
    def __init__(
            self, 
            in_channels: int,
            channels: int, 
            kernel_size: int, 
            dilations: List[int], 
            do_norm: bool = False):
        
        super().__init__()
        
        self.proj = nn.Conv1d(in_channels, channels, 1, 1, 0)
        self.stack = AntiCausalStack(
            channels, kernel_size, dilations, do_norm=do_norm)
    
    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        x = self.stack(x)
        return x