from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from modules import pos_encoded


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
            do_norm: bool = False,
            pos_encodings: bool = False):
        
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels

        self.pos_encodings = pos_encodings
        self.proj = nn.Conv1d(in_channels, channels, 1, 1, 0)

        if pos_encodings:
            self.pos_projection = nn.Conv1d(33, channels, 1, 1, 0)

        self.stack = AntiCausalStack(
            channels, kernel_size, dilations, do_norm=do_norm)

    
    def forward(self, x: torch.Tensor):
        batch, channels, time = x.shape

        x = self.proj(x)

        if self.pos_encodings:
            p = pos_encoded(batch, time, n_freqs=16, device=x.device).permute(0, 2, 1)
            p = self.pos_projection.forward(p)
            x = x + p

        x = self.stack(x)
        return x