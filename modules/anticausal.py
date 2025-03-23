from typing import List
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm

from modules import pos_encoded


class AntiCausalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, reverse_causality: bool = False):
        super().__init__()
        conv = weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation))
        self.conv = conv
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.reverse_causality = reverse_causality
    
    def forward(self, x: torch.Tensor):
        if self.reverse_causality:
           x =  F.pad(x, (((self.kernel_size * self.dilation) // 2), 0))
        else:
            x = F.pad(x, (0, ((self.kernel_size * self.dilation) // 2)))
        x = self.conv.forward(x)
        return x


class AntiCausalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, do_norm: bool = False, reverse_causality: bool = False):
        super().__init__()
        self.dilation = dilation
        self.conv = AntiCausalConv(channels, channels, kernel_size, dilation, reverse_causality=reverse_causality)
        self.gate = AntiCausalConv(channels, channels, kernel_size, dilation, reverse_causality=reverse_causality)

        self.tanh_weight = nn.Parameter(torch.zeros(1).fill_(0.5))
        self.sigmoid_weight = nn.Parameter(torch.zeros(1).fill_(0.5))

        self.norm = nn.BatchNorm1d(channels)
        self.do_norm = do_norm
    
    def forward(self, x):
        skip = x
        a = torch.tanh(self.conv(x) * self.tanh_weight)
        b = torch.sigmoid(self.gate(x) * self.sigmoid_weight)

        x = a * b
        x = x + skip
        
        if self.do_norm:
            x = self.norm(x)
        return x


class AntiCausalStack(nn.Module):
    def __init__(self, channels, kernel_size, dilations, do_norm: bool = False, reverse_causality: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            AntiCausalBlock(channels, kernel_size, d, do_norm=do_norm, reverse_causality=reverse_causality) for d in dilations])
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
            pos_encodings: bool = False,
            reverse_causality: bool = False,):
        
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels

        self.pos_encodings = pos_encodings
        self.proj = nn.Conv1d(in_channels, channels, 1, 1, 0)

        if pos_encodings:
            self.pos_projection = nn.Conv1d(33, channels, 1, 1, 0)

        self.stack = AntiCausalStack(
            channels, kernel_size, dilations, do_norm=do_norm, reverse_causality=reverse_causality)

    
    def forward(self, x: torch.Tensor):
        batch, channels, time = x.shape

        x = self.proj(x)

        if self.pos_encodings:
            p = pos_encoded(batch, time, n_freqs=16, device=x.device).permute(0, 2, 1)
            p = self.pos_projection.forward(p)
            x = x + p

        x = self.stack(x)
        return x