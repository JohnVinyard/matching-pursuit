import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

from modules.sparse import sparsify


class DilatedBlock(nn.Module):
    def __init__(
            self,
            channels,
            dilation,
            dropout=None,
            padding: str = None,
            sparsity_amt: float = 1,
            soft_sparsity: bool = False):

        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)

        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation)

        self.dropout = dropout
        self.padding = padding
        self.sparsity_amt = sparsity_amt
        self.soft_sparsity = soft_sparsity

    def forward(self, x):
        batch = x.shape[0]

        if self.dropout:
            x = F.dropout(x, p=self.dropout)

        skip = x
        if self.padding == 'only-past':
            x = F.pad(x, (self.dilation * 2, 0))
        elif self.padding == 'only-future':
            x = F.pad(x, (0, self.dilation * 2))
        else:
            x = F.pad(x, (self.dilation, self.dilation))

        scale = self.scale(x)
        gate = self.gate(x)
        x = torch.tanh(scale) * torch.sigmoid(gate)
        out = self.out(x)
        n = self.next(x) + skip

        if self.sparsity_amt < 1:
            k_sparse = int(np.product(n.shape[1:]) * self.sparsity_amt)
            next = sparsify(n, k_sparse, return_indices=False, soft=self.soft_sparsity)

        return n, out


class DilatedStack(nn.Module):
    def __init__(
            self, 
            channels, 
            dilations, 
            dropout=None,
            padding: str = None,
            sparsity_amt: float = None,
            soft_sparsity: bool = False,
            internally_sparse: bool = False):
    
        super().__init__()

        self.stack = nn.Sequential(
            *[DilatedBlock(
                channels, 
                d, 
                dropout=dropout, 
                padding=padding, 
                sparsity_amt=sparsity_amt if internally_sparse else 1, 
                soft_sparsity=soft_sparsity) for d in dilations])
        
        self.channels = channels
        self.padding = padding
        self.sparsity_amt = sparsity_amt or 1
        self.soft_sparsity = soft_sparsity

    def forward(self, x, return_features=False):
        batch = x.shape[0]
        n = x
        outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)
        features = []

        for layer in self.stack:
            n, o = layer.forward(n)
            features.append(n)
            outputs = outputs + o
        
        if self.sparsity_amt < 1:
            k_sparse = int(np.product(outputs.shape[1:]) * self.sparsity_amt)
            outputs = sparsify(outputs, k_sparse, return_indices=False, soft=self.soft_sparsity)
        
        if return_features:
            return outputs, torch.cat([x.view(-1) for x in features])
        else:
            return outputs
