import torch
from torch import nn
from .diffindex import quantize

class VQ(nn.Module):
    def __init__(self, n_codes, code_dim):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.codebook = nn.Parameter(
            torch.zeros(self.n_codes, self.code_dim).uniform_(-0.01, 0.01))
    
    def forward(self, x):
        codes = quantize(x, self.codebook)
        return x, codes