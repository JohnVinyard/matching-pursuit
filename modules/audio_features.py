from torch import nn
import torch

class MFCC(nn.Module):
    def __init__(self, n_coeffs=12):
        super().__init__()
        self.n_coeffs = n_coeffs
    
    def mfcc(self, x):
        x = x.view(batch_size, 64, 32, -1)
        x = torch.norm(x, dim=-1)
        x = x.permute(0, 2, 1) # (batch, 32, 64)
        x = torch.rfft(x, signal_ndim=1, normalized=True) # (batch, 32, 33, 2)
        x = torch.norm(x, dim=-1) # (bach, 32, 33)
        x = x.permute(0, 2, 1) # (batch, 33, 32)
        x = x[:, 1:self.n_coeffs + 1, :]
        norms = torch.norm(x, dim=1, keepdim=True)
        x = x / (norms + 1e-12)
        return x
    
    def forward(self, x):
        mfcc = x = self.mfcc(x)
        x = self.transform(x)
        return mfcc, x


class Chroma(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.register_buffer('basis', torch.from_numpy(basis).float())
        self.transform = nn.Conv1d(12, network_channels, 1, 1, 0)
    
    def chroma(self, x):
        x = x.view(batch_size, 64, 32, -1)
        x = torch.norm(x, dim=-1)
        x = x.permute(0, 2, 1) # (batch, 32, 64)
        x = torch.matmul(x, self.basis.permute(1, 0))
        x = x.permute(0, 2, 1) # (batch, 12, 32)
        norms = torch.norm(x, dim=1, keepdim=True)
        x = x / (norms + 1e-12)
        return x
    
    def forward(self, x):
        chroma = x = self.chroma(x)
        x = self.transform(x)
        return chroma, x
