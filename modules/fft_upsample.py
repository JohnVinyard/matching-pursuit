import torch
from torch import nn

class FFTUpsample(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch, channels, time = x.shape
        new_time = time * 2
        coeffs = torch.fft.rfft(x, axis=-1, norm='ortho')
        new_coeffs = torch.zeros(batch, channels, new_time // 2 + 1).to(x.device)
        new_coeffs[:, :, :(time // 2 + 1)] = coeffs
        x = torch.fft.irfft(new_coeffs, n=new_time, norm='ortho')
        return x
