import numpy as np
from scipy.signal import morlet
import torch
from torch import nn
from torch.nn import functional as F

def complex_filters(support, n_filters):
    pass


class SynthesisBank(nn.Module):
    def __init__(self, samplerate, n_osc, n_samples):
        super().__init__()
        self.samplerate = samplerate
        self.n_osc = n_osc
        self.n_samples = n_samples

        freqs = torch.linspace(20 / samplerate.nyquist, samplerate.nyquist * 0.99, n_osc) ** 2
        freqs = freqs.view(n_osc, 1).repeat(1, n_samples) * np.pi
        osc = torch.sin(torch.cumsum(freqs, dim=-1)).view(1, n_osc, n_samples)

        noise = torch.zeros(1, 1, n_samples).uniform_(-1, 1)

        noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')
        osc_spec = torch.fft.rfft(osc, dim=-1, norm='ortho')
        conv = noise_spec * osc_spec
        noise_bank = torch.fft.irfft(conv, dim=-1, norm='ortho').view(1, n_osc, n_samples)

        synth_filters = torch.cat([osc, noise_bank], dim=1)

        self.register_buffer('synth_filters', synth_filters)
    
    @property
    def total_bands(self):
        return self.n_osc * 2
    
    def forward(self, x):
        x = x.view(x.shape[0], self.total_bands, -1)
        x = F.interpolate(x, size=self.n_samples, mode='linear')
        x = x * self.synth_filters
        x = torch.sum(x, dim=1, keepdim=True)
        return x








