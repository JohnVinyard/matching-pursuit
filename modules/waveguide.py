from cmath import isnan
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F

class WaveguideSynth(nn.Module):
    def __init__(self, max_delay=512, n_samples=2**15, filter_kernel_size=512):
        super().__init__()

        self.n_delays = max_delay
        self.n_samples = n_samples
        self.filter_kernel_size = filter_kernel_size

        delays = torch.zeros(max_delay, n_samples)
        for i in range(max_delay):
            delays[i, ::(i + 1)] = 1
        
        self.register_buffer('delays', delays)
    
    def forward(self, impulse, delay_selection, damping, filt):
        batch = delay_selection.shape[0]
        n_frames = filt.shape[-1]

        # filter (TODO: Should I allow a time-dependent transfer function?)

        f = torch.sigmoid(filt).view(-1, 1, 16)
        f = F.interpolate(f, size=self.n_samples // 2, mode='linear')
        f = F.pad(f, (0, 1))
        filt_spec = f
        
        # filt = torch.fft.irfft(f, dim=-1, norm='ortho').view(batch, self.filter_kernel_size)
        # filt = filt * torch.hamming_window(self.filter_kernel_size, device=impulse.device)[None, :]
        # diff = self.n_samples - self.filter_kernel_size
        # filt = torch.cat([filt, torch.zeros(batch, diff)], dim=-1).view(batch, 1, self.n_samples)

        # Impulse / energy
        impulse = impulse.view(batch, 1, -1) ** 2
        impulse = F.interpolate(impulse, size=self.n_samples, mode='linear')
        noise = torch.zeros(batch, 1, self.n_samples).uniform_(-1, 1)
        impulse = impulse * noise

        # Damping for delay (single value per batch member/item)
        damping = torch.sigmoid(damping.view(batch, 1)) * 0.9999
        powers = torch.linspace(1, damping.shape[-1], steps=n_frames)
        damping = damping[:, :, None] ** powers[None, None, :]
        damping = F.interpolate(damping, size=self.n_samples, mode='nearest')

        # delay count (TODO: should this be sparsified?)
        delay_selection = delay_selection.view(batch, self.n_delays, -1)
        delay_selection = torch.softmax(delay_selection, dim=1)
        delay_selection = F.interpolate(delay_selection, size=self.n_samples, mode='nearest')

        d = (delay_selection * self.delays).sum(dim=1, keepdim=True) * damping

        delay_spec = torch.fft.rfft(d, dim=-1, norm='ortho')
        impulse_spec = torch.fft.rfft(impulse, dim=-1, norm='ortho')
        # filt_spec = torch.fft.rfft(filt, dim=-1, norm='ortho')

        spec = delay_spec * impulse_spec * filt_spec

        final = torch.fft.irfft(spec, dim=-1, norm='ortho')
        return final


def waveguide_synth(
    impulse: np.ndarray, 
    delay: np.ndarray, 
    damping: np.ndarray, 
    filter_size: int) -> np.ndarray:
    
    n_samples = impulse.shape[0]

    output = impulse.copy()
    buf = np.zeros_like(impulse)

    for i in range(n_samples):
        delay_val = 0

        delay_amt = delay[i]
        
        if i > delay_amt:
            damping_amt = damping[i]
            delay_val += output[i - delay_amt] * damping_amt

        
        buf[i] = delay_val

        filt_size = filter_size[i]        
        filt_slice = buf[i - filt_size: i]   
        if filt_slice.shape[0]:
            new_val = np.mean(filt_slice)
        else:
            new_val = delay_val

        output[i] += new_val
    
    return output

