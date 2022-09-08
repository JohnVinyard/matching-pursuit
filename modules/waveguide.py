from cmath import isnan
import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from fft_basis import morlet_filter_bank
from modules.ddsp import overlap_add
from modules.normalization import ExampleNorm
from modules.shape import Reshape
import zounds

from upsample import ConvUpsample, PosEncodedUpsample

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
        noise = torch.zeros(batch, 1, self.n_samples, device=impulse.device).uniform_(-1, 1)
        impulse = impulse * noise

        # Damping for delay (single value per batch member/item)
        damping = torch.sigmoid(damping.view(batch, 1)) * 0.9999
        powers = torch.linspace(1, damping.shape[-1], steps=n_frames, device=impulse.device)
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



class TransferFunctionSegmentGenerator(nn.Module):
    def __init__(self, model_dim, n_frames, window_size, n_samples):
        super().__init__()
        
        self.model_dim = model_dim
        self.n_frames = n_frames
        self.window_size = window_size
        self.n_samples = n_samples

        self.to_damping = nn.Linear(model_dim, 1)

        self.env = ConvUpsample(
            model_dim, model_dim, 4, n_frames * 4, mode='nearest', out_channels=1, norm=ExampleNorm())

        # n_coeffs = window_size // 2 + 1
        # self.n_coeffs = n_coeffs
        self.n_bands = 128


        self.transfer = ConvUpsample(
            model_dim, model_dim, 4, 8, mode='nearest', out_channels=self.n_bands, norm=ExampleNorm()
        )

        samplerate = zounds.SR22050()
        band = zounds.FrequencyBand(40, samplerate.nyquist)
        scale = zounds.MelScale(band, self.n_bands)
        bank = morlet_filter_bank(samplerate, 2**15, scale, 0.1).real
        mx = np.max(bank, axis=-1, keepdims=True)
        bank = bank / mx

        self.register_buffer('bank', torch.from_numpy(bank).float()[None, :, :])

        
        

    def forward(self, x):
        x = x.view(-1, self.model_dim)

        # damping
        d = 0.5 + (torch.sigmoid(self.to_damping(x)) * 0.4999)
        d = d\
            .view(-1, 1, 1)\
            .repeat(1, 1, self.n_frames)
        # pow = torch.arange(1, self.n_frames + 1, device=x.device)[None, None, :]
        d = torch.cumprod(d, dim=-1)
        # d = d ** pow
        d = F.interpolate(d, size=self.n_samples, mode='linear')

        # TODO: envelope generator
        env = self.env(x) ** 2
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        tf = torch.clamp(self.transfer(x), 0, 1)
        tf = F.interpolate(tf, size=self.n_samples, mode='linear')
        tf = (tf * self.bank).sum(dim=1, keepdim=True) * d

        
        # TODO: FFT convolve, if it doesn't already exist
        env_spec = torch.fft.rfft(env, dim=-1, norm='ortho')
        tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
        spec = env_spec * tf_spec
        final = torch.fft.irfft(spec, dim=-1, norm='ortho')

        return final, None


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

