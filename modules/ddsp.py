import torch
from torch.nn import functional as F
from torch import nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def noise_bank2(x):
    # TODO: Understand and apply stuff about periodic, zero-phase, causal
    # TODO: Understand and apply stuff about windowing the filter coefficients
    # windows
    batch, magnitudes, samples = x.shape
    window_size = (magnitudes - 1) * 2
    hop_size = window_size // 2
    total_samples = hop_size * samples


    # (batch, frames, coeffs, 2)

    # create the noise
    noise = torch.FloatTensor(batch, total_samples).uniform_(-1, 1).to(x.device)
    # window the noise
    noise = F.pad(noise, (0, hop_size))
    noise = noise.unfold(-1, window_size, hop_size)

    # window = torch.hann_window(32).to(x.device).float()
    # noise = noise * window[None, None, :]

    noise_coeffs = torch.fft.rfft(noise, norm='ortho')
    # (batch frames, coeffs, 2)

    x = x.permute(0, 2, 1)#[..., None]
    # apply the filter in the frequency domain
    # filtered = noise_coeffs * x

    # noise_coeffs[..., :1] *= x[..., None]
    # noise_coeffs[..., 1:] *= x[..., None]

    noise_coeffs = noise_coeffs * x

    filtered = noise_coeffs

    # recover the filtered noise in the time domain
    audio = torch.fft.irfft(
        filtered, n=window_size, norm='ortho')
    
    audio = overlap_add(audio[:, None, :, :], apply_window=True)
    audio = audio[..., :total_samples]
    audio = audio.view(batch, 1, -1)
    return audio


from scipy.signal import hann



def overlap_add(x, apply_window=True):
    batch, channels, frames, samples = x.shape

    if apply_window:
        window = torch.from_numpy(hann(samples, False)).to(x.device).float()
        # window = torch.hamming_window(samples, periodic=False).to(x.device)
        # window = torch.hann_window(samples, periodic=False).to(x.device)
        x = x * window[None, None, None, :]

    hop_size = samples // 2
    first_half = x[:, :, :, :hop_size].contiguous().view(batch, channels, -1)
    second_half = x[:, :, :, hop_size:].contiguous().view(batch, channels, -1)
    first_half = F.pad(first_half, (0, hop_size))
    second_half = F.pad(second_half, (hop_size, 0))
    output = first_half + second_half
    return output



class DDSP(nn.Module):
    def __init__(self, channels, band_size, constrain):
        super().__init__()
        self.channels = channels
        self.band_size = band_size
        noise_samples = 64
        self.noise_frames = band_size // noise_samples
        self.noise_coeffs = (self.noise_frames // 2) + 1
        self.constrain = constrain
        
        self.amp = nn.Conv1d(64, 64, 1, 1, 0)
        self.freq = nn.Conv1d(64, 64, 1, 1, 0)

        bands = np.geomspace(0.01, 1, 64) * np.pi
        bp = np.concatenate([[0], bands])
        spans = np.diff(bp)

        self.bands = torch.from_numpy(bands).float().to(device)
        self.spans = torch.from_numpy(spans).float().to(device)
        
        
        self.noise = nn.Conv1d(64, self.noise_coeffs, 1, 1, 0)
        self.noise_factor = nn.Parameter(torch.FloatTensor(1).fill_(1e-5))
    
    def forward(self, x):
        # noise = torch.sigmoid(self.noise(x))
        
        noise = torch.clamp(self.noise(x), 0, 1)
        noise_step = self.noise_frames // 2
        noise = F.avg_pool1d(noise, noise_step, noise_step) ** 2
        noise = noise_bank2(noise) * self.noise_factor

        # amp = torch.sigmoid(self.amp(x))
        # freq = torch.sigmoid(self.freq(x))

        amp = torch.clamp(self.amp(x), 0, 1)
        freq = torch.clamp(self.freq(x), 0, 1)

        amp = F.avg_pool1d(amp, 64, 1, 32)[..., :-1] ** 2
        freq = F.avg_pool1d(freq, 64, 1, 32)[..., :-1]

        if self.constrain:
            freq = self.bands[None, :, None] + (freq * self.spans[None, :, None])

        freq = torch.sin(torch.cumsum(freq, dim=-1)) * amp
        x = torch.mean(x, dim=1, keepdim=True)
        
        return x + noise