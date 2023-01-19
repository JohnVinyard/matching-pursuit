import torch
from torch.distributions import Normal
from torch import nn
from modules.ddsp import overlap_add
from modules.fft import fft_convolve
from modules.normalization import max_norm
from upsample import ConvUpsample
from torch.nn import functional as F


class PhysicalSimulation(nn.Module):
    def __init__(self, room_size, buffer_size):
        super().__init__()
        self.room_size = room_size
        self.buffer_size = buffer_size

    def forward(
            self,
            n_samples: int,
            impulses: torch.Tensor,
            delays: torch.Tensor):

        pass


class Window(nn.Module):
    def __init__(self, n_samples, mn, mx, epsilon=1e-8, padding=0):
        super().__init__()
        self.n_samples = n_samples
        self.mn = mn
        self.mx = mx
        self.scale = self.mx - self.mn
        self.epsilon = epsilon
        self.padding = padding

    def forward(self, means, stds):
        dist = Normal(self.mn + (means * self.scale), self.epsilon + stds)
        rng = torch.linspace(0, 1, self.n_samples, device=means.device)[
            None, None, :]
        windows = torch.exp(dist.log_prob(rng))
        windows = max_norm(windows)
        return windows


class BlockwiseResonatorModel(nn.Module):
    def __init__(self, n_samples, n_frames, channels, n_events):
        super().__init__()
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.channels = channels
        self.n_events = n_events

        self.step = n_samples // n_frames
        self.window_size = self.step * 2
        self.n_coeffs = self.window_size // 2 + 1

        self.to_impulse = ConvUpsample(
            channels, channels, 8, self.n_frames, mode='linear', out_channels=1)
        
        self.to_transfer = ConvUpsample(
            channels, channels, 8, self.n_frames, mode='linear', out_channels=self.n_coeffs)
        
        self.to_loc = ConvUpsample(
            channels, channels, start_size=8, end_size=n_frames, mode='learned', out_channels=1
        )
    
    def forward(self, x):

        

        loc = self.to_loc(x)
        loc = F.gumbel_softmax(loc, dim=-1, hard=True)
        loc_full = torch.zeros(loc.shape[0], 1, self.n_samples, device=loc.device)
        step = self.n_samples // self.channels

        loc_full[:, :, ::step] = loc

        imp = self.to_impulse(x) ** 2

        imp = F.interpolate(imp, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples).uniform_(-1, 1)
        imp = imp * noise
        imp = F.pad(imp, (0, self.step))
        windowed = imp.unfold(-1, 512, 256) * torch.hamming_window(self.window_size)[None, None, None, :]
        freq = torch.fft.rfft(windowed, dim=-1, norm='ortho')

        t = torch.sigmoid(self.to_transfer(x)) ** 2

        output = []

        for i in range(self.n_frames):
            x = freq[:, :, i, :]
            if i > 0:
                recur = output[i - 1].view(x.shape[0], 1, self.n_coeffs) * t[..., i][:, None, :]
                x = x + recur
            x = x[:, :, None, :]
            output.append(x)
        
        x = torch.cat(output, dim=2)
        x = torch.fft.irfft(x, dim=-1, norm='ortho')
        x = overlap_add(x, apply_window=False)[..., :self.n_samples]

        x = fft_convolve(x, loc_full)[..., :self.n_samples]

        x = x.view(-1, self.n_events, self.n_samples)
        return x

def harmonics(n_octaves, waveform, device):

    rng = torch.arange(1, n_octaves + 1, device=device)

    if waveform == 'sawtooth':
        return 1 / rng
    elif waveform == 'square':
        rng = 1 / rng
        rng[::2] = 0
        return rng
    elif waveform == 'triangle':
        rng = 1 / (rng ** 2)
        rng[::2] = 0
        return rng
