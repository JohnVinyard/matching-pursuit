from torch import exp, nn
import torch
from torch.nn import functional as F

from modules.normalization import unit_norm

def fft_based_pif(audio: torch.Tensor, freq_window_size: int, time_window_size: int):
    batch_size = audio.shape[0]
    spec: torch.Tensor = torch.fft.rfft(audio, dim=-1)
    
    freq_step = freq_window_size // 2
    
    windowed = spec.unfold(-1, freq_window_size, freq_step)
    windowed = windowed * torch.hamming_window(freq_window_size, device=windowed.device)[None, None, None, :]
    channels = torch.fft.irfft(windowed, dim=-1)
    n_channels = channels.shape[2]
    
    # half-wave rectification
    channels = torch.relu(channels)
    
    # compression
    # TODO: How does this compare to the decibel-like transformation
    # transformation
    channels = torch.sqrt(channels)
    
    time_domain_window_size = time_window_size
    time_domain_step_size = time_domain_window_size // 2
    
    channels = channels.view(batch_size, n_channels, -1)
    channels = channels.unfold(-1, time_domain_window_size, time_domain_step_size)
    channels = channels * torch.hamming_window(channels.shape[-1], device=channels.device)[None, None, None, :]
    
    # we want to capture fine-scale information in a phase-invariant way
    # just keep the magnitudes
    channels = torch.abs(torch.fft.rfft(channels, dim=-1))
    
    return channels



class AuditoryImage(nn.Module):
    """
    Take the half-wave rectified output from a filterbank
    and convert it to a three-dimensional representation
    of (time, frequency, periodicity) with the new dimension
    representing intervals present within each channel, ignoring
    absolute phase
    """

    def __init__(
            self,
            window_size,
            time_steps,
            do_windowing=True,
            check_cola=True,
            causal=False,
            exp_decay=False,
            residual=False,
            twod=False,
            norm_periodicities=False):

        super().__init__()
        self.window_size = window_size
        self.time_steps = time_steps
        if do_windowing:
            self.register_buffer('window', torch.hamming_window(window_size))
        elif exp_decay:
            self.register_buffer('window', torch.hamming_window(
                window_size * 2)[:window_size])
        self.do_windowing = do_windowing
        self.check_cola = check_cola
        self.causal = causal
        self.exp_decay = exp_decay
        self.residual = residual
        self.twod = twod
        self.norm_periodicities = norm_periodicities

    def forward(self, x):
        batch, channels, time = x.shape
        padding = self.window_size // 2


        pad = (padding, 0) if self.causal else (0, padding)

        x = F.pad(x, pad)
        step = time // self.time_steps

        if self.check_cola:
            if step != self.window_size // 2:
                raise ValueError(
                    f'window and step ({self.window_size}, {step}) violate COLA')

        x = x.unfold(-1, self.window_size, step)

        if self.do_windowing or self.exp_decay:
            x = x * self.window[None, None, None, :]
        
        if self.residual:
            mean = torch.mean(x, dim=-1, keepdim=True)
            residual = x - mean
            r = torch.fft.rfft(residual, dim=-1, norm='ortho')
            r = torch.abs(r)
            return torch.cat([mean.view(batch, -1), r.view(batch, -1)], dim=-1)
        elif self.twod:
            x = x.permute(0, 2, 1, 3)
            r = torch.fft.rfft2(x, dim=(-1, -2), norm='ortho')
            r = torch.abs(r)
            return r
        else:
            x = torch.fft.rfft(x, dim=-1, norm='ortho')
            x = torch.abs(x)

        
        if self.norm_periodicities:
            x = unit_norm(x, dim=-1)

        return x


if __name__ == '__main__':
    inp = torch.FloatTensor(8, 128, 16384)

    model = AuditoryImage(256, 64)

    inp = model(inp)
    print(inp.shape)
