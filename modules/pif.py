from torch import nn
import torch
from torch.nn import functional as F

class AuditoryImage(nn.Module):
    """
    Take the half-wave rectified output from a filterbank
    and convert it to a three-dimensional representation
    of (time, frequency, periodicity) with the new dimension
    representing intervals present within each channel, ignoring
    absolute phase
    """

    def __init__(self, window_size, time_steps, do_windowing=True):
        super().__init__()
        self.window_size = window_size
        self.time_steps = time_steps
        self.register_buffer('window', torch.hamming_window(window_size))
        self.do_windowing = do_windowing
    
    def forward(self, x):
        batch, channels, time = x.shape
        padding = self.window_size // 2
        x = F.pad(x, (0, padding))
        step = time // self.time_steps
        x = x.unfold(-1, self.window_size, step)
        if self.do_windowing:
            x = x * self.window[None, None, None, :]
        x = torch.abs(torch.fft.rfft(x, dim=-1, norm='ortho'))
        return x


if __name__ == '__main__':
    inp = torch.FloatTensor(8, 128, 16384)

    model = AuditoryImage(256, 64)

    inp = model(inp)
    print(inp.shape)
