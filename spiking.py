import torch
from torch import nn
from torch.nn import functional as F
from data import get_one_audio_segment
from modules import gammatone_filter_bank
from modules.transfer import fft_convolve

n_samples = 2 ** 17

class SpikingModel(nn.Module):
    def __init__(self, n_channels, filter_size, periodicity_size):
        super().__init__()
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.periodicity_size = periodicity_size


        gfb = gammatone_filter_bank(
            n_filters=self.n_channels, size=self.filter_size, device='cpu', band_spacing='geometric').view(1, self.n_channels, self.filter_size)

        self.register_buffer('gammatone', gfb, persistent=False)

        self.memory_size = 512
        self.periodicity_memory_size = 8

        memory = (torch.linspace(0, 1, self.memory_size) ** 2).view(1, 1, self.memory_size)
        self.register_buffer('memory', memory, persistent=False)

        periodicity_memory = (torch.linspace(0, 1, self.periodicity_memory_size) ** 2).view(1, 1, self.periodicity_memory_size)
        self.register_buffer('periodicity_memory', periodicity_memory, persistent=False)

    def forward(self, audio: torch.Tensor):

        batch = audio.shape[-1]

        n_samples = audio.shape[-1]
        audio = audio.view(-1, 1, n_samples)

        # convolve with gammatone filters
        g = F.pad(self.gammatone, (0, n_samples - self.filter_size))
        channels = fft_convolve(audio, g)

        # half-wave rectification
        channels = torch.relu(channels)

        # compression
        channels = torch.sqrt(channels)

        print(channels.shape)


        # inhibition via recent-average subtraction
        # (this should eventually include neighboring channel inhibition as well)
        m = F.pad(self.memory, (0, n_samples - self.memory_size))

        print(m.shape)

        pooled = fft_convolve(m, channels)
        normalized = channels - pooled
        normalized = torch.relu(normalized)

        fwd = (channels > normalized).float()
        back = normalized

        # layer one of spiking response.  Unit responses propagate forward,
        # initial real-values propagate backward
        y = back + (fwd - back).detach()


        # compute periodicity
        y = F.pad(y, (0, self.periodicity_size // 2))
        y = y.unfold(-1, self.periodicity_size, self.periodicity_size // 2)
        y = torch.abs(torch.fft.rfft(y, dim=-1))
        y = torch.sqrt(y)


        # y will be (batch, channels, time, coeffs)

        n_frames = y.shape[2]

        pm = self.periodicity_memory.view(1, 1, self.periodicity_memory_size)
        pm = F.pad(pm, (0, n_frames - self.periodicity_memory_size))
        pm = pm.view(1, 1, 1, -1)

        y = y.permute(0, 1, 3, 2)

        pooled = fft_convolve(y, pm)
        normalized = y - pooled
        normalized = torch.relu(normalized)

        fwd = (normalized > 0).float()
        back = normalized
        y = back + (fwd - back).detach()


        return y




if __name__ == '__main__':
    samples = get_one_audio_segment(n_samples, device='cpu')
    samples = samples.view(1, 1, n_samples).to('cpu')
    print(samples.device)

    model = SpikingModel(n_channels=128, filter_size=128, periodicity_size=256)
    binary = model.forward(samples)
    active = (binary > 0).sum()
    sparsity = active / binary.numel()
    print(binary.shape, binary.max().item(), binary.min().item(), sparsity)

