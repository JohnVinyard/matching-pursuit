import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from data import get_one_audio_segment
from modules import gammatone_filter_bank, max_norm, unit_norm
from modules.overfitraw import OverfitRawAudio
from modules.transfer import fft_convolve
from conjure import LmdbCollection, Logger, loggers, serve_conjure
from util import device, encode_audio
from itertools import count
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_

n_samples = 2 ** 17


class SpikingModel(nn.Module):
    def __init__(self, n_channels, filter_size, periodicity_size, memory_size=512, frame_memory_size=8):
        super().__init__()
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.periodicity_size = periodicity_size

        gfb = gammatone_filter_bank(
            n_filters=self.n_channels, size=self.filter_size, device='cpu', band_spacing='geometric').view(1,
                                                                                                           self.n_channels,
                                                                                                           self.filter_size)
        gfb = unit_norm(gfb)

        self.register_buffer('gammatone', gfb, persistent=False)

        self.memory_size = memory_size
        self.periodicity_memory_size = frame_memory_size

        memory = (torch.linspace(0, 1, self.memory_size) ** 2).view(1, 1, self.memory_size)
        self.register_buffer('memory', memory, persistent=False)

        periodicity_memory = (torch.linspace(0, 1, self.periodicity_memory_size) ** 2).view(1, 1,
                                                                                            self.periodicity_memory_size)
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
        # channels = torch.log(channels)

        # print(channels.shape)

        # inhibition via recent-average subtraction
        # (this should eventually include neighboring channel inhibition as well)
        m = F.pad(self.memory, (0, n_samples - self.memory_size))

        # print(m.shape)

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
        # y = torch.log(y)

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


def overfit_model():
    target = get_one_audio_segment(n_samples).to(device).view(1, 1, n_samples)
    target = max_norm(target)

    loss_model = SpikingModel(
        n_channels=128,
        filter_size=128,
        periodicity_size=128,
        memory_size=64,
        frame_memory_size=8).to(device)

    overfit_model = OverfitRawAudio(target.shape, std=0.01, normalize=True).to(device)
    optim = Adam(overfit_model.parameters(), lr=1e-3)

    collection = LmdbCollection('spiking')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig'],
        'audio/wav',
        encode_audio,
        collection)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=9999, n_workers=1)

    target_features = loss_model.forward(target)

    for i in count():
        optim.zero_grad()
        recon = overfit_model.forward(None)
        recon = max_norm(recon)
        recon_audio(max_norm(recon))

        recon_features = loss_model.forward(recon)

        loss = torch.abs(target_features - recon_features).sum()
        loss.backward()
        # clip_grad_value_(overfit_model.parameters(), 0.1)
        optim.step()

        print(i, loss.item())


if __name__ == '__main__':
    overfit_model()

    # samples = get_one_audio_segment(n_samples, device='cpu')
    # samples = samples.view(1, 1, n_samples).to('cpu')
    # print(samples.device)
    #
    # model = SpikingModel(n_channels=128, filter_size=128, periodicity_size=256)
    # binary = model.forward(samples)
    # active = (binary > 0).sum()
    # sparsity = active / binary.numel()
    # print(binary.shape, binary.max().item(), binary.min().item(), sparsity, active.item())
