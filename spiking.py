from typing import Dict

import torch
from openpyxl.styles.builtins import total
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from data import get_one_audio_segment, AudioIterator
from modules import gammatone_filter_bank, max_norm, unit_norm, fft_frequency_decompose, stft, sparsify
from modules.anticausal import AntiCausalAnalysis
from modules.mixer import MixerStack
from modules.overfitraw import OverfitRawAudio
from modules.overlap_add import overlap_add
from modules.transfer import fft_convolve
from conjure import LmdbCollection, Logger, loggers, serve_conjure
from util import device, encode_audio, make_initializer
from itertools import count

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt
import numpy as np

n_samples = 2 ** 17

init = make_initializer(0.02)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        # self.ln = torch.nn.LayerNorm(dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        orig_shape = input.shape
        batch_size = orig_shape[0]
        input = input.view(batch_size, -1)
        mx, _ = torch.max(input, dim=-1, keepdim=True)
        input = input / (mx + 1e-8)
        x = input.view(*orig_shape)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels: int, bottleneck_channels:int):
        super().__init__()

        self.bottleneck_channels = bottleneck_channels
        self.channels = channels

        self.window_size = 512
        self.step_size = 256
        self.n_frames = n_samples // self.step_size
        self.n_coeffs = self.window_size // 2 + 1

        self.start_size = self.n_frames

        self.encoder = AntiCausalAnalysis(
            1,
            self.channels,
            2,
            [1, 2, 4, 8, 16, 32, 1],
            # with_activation_norm=True,
            do_norm=True)

        self.up_proj = nn.Conv1d(self.channels, self.bottleneck_channels, 1, 1, 0)
        self.down_proj = nn.Conv1d(self.bottleneck_channels, self.channels, 1, 1, 0)


        self.decoder = AntiCausalAnalysis(
            self.channels,
            self.channels,
            2,
            [1, 2, 4, 8, 16, 32, 1],
            reverse_causality=True,
            # with_activation_norm=True,
            do_norm=True)


        self.to_samples = nn.Conv1d(self.channels, 1, 1, 1, 0)

        self.apply(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        x = self.up_proj(x)
        x = self.down_proj(x)
        x = self.decoder(x)
        x = self.to_samples(x)
        return x


class SomethingSomething(nn.Module):
    def __init__(self, window_size: int, step_size: int, desired_frames: int):
        super().__init__()
        self.window_size = window_size
        self.n_coeffs = self.window_size // 2 + 1
        self.step_size = step_size
        self.desired_frames = desired_frames

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        y = self.forward(y)
        return torch.abs(x - y).sum()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        x = fft_frequency_decompose(x, 512)

        bands = []

        for size, band in x.items():
            spec = stft(band, self.window_size, self.step_size, pad=True)
            spec = spec.view(batch, -1, self.n_coeffs).permute(0, 2, 1)
            f = spec.shape[-1]
            if f < self.desired_frames:
                spec = F.upsample(spec, size=self.desired_frames, mode='linear')
            elif f == self.desired_frames:
                spec = spec
            else:
                step = f // self.desired_frames
                spec = F.max_pool1d(spec, kernel_size=step * 2, stride=step, padding=step)[..., :self.desired_frames]

            bands.append(spec)

        bands = torch.cat(bands, dim=1)
        return bands


class DecayLoss(nn.Module):
    def __init__(self, n_samples, n_decays: int, min_decay: float, max_decay: float, window_size: int):
        super().__init__()
        self.n_samples = n_samples
        self.n_decays = n_decays
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.window_size = window_size
        self.step_size = window_size // 2
        self.n_coeffs = self.window_size // 2 + 1
        self.n_frames = n_samples // self.step_size

        base = torch.linspace(1, 0, self.n_frames)[None, :]
        decays = torch.linspace(min_decay, max_decay, n_decays)[:, None]

        decays = base ** decays

        decays = decays.view(1, 1, self.n_decays, self.n_frames)
        decays = unit_norm(decays)

        self.register_buffer('decays', decays, persistent=False)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        x = self.forward(x)
        y = self.forward(y)
        return torch.abs(x - y).sum()

    def forward(self, x):
        batch = x.shape[0]

        x = stft(x, ws=self.window_size, step=self.step_size, pad=True).permute(0, 1, 3, 2)


        x = fft_convolve(x[:, :, :, None, :], self.decays[:, :, None, :, :])
        x = x.view(batch, -1, self.n_frames)

        kernel_size = 16

        pooled = F.avg_pool1d(F.pad(x, [kernel_size, 0]), kernel_size=kernel_size, stride=1, padding=0)[..., :self.n_frames]


        # print(x.shape, pooled.shape)


        x = x - pooled
        x = torch.relu(x)


        # x = x.view(batch, )

        # plt.matshow(np.abs(x.data.cpu().numpy()[0, :, ::512]))
        # plt.show()

        return x


class SpikingModel(nn.Module):
    def __init__(self, n_channels, filter_size, periodicity_size, memory_size=512, frame_memory_size=8):
        super().__init__()
        self.n_channels = n_channels
        self.filter_size = filter_size
        self.periodicity_size = periodicity_size

        gfb = gammatone_filter_bank(
            n_filters=self.n_channels,
            size=self.filter_size,
            device='cpu',
            band_spacing='linear').view(1, self.n_channels, self.filter_size)
        gfb = unit_norm(gfb)

        self.register_buffer('gammatone', gfb, persistent=False)

        self.memory_size = memory_size
        self.periodicity_memory_size = frame_memory_size

        memory = (torch.linspace(0, 1, steps=self.memory_size))[None, :]
        decay = torch.linspace(1.1, 10, steps=n_channels)[:, None]
        memory = memory ** decay
        memory /= memory.sum(dim=-1, keepdim=True)
        self.register_buffer('memory', memory, persistent=False)


    def multiband(self, audio: torch.Tensor, hard: bool = False, normalize: bool = True) -> Dict[int, torch.Tensor]:

        # audio = torch.cat([torch.zeros_like(audio).uniform_(-1, 1), audio], dim=-1)
        bands = fft_frequency_decompose(audio, 512)
        bands = {size: self.forward(band, hard=hard, normalize=normalize) for size, band in bands.items()}
        return bands

    def compute_multiband_loss(self, target: torch.Tensor, recon: torch.Tensor, hard: bool = False, normalize=True) -> torch.Tensor:
        loss = 0
        target_bands = self.multiband(target, hard=hard, normalize=normalize)
        recon_bands = self.multiband(recon, hard=hard, normalize=normalize)

        for size, band in target_bands.items():
            loss = loss + torch.abs(band - recon_bands[size]).sum()
        return loss

    def compute_loss(self, target: torch.Tensor, recon: torch.Tensor, hard: bool = True, normalize: bool = True):
        t = self.forward(target, hard=hard, normalize=normalize)
        r = self.forward(recon, hard=hard, normalize=normalize)
        loss = torch.abs(t - r).sum()
        return loss

    def compute_masked_loss(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        t = self.forward(target)
        r = self.forward(recon)

        mask = t == 1

        loss = torch.abs(t[mask] - r[mask]).sum()
        return loss

    def compute_sum_loss(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        t = self.forward(target)
        r = self.forward(recon)

        a = torch.abs(t - r).sum()
        b = torch.abs(t.sum() - r.sum()).sum() * 0.25
        return a + b


    def forward(self, audio: torch.Tensor, hard: bool = True, normalize: bool = True):

        audio_size = audio.numel()

        batch = audio.shape[-1]

        n_samples = audio.shape[-1]
        audio = audio.view(-1, 1, n_samples)

        # convolve with gammatone filters
        g = F.pad(self.gammatone, (0, n_samples - self.filter_size))
        channels = fft_convolve(audio, g)

        # half-wave rectification
        channels = torch.relu(channels)

        # compression
        # channels = torch.log(channels + 1e-8)
        # channels = channels ** 2

        if normalize:
            m = F.pad(self.memory, (0, n_samples - self.memory_size))
            pooled = fft_convolve(m, channels)
            normalized = channels - pooled
            normalized = torch.relu(normalized)
        else:
            normalized = channels


        if not hard:
            y = normalized
        else:
            fwd = (normalized > 0).float()
            back = normalized
            # layer one of spiking response.  Unit responses propagate forward,
            # initial real-values propagate backward
            y = back + (fwd - back).detach()



        # compute periodicity
        y = F.pad(y, (0, self.periodicity_size // 4))
        y = y.unfold(-1, self.periodicity_size, self.periodicity_size // 4)
        y = torch.abs(torch.fft.rfft(y, dim=-1))

        values, indices = torch.topk(y, k=8, dim=-1)

        z = torch.zeros_like(y)
        z = torch.scatter(z, dim=-1, index=indices, src=values)


        fwd = z
        back = y
        # layer two of spiking response.  Sparse periodicities
        # propagate forward, full spectrum propagates back
        y = back + (fwd - back).detach()

        total_spikes = (fwd > 0).sum()

        # print(y.shape, y.numel(), total_spikes, total_spikes.item() / y.numel())
        ratio = total_spikes / audio_size

        # print(f'Band {audio.shape[-1]}: {ratio:.2f}', total_spikes, fwd.shape)

        return y


class HyperDimensionalLoss(nn.Module):
    def __init__(
            self,
            window_size: int = 2048,
            step_size: int = 256,
            hdim: int = 16384):
        super().__init__()
        self.window_size = window_size
        self.n_coeffs = self.window_size // 2 + 1
        self.step_size = step_size
        self.hdim = hdim

        proj = torch.zeros(self.n_coeffs, self.hdim).uniform_(-3, 3)
        self.register_buffer('proj', proj)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        y = self.forward(y)
        return torch.abs(x - y).sum()

    def forward(self, x):
        batch_size = x.shape[0]
        x = stft(x, self.window_size, self.step_size, pad=True)
        x = x.view(batch_size, -1, self.n_coeffs).permute(0, 2, 1)
        x = self.proj.T @ x
        x = torch.tanh(x)

        items = []

        for i in range(x.shape[-1]):
            item = x[:, :, i: i + 1]
            torch.roll(item, shifts=i, dims=1)
            items.append(item)
            # x[:, :, i: i + 1] = item

        x = torch.cat(items, dim=-1)
        x = torch.sum(x, dim=-1)

        # back = x
        # fwd = torch.sign(x)
        # x = back + (fwd - back).detach()
        return x


class AutocorrelationLoss(nn.Module):
    def __init__(self, n_channels, filter_size):
        super().__init__()
        self.n_channels = n_channels
        self.filter_size = filter_size

        gfb = gammatone_filter_bank(
            n_filters=self.n_channels,
            size=self.filter_size,
            device='cpu',
            band_spacing='linear').view(
            1,
            self.n_channels,
            self.filter_size)

        gfb = unit_norm(gfb)

        self.register_buffer('gammatone', gfb, persistent=False)

    def multiband_forward(self, audio: torch.Tensor, window_size: int, step_size: int):
        bands = fft_frequency_decompose(audio, 512)
        return {k: self.forward(v, window_size, step_size) for k, v in bands.items()}

    def compute_multiband_loss(
            self,
            target: torch.Tensor,
            recon: torch.Tensor,
            window_size: int,
            step_size: int):
        tb = self.multiband_forward(target, window_size, step_size)
        rb = self.multiband_forward(recon, window_size, step_size)
        loss = 0
        print('=======================')
        for k, v in rb.items():
            loss = loss + torch.abs(v - tb[k]).sum()
        return loss

    def forward(self, audio: torch.Tensor, window_size: int = 128, step_size: int = 64):
        batch = audio.shape[0]

        n_samples = audio.shape[-1]
        audio = audio.view(-1, 1, n_samples)

        # convolve with gammatone filters
        g = F.pad(self.gammatone, (0, n_samples - self.filter_size))
        channels = fft_convolve(audio, g)

        # half-wave rectification
        channels = torch.relu(channels)
        # channels = torch.log(channels + 1e-8) + 27

        channels = F.pad(channels, (0, step_size))
        channels = channels.unfold(-1, window_size, step_size)
        # channels = unit_norm(channels, dim=-1)

        spec = torch.fft.rfft(channels, dim=-1)
        # spec = torch.abs(spec)

        # within-channel correlation
        corr = spec[:, :, :, 1:] * spec[:, :, :, :-1]
        # corr = torch.fft.irfft(corr, dim=-1)
        corr = torch.abs(corr)
        # corr = unit_norm(corr, dim=-1)
        # corr = torch.relu(torch.log(corr + 1e-8) + 27)

        # neighboring channel correlation
        corr2 = spec[:, :, 1:, :] * spec[:, :, :-1, :]
        # corr2 = torch.fft.irfft(corr2, dim=-1)
        corr2 = torch.abs(corr2)
        # corr2 = unit_norm(corr2, dim=-1)
        # corr2 = torch.relu(torch.log(corr2 + 1e-8) + 27)

        x = torch.cat([corr.view(-1), corr2.view(-1)])
        return x

    def compute_loss(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        t = self.forward(target)
        r = self.forward(recon)
        return torch.abs(t - r).sum()


def train_resource_constrained_autoencoder():

    loss_model = DecayLoss(
        n_samples=n_samples,
        n_decays=16,
        min_decay=2,
        max_decay=32,
        window_size=64).to(device)


    ae = AutoEncoder(channels=32, bottleneck_channels=32).to(device)

    optim = Adam(ae.parameters(), lr=1e-3)
    collection = LmdbCollection('spiking')

    # TODO: Figure out how to set up a server prior to using logger
    recon_audio, orig_audio = loggers(
        ['recon', 'orig'],
        'audio/wav',
        encode_audio,
        collection)

    # orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=9999, n_workers=1)

    stream = AudioIterator(
        batch_size=2,
        n_samples=n_samples,
        samplerate=22050,
        normalize=True,
        overfit=False)

    for i, item in enumerate(iter(stream)):
        optim.zero_grad()
        target = item.view(-1, 1, n_samples)
        recon = ae.forward(target)


        loss = loss_model.compute_loss(target, recon)

        loss.backward()
        optim.step()
        orig_audio(target)
        recon_audio(max_norm(recon))
        print(i, loss.item())


def overfit_model():
    target = get_one_audio_segment(n_samples).to(device).view(1, 1, n_samples)
    target = max_norm(target)

    loss_model = DecayLoss(
        n_samples=n_samples,
        n_decays=16,
        min_decay=2,
        max_decay=32,
        window_size=512).to(device)

    # loss_model = SomethingSomething(64, 16, 512).to(device)

    # loss_model = SpikingModel(
    #     n_channels=64,
    #     filter_size=64,
    #     periodicity_size=64,
    #     memory_size=64,
    #     frame_memory_size=64).to(device)


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

    # target_features = loss_model.forward(target)

    for i in count():
        optim.zero_grad()
        recon = overfit_model.forward(None)
        recon = max_norm(recon)
        recon_audio(max_norm(recon))

        # ae.forward(target)

        loss = loss_model.compute_loss(target, recon)
        # loss = loss_model.compute_multiband_loss(target, recon, hard=True, normalize=True)

        # loss = torch.abs(target_features - recon_features).sum()
        loss.backward()
        # clip_grad_value_(overfit_model.parameters(), 0.1)
        optim.step()

        print(i, loss.item())


if __name__ == '__main__':
    # overfit_model()
    train_resource_constrained_autoencoder()
