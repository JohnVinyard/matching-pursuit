from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from data import get_one_audio_segment, AudioIterator
from modules import gammatone_filter_bank, max_norm, unit_norm, fft_frequency_decompose, stft, sparsify
from modules.anticausal import AntiCausalAnalysis
from modules.infoloss import CorrelationLoss
from modules.mixer import MixerStack
from modules.overfitraw import OverfitRawAudio
from modules.overlap_add import overlap_add
from modules.transfer import fft_convolve
from conjure import LmdbCollection, Logger, loggers, serve_conjure
from util import device, encode_audio, make_initializer
from itertools import count
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from torch.nn.utils import weight_norm

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

        # self.encoder = AntiCausalAnalysis(
        #     self.n_coeffs * 2,
        #     self.channels,
        #     2,
        #     [1, 2, 4, 8, 16, 32, 1],
        #     # with_activation_norm=True,
        #     do_norm=True)
        self.encoder = MixerStack(self.n_coeffs * 2, channels, self.n_frames, 4, 4, channels_last=False)

        self.up_proj = nn.Conv1d(self.channels, self.bottleneck_channels, 1, 1, 0)
        self.down_proj = nn.Conv1d(self.bottleneck_channels, self.channels, 1, 1, 0)


        self.decoder = MixerStack(channels, channels, self.n_frames, 4, 4, channels_last=False)
        # self.decoder = AntiCausalAnalysis(
        #     self.channels,
        #     self.channels,
        #     2,
        #     [1, 2, 4, 8, 16, 32, 1],
        #     reverse_causality=True,
        #     # with_activation_norm=True,
        #     do_norm=True)

        self.to_mag = nn.Linear(self.channels, self.n_coeffs)
        self.to_phase = nn.Linear(self.channels, self.n_coeffs)

        self.register_buffer('group_delay', torch.linspace(0, np.pi, self.n_coeffs))

        # self.to_samples = nn.Conv1d(self.channels, self.n_coeffs * 2, 1, 1, 0)

        self.apply(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        n_samples = x.shape[-1]
        orig = x
        batch_size = x.shape[0]
        x = stft(x, self.window_size, self.step_size, pad=True, return_complex=True)

        # print('ORIG', x.shape)
        x = x.view(batch_size, self.n_frames, self.n_coeffs, 2)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, -1, self.n_frames)

        x = self.encoder(x)
        x = self.up_proj(x)


        # x = F.dropout(x, 0.01)
        # x = x - x.mean()
        # x = torch.relu(x)
        # x = x / (x.sum() + 1e-8)
        # x = sparsify(x, n_to_keep=256)
        x = self.down_proj(x)
        x = self.decoder(x)
        x = x.permute(0, 2, 1)

        # print(x.shape)

        mag = self.to_mag(x)
        phase = torch.tanh(self.to_phase(x))

        b = mag.shape[0]
        mag = torch.abs(mag.view(b, self.n_frames, -1))
        phase = phase.view(b, self.n_frames, -1) * self.group_delay[None, None, :]

        gd = self.group_delay.view(1, 1, -1).repeat(1, self.n_frames, 1)
        phase = gd + (phase * torch.zeros_like(phase).uniform_(-1, 1))

        phase = torch.cumsum(phase, dim=1)

        x = mag * torch.exp(1j * phase)

        x = torch.fft.irfft(x, dim=-1)
        x = overlap_add(x[:, None, :, :])[..., :n_samples]

        # x = self.to_samples(x)
        # x = x.view(batch_size, self.n_coeffs, 2, self.n_frames)
        # x = x.permute(0, 3, 1, 2)
        # x = torch.view_as_complex(x.contiguous())
        # x = torch.fft.irfft(x)
        # x = overlap_add(x[:, None, :, :], apply_window=True)[..., :n_samples]
        # x = torch.tanh(x)
        # x = max_norm(x)
        # x = torch.sin(x)
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
    def __init__(self, n_samples, n_decays: int, min_decay: float, max_decay: float):
        super().__init__()
        self.n_samples = n_samples
        self.n_decays = n_decays
        self.min_decay = min_decay
        self.max_decay = max_decay

        base = torch.linspace(1, 0, n_samples)[None, :]
        decays = torch.linspace(min_decay, max_decay, n_decays)[:, None]

        decays = base ** decays

        decays = decays.view(1, self.n_decays, self.n_samples) * torch.zeros(1, 1, self.n_samples).uniform_(-1, 1)

        decays = unit_norm(decays)

        # plt.matshow(np.abs(decays.data.cpu().numpy()[0, :, ::512]))
        # plt.show()

        self.register_buffer('decays', decays, persistent=False)

    def compute_loss(self, x: torch.Tensor, y: torch.Tensor):
        x = self.forward(x)
        y = self.forward(y)
        return torch.abs(x - y).sum()

    def forward(self, x):
        x = fft_convolve(x, self.decays)

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
        decay = torch.linspace(2, 32, steps=n_channels)[:, None]
        memory = memory ** decay
        memory /= memory.sum(dim=-1, keepdim=True)

        self.register_buffer('memory', memory, persistent=False)

        periodicity_memory = \
            (torch.linspace(0, 1, self.periodicity_memory_size) ** 2) \
                .view(1, 1, self.periodicity_memory_size)

        self.register_buffer('periodicity_memory', periodicity_memory, persistent=False)

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


    def forward(self, audio: torch.Tensor, hard: bool = True, normalize: bool = True):
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



        # print(f'Channel {audio.shape[-1]} with sparsity {(fwd.sum() / fwd.numel())} and {fwd.sum()} non-zero elements')

        # TODO: Figure out mask here

        # compute periodicity
        # y = F.pad(y, (0, self.periodicity_size // 4))
        # y = y.unfold(-1, self.periodicity_size, self.periodicity_size // 4)
        # y = torch.abs(torch.fft.rfft(y, dim=-1))

        #
        # y = y - torch.mean(y, dim=-1, keepdim=True)
        # y = y[:, :, 1:, :] - y[:, :, :-1, :]

        # y = torch.relu(y)
        #
        # fwd = (y > 0).float()
        # back = y
        #
        # # layer one of spiking response.  Unit responses propagate forward,
        # # initial real-values propagate backward
        #
        # y = back + (fwd - back).detach()

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
    loss_model = SpikingModel(
        n_channels=64,
        filter_size=64,
        periodicity_size=64,
        memory_size=64,
        frame_memory_size=8).to(device)

    # loss_model = SomethingSomething(64, 16, 512).to(device)

    # loss_model = DecayLoss(n_samples, 64, 1, 32).to(device)

    # loss_model = CorrelationLoss(n_elements=2048).to(device)
    # loss_model = AutocorrelationLoss(n_channels=64, filter_size=64).to(device)

    ae = AutoEncoder(channels=128, bottleneck_channels=512).to(device)

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

        # target_env = F.max_pool1d(target, kernel_size=512, stride=256, padding=256)
        # recon_env = F.max_pool1d(recon, kernel_size=512, stride=256, padding=256)
        # env_loss = torch.abs(target_env - recon_env).sum()
        # loss = loss_model.compute_multiband_loss(target, recon, 64, 16) #+ env_loss

        # loss = loss_model.multiband_noise_loss(target, recon, window_size=64, step=16)

        # loss = loss_model.compute_loss(target, recon)

        loss = loss_model.compute_multiband_loss(target, recon)

        # loss = torch.abs(
        #     stft(target, 2048, 256, pad=True) \
        #     - stft(recon, 2048, 256, pad=True)).sum()
        loss.backward()
        optim.step()
        orig_audio(target)
        recon_audio(max_norm(recon))
        print(i, loss.item())


def overfit_model():
    target = get_one_audio_segment(n_samples).to(device).view(1, 1, n_samples)
    target = max_norm(target)

    # loss_model = DecayLoss(n_samples, 64, 1, 32).to(device)

    # loss_model = SomethingSomething(64, 16, 512).to(device)

    loss_model = SpikingModel(
        n_channels=64,
        filter_size=64,
        periodicity_size=64,
        memory_size=64,
        frame_memory_size=64).to(device)

    # ae = AutoEncoder(2048, 256, 32).to(device)

    # loss_model = AutocorrelationLoss(n_channels=64, filter_size=64).to(device)

    loss_model = HyperDimensionalLoss().to(device)

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
        # loss = loss_model.compute_multiband_loss(target, recon)

        # loss = torch.abs(target_features - recon_features).sum()
        loss.backward()
        # clip_grad_value_(overfit_model.parameters(), 0.1)
        optim.step()

        print(i, loss.item())


if __name__ == '__main__':
    overfit_model()
    # train_resource_constrained_autoencoder()
