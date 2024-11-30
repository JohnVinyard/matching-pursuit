import torch
from torch import nn
from torch.optim import Adam
from conjure import loggers, serve_conjure, LmdbCollection
from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import stft, sparsify, sparsify2, gammatone_filter_bank, fft_frequency_decompose, pos_encoded, \
    hard_pos_encoding
from modules.latent_loss import normalized_covariance, covariance
from modules.normal_pdf import pdf2
from modules.overfitraw import OverfitRawAudio
from modules.transfer import fft_convolve
from util import device
from torch.nn import functional as F
from itertools import count
import numpy as np
from kymatio.torch import Scattering1D
from torch.distributions import Normal

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

n_samples = 2 ** 16
transform_window_size = 2048
transform_step_size = 256


def stft_transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, -1, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


class MeanSquaredError(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, target)


class ScatteringLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.scat = Scattering1D(J=6, shape=n_samples, Q=16)
        meta = self.scat.meta()

        self.order_0_indices = np.where(meta['order'] == 0)
        self.order_1_indices = np.where(meta['order'] == 1)
        self.order_2_indices = np.where(meta['order'] == 2)

        print(self.order_0_indices)
        print(self.order_1_indices)
        print(self.order_2_indices)

    def _forward(self, audio: torch.Tensor):
        print(audio.shape)
        x = self.scat(audio)
        print(x.shape, x.dtype)

        # zero = x[self.order_0_indices]
        # one = x[self.order_1_indices]
        # two = x[self.order_2_indices]
        #
        # print(zero.shape, zero.dtype)
        # print(one.shape, one.dtype)
        # print(two.shape, two.dtype)

        return x

    def forward(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        t = self._forward(target)
        r = self._forward(recon)
        l = torch.abs(t - r).sum()
        return l



class HingeyTypeLoss(nn.Module):

    def __init__(self, n_elements: int = 256):
        super().__init__()
        self.n_elements = n_elements

    def forward(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        batch, _, time = target.shape

        t_spec = stft_transform(target).reshape(batch, -1)
        r_spec = stft_transform(recon).reshape(batch, -1)
        residual = t_spec - r_spec
        noise_spec = torch.zeros_like(residual).normal_(residual.mean().item(), residual.std().item())

        target_norm = torch.norm(t_spec, dim=-1, keepdim=True)
        recon_norm = torch.norm(r_spec, dim=-1, keepdim=True)

        # ensure that the norm does not grow
        norm_loss = torch.clip(recon_norm - target_norm, min=0, max=np.inf).sum()

        # choose a random subset of the indices
        indices = torch.randperm(t_spec.shape[-1], device=device)[:self.n_elements]

        t_spec = t_spec[:, indices]
        r_spec = r_spec[:, indices]
        residual = t_spec - r_spec
        n_spec = noise_spec[:, indices]

        # The residual covariance should resemble/move toward noise
        t_cov = covariance(n_spec)
        r_cov = covariance(residual)

        cov_loss = torch.abs(t_cov - r_cov).sum()

        return norm_loss + cov_loss


def pos_encoding(n_elements: int, n_sinusoids: int) -> torch.Tensor:
    t = torch.linspace(1e-8, 1, n_elements)
    freq = torch.linspace(0.001, 1, n_sinusoids)
    ps = torch.sin(t[:, None] * freq[None, :])
    return ps

    # r = pdf2(t[:, None], t[None, :], n_elements=n_sinusoids)
    # print(r.shape)
    # return r


class SparseLossFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_size = 64
        self.n_filters = 64
        self.n_frames = 64

        f = gammatone_filter_bank(self.n_filters, self.filter_size, device=device, band_spacing='linear')
        self.filters = nn.Parameter(f)

        self.proj_time = nn.Parameter(pos_encoding(self.n_frames, 128))
        self.proj_freq = nn.Parameter(pos_encoding(self.n_filters, 128))

        # self.proj_time = nn.Parameter(torch.zeros(self.n_frames, 128).uniform_(-1, 1))
        # self.proj_freq = nn.Parameter(torch.zeros(self.n_filters, 128).uniform_(-1, 1))

    def forward(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        t = self._forward(target)
        r = self._forward(recon)
        return torch.abs(t - r).mean()

    def _forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = fft_frequency_decompose(x, min_size=512)
        results = []
        for size, band in bands.items():
            batch, _, samples = band.shape
            filters = F.pad(self.filters[None, :, :], (0, samples - self.filter_size))
            result = fft_convolve(band, filters)
            stride = samples // 64
            step = stride * 2
            pooled = F.max_pool1d(result, step, stride=stride, padding=stride // 2)[..., :samples]
            sparse, packed, one_hot = sparsify2(pooled, n_to_keep=128)
            a = packed @ self.proj_time
            b = one_hot @ self.proj_freq
            result = torch.cat((a, b), dim=-1)

            results.append(result)

        result = torch.cat(results, dim=-1)
        return result


def train(n_samples: int = 2 ** 16):
    target = get_one_audio_segment(n_samples=n_samples, device=device)

    # loss_model = SparseLossFeature().to(device)
    # loss_model = CorrelationLoss(n_elements=512)
    # loss_model = MeanSquaredError()
    loss_model = HingeyTypeLoss()
    # loss_model = ScatteringLoss().to(device)

    model = OverfitRawAudio(shape=(1, 1, n_samples), std=1e-3, normalize=False).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    collection = LmdbCollection(path='noise')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=8888, n_workers=1)

    orig_audio(target)

    for i in count():
        optim.zero_grad()
        recon = model.forward(None)
        recon_audio(recon)
        loss = loss_model.forward(target, recon)
        loss.backward()
        optim.step()
        print(i, loss.item())


if __name__ == '__main__':
    train(n_samples=n_samples)

