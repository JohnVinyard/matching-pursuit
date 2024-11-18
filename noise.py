import torch
from torch import nn
from torch.optim import Adam
from conjure import loggers, serve_conjure, LmdbCollection
from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import stft, sparsify, sparsify2, gammatone_filter_bank
from modules.latent_loss import normalized_covariance, covariance
from modules.overfitraw import OverfitRawAudio
from modules.transfer import fft_convolve
from util import device
from torch.nn import functional as F
from itertools import count
import numpy as np

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


class CorrelationLoss(nn.Module):
    def __init__(self, n_elements: int = 256):
        super().__init__()
        self.n_elements = n_elements

    def forward(self, target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        batch, _, time = target.shape
        # noise = torch.zeros_like(target).uniform_(-1, 1)

        t_spec = stft_transform(target).reshape(batch, -1)
        r_spec = stft_transform(recon).reshape(batch, -1)
        # noise_spec = stft_transform(noise).reshape(batch, -1)

        indices = torch.randperm(t_spec.shape[-1], device=device)[:self.n_elements]

        t_spec = t_spec[:, indices]
        r_spec = r_spec[:, indices]
        # noise_spec = noise_spec[:, indices]

        print(t_spec.norm(), r_spec.norm())


        t_cov = covariance(t_spec)
        r_cov = covariance(r_spec)

        # t_cov = torch.triu(t_cov)
        # r_cov = torch.triu(r_cov)

        # noise_cov = covariance(noise_spec)

        return torch.abs(t_cov - r_cov).sum()


class SparseLossFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_size = 1024
        self.n_filters = 512
        f = gammatone_filter_bank(self.n_filters, self.filter_size, device=device, band_spacing='linear')
        self.filters = nn.Parameter(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time = x.shape
        filters = F.pad(self.filters[None, :, :], (0, n_samples - self.filter_size))
        result = fft_convolve(x, filters)
        result = torch.relu(result)
        pooled = F.avg_pool1d(result, 512, stride=1, padding=256)[..., :n_samples]
        result = result - pooled
        result = torch.relu(result)

        plt.matshow(np.flipud(result.data.cpu().numpy()[0, :, :4096]))
        plt.show()

        return result



def train(n_samples: int = 2 ** 16):
    target = get_one_audio_segment(n_samples=n_samples, device=device)
    # loss_model = SparseLossFeature().to(device)
    loss_model = CorrelationLoss(n_elements=512)

    model = OverfitRawAudio(shape=(1, 1, n_samples), std=0.1, normalize=True).to(device)
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
