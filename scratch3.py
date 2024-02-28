from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from data.audioiter import AudioIterator
from modules import stft
from modules.fft import fft_convolve
from modules.infoloss import SpectralInfoLoss
from modules.latent_loss import normalized_covariance
from modules.overlap_add import overlap_add

from matplotlib import pyplot as plt

from modules.refractory import make_refractory_filter
from modules.sparse import sparsify
from util.music import musical_scale, musical_scale_hz
import zounds

def convolve_spectrograms(batch: torch.Tensor, channels: torch.Tensor):
    """
    
    """
    b, time, ch = batch.shape
    
    padding = torch.zeros_like(batch)
    
    bias_start = torch.linspace(2, 1, steps=time, device=batch.device)
    bias_end = torch.linspace(1, 2, steps=time, device=batch.device)
    bias_mid = torch.ones_like(bias_start)
    bias = torch.cat([bias_start, bias_mid, bias_end])
    
    batch_padded = torch.cat([padding, batch, padding], dim=1)
    channels_padded = torch.cat([channels, padding, padding], dim=1)
    
    norm = torch.norm(channels_padded, dim=(1, 2), keepdim=True)
    channels_padded = channels_padded / (norm + 1e-12)
    
    batch_spec = torch.fft.rfft(batch_padded, dim=1, norm='ortho')
    channels_spec = torch.fft.rfft(channels_padded, dim=1, norm='ortho')
    conv = batch_spec * channels_spec
    
    fm = torch.fft.irfft(conv, dim=1, norm='ortho')
    plt.matshow(fm[0])
    plt.show()
    
    print(fm.shape)
    
    fm = fm.norm(dim=-1)
    plt.plot(fm[0])
    plt.show()
    
    
    biased = fm * bias[None, :]
    
    
    values, indices = torch.max(biased, dim=-1, keepdim=True)
    
    
    amps = torch.gather(fm, dim=-1, index=indices)
    
    # print(values.shape, indices.shape, amps.shape)
    
    
    residual = batch_padded.clone()
    
    workspace = torch.zeros_like(batch_padded)
    
    
    for i in range(b):
        index = indices[i]
        amp = fm[i, index: index + 1]
        workspace[i, index: index + time, :] = channels[i, :, :] * amp
    
    residual = residual - workspace
    
    residual = residual[:, time:time*2, :]
    
    
    plt.matshow(residual[0])
    plt.show()
    
    return amps, indices - 128, residual
    
    
    
    
    
    



# TODO: try matrix rotation instead: https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
def to_polar(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    return mag, phase

def to_complex(mag, phase):
    return mag * torch.exp(1j * phase)

def advance_one_frame(x):
    mag, phase = to_polar(x)
    phase = phase + torch.linspace(0, np.pi, x.shape[-1])[None, None, :]
    x = to_complex(mag, phase)
    return x


def test():

    n_samples = 2 ** 15
    window_size = 1024
    step_size = window_size // 2
    n_coeffs = window_size // 2 + 1
    
    impulse = torch.zeros(1, 1, 2048).uniform_(-1, 1)
    impulse = F.pad(impulse, (0, n_samples - 2048))
    windowed = windowed_audio(impulse, window_size, step_size)
    
    n_frames = windowed.shape[-2]
    
    transfer_func = torch.zeros(1, n_coeffs).uniform_(0, 0.5)
    
    
    frames = []
    
    for i in range(n_frames):
        
        if i == 0:
            spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
            spec = spec * transfer_func
            audio = torch.fft.irfft(spec, dim=-1)
            frames.append(audio)
        else:
            prev = frames[i - 1]
            prev_spec = torch.fft.rfft(prev, dim=-1)
            prev_spec = advance_one_frame(prev_spec)
            
            current_spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
            spec = current_spec + prev_spec
            spec = spec * transfer_func
            audio = torch.fft.irfft(spec, dim=-1)
            frames.append(audio)
    
    
    frames = torch.cat([f[:, :, None, :] for f in frames], dim=2)
    audio = overlap_add(frames, apply_window=True)[..., :n_samples]
    
    return audio


def anticausal_inhibition(x: torch.Tensor, inhibitory_area: Tuple[int, int]):
    batch, channels, time = x.shape
    width, height = inhibitory_area
    x = x[:, None, :, :]
    mx = F.max_pool2d(x, inhibitory_area, stride=(1, 1), padding=(width // 2, height // 2))
    x = x - (mx * 0.9)
    x = torch.relu(x)
    return x.view(batch, channels, time)
    

def sparsify_test(iterations: int = 10):
    
    start_shape = (1, 128, 128)
    start = torch.zeros(*start_shape).uniform_(-10, 10)
    start = torch.relu(start)
    
    
    for i in range(iterations):
        start = anticausal_inhibition(start, (55, 3))
        print(start.shape)
        print(f'iteration {i}, non-zero: {(start > 0).sum().item()}, max value: {start.max().item()}')
        display = start.data.cpu().numpy().reshape(*start_shape[1:])
        plt.matshow(display)
        plt.show()
    

def gumbel_softmax_test():
    t = torch.zeros(128).normal_(0, 1)
    t = F.gumbel_softmax(t, tau=0.1, hard=False)
    t = t.data.cpu().numpy()
    plt.plot(t)
    plt.show()


if __name__ == '__main__':
    
    bad_scheduling = torch.zeros(16, 128)
    bad_scheduling[:, 10] = 1
    
    good_scheduling = torch.zeros(16, 128).bernoulli_(p=0.125)
    
    bad_cov = normalized_covariance(bad_scheduling)
    good_cov = normalized_covariance(good_scheduling)
    
    print(bad_cov.item(), good_cov.item())
    
    
    # signal = torch.zeros(8, 1, 128).uniform_(0, 1)
    # signal = sparsify(signal, n_to_keep=16)
    
    # nz = signal > 0
    
    # signal = torch.zeros(1, 1, 2**15).uniform_(-1, 1)
    
    # sil = SpectralInfoLoss(2048, 256, patch_size=(16, 16), patch_step=(8, 8), embedding_channels=32)
    # one_hot, codes, weights, norms, normed, raw = sil.encode(signal)
    # print(codes)
    # print(weights)
    
    # print(one_hot.shape)
    # print(codes.shape)
    # print(weights.shape)
    
    
    
    
    # scale = [band.center_frequency for band in musical_scale(21, 109)]
    # print(scale)
    
    # scale = musical_scale_hz(21, 106, 512)
    # f0s = np.linspace(27, 4000, len(scale))
    
    # plt.plot(scale)
    # plt.plot(f0s)
    # plt.show()