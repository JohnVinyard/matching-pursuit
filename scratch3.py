from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from data.audioiter import AudioIterator
from modules import stft
from modules.fft import fft_convolve
from modules.overlap_add import overlap_add

from matplotlib import pyplot as plt

from modules.refractory import make_refractory_filter


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
    # audio = test()
    # print(audio.shape)
    
    # audio = playable(audio.squeeze(), zounds.SR22050(), normalize=True)
    # audio.save('test.wav')
    
    # inp = torch.zeros(4, 16, 128).uniform_(0, 1)
    
    # net = AntiCausalStack(16, 2, dilations=[1, 2, 4, 8, 16, 32, 64])
    
    # result = net.forward(inp)
    
    # result = result.data.cpu().numpy()[0]
    
    # plt.matshow(result)
    # plt.show()
    
    
    # sparsify_test(5)
    # gumbel_softmax_test()
    
    gaussian = torch.hamming_window(512) * torch.zeros(512).uniform_(-1, 1)
    
    a1 = torch.zeros(4, 1, 2**15)
    a1[:, :, 2048:2048 + 512] = gaussian[None, None, :]
    
    a2 = torch.zeros(4, 1, 2**15)
    a2[:, :, :512] = gaussian[None, None, :]
    
    spec1 = stft(a1, 2048, 256, pad=True).view(4, 128, -1)
    spec2 = stft(a2, 2048, 256, pad=True).view(4, 128, -1)
    
    
    # print(spec1.shape, spec2.shape)
    
    a, b, c = convolve_spectrograms(spec1, spec2)
    print(a, b, c.shape)
    