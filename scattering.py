from pickletools import optimize
import numpy as np
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from data import audio_stream
from modules.ddsp import band_filtered_noise
from modules.overfitraw import OverfitRawAudio
from train.optim import optimizer
from util import playable


sr = zounds.SR22050()
n_samples = 2**14
noise_step = 256
noise_win = noise_step * 2
n_frames = n_samples // noise_step
n_filters = 16

band = zounds.FrequencyBand(40, sr.nyquist)
scale = zounds.MelScale(band, n_filters)
fb = zounds.learn.FilterBank(
    sr, 512, scale, 0.1, normalize_filters=True, a_weighting=False)

def feature(x):

    # convolve and modulus
    x = fb.convolve(x)[..., :n_samples]

    # first order coefficients
    mean = F.avg_pool1d(torch.abs(x), 1024, 1, padding=512)[..., :n_samples]

    # remove the mean
    residual = x - mean

    residual = residual.view(n_filters, 1, -1)

    second = F.conv1d(residual, fb.filter_bank, stride=1, padding=256)
    second = second.view(1, -1, second.shape[-1])

    second = F.avg_pool1d(torch.abs(second), 1024, 1, padding=512)[..., :n_samples]

    return mean, residual, second

def loss(inp, target):
    inp_feat = feature(inp)
    target_feat = feature(target)
    return F.mse_loss(inp_feat, target_feat)

def make_signal():
    sine_freq = 220 / sr.nyquist
    sine = torch.zeros(n_samples).fill_(sine_freq)
    sine = torch.sin(torch.cumsum(sine, dim=-1) * np.pi)
    sine = sine / torch.abs(sine.max())

    noise_freq = 440 / sr.nyquist
    mean = torch.zeros(1, 1, n_frames).fill_(noise_freq)
    std = torch.zeros(1, 1, n_frames).fill_(0.001)
    noise = band_filtered_noise(
        n_samples, ws=noise_win, step=noise_step, mean=mean, std=std)
    noise = noise / torch.abs(noise.max())
    
    print(sine.max().item(), noise.max().item())

    signal = sine + noise
    return signal

model = OverfitRawAudio((1, 1, n_samples), std=0.1)
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    # stream = audio_stream(1, n_samples, overfit=True, normalize=True, as_torch=True)

    # audio = next(stream)

    audio = make_signal().view(1, 1, n_samples)

    spec, residual, second = feature(audio)

    spec = spec.data.cpu().numpy().squeeze()
    residual = residual.data.cpu().numpy().squeeze()
    second = second.data.cpu().numpy().squeeze()

    # def listen():
    #     return playable(current, sr)

    def real():
        return playable(audio, sr)
    
    # while True:
    #     optim.zero_grad()
    #     current = model.forward(None)
    #     l = loss(current, audio)
    #     l.backward()
    #     optim.step()
    #     print(l.item())

    input('Waiting...')