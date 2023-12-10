# import zounds
from config.dotenv import Config
import numpy as np
from util import device
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from data.datastore import batch_stream


def windowed_audio(audio_batch, window_size, step_size):
    audio_batch = F.pad(audio_batch, (0, step_size))
    windowed = audio_batch.unfold(-1, window_size, step_size)
    window = torch.hann_window(window_size).to(audio_batch.device)
    return windowed * window

def preprocess(batch, n_samples):
    batch /= (np.abs(batch).max() + 1e-12)
    batch = torch.from_numpy(batch).to(device).reshape(-1, n_samples)
    return batch    

def spec_process(batch):
    windowed = windowed_audio(batch, 512, 256)
    spec = torch.fft.rfft(windowed, dim=-1, norm='ortho')
    mag = torch.abs(spec)
    phase = spec.imag
    # phase = torch.angle(spec)

    # phase = phase.data.cpu().numpy()
    # phase = np.unwrap(phase, axis=1)
    # phase = torch.from_numpy(phase).to(spec.device)

    # phase = torch.diff(
    #     phase,
    #     dim=1,
    #     prepend=torch.zeros(spec.shape[0], 1, spec.shape[-1]).to(spec.device))

    # freqs = torch.fft.rfftfreq(512) * 2 * np.pi
    # phase = phase - freqs[None, None, :]

    spec = torch.cat([mag[..., None], phase[..., None]], dim=-1)
    return spec


def loss_func(a, b):
    a = spec_process(a)
    b = spec_process(b)
    return F.mse_loss(a[..., 0], b[..., 0]) + F.mse_loss(a[..., 1], b[..., 1])


def stream(batch_size, n_samples, overfit=False):
    bs = batch_stream(
        Config.audio_path(),
        '*.wav',
        batch_size,
        n_samples)

    batch = preprocess(next(bs), n_samples)
    while True:
        yield batch
        if not overfit:
            batch = preprocess(next(bs), n_samples)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(torch.FloatTensor(1, 2**15).normal_(0, 0.1))
    
    def forward(self):
        return self.params