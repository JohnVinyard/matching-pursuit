import torch
import zounds
from torch import nn
from modules.pos_encode import pos_encoded
from upsample import FFTUpsampleBlock
from util import device
from torch.nn import functional as F
import numpy as np
from torch.nn.utils import weight_norm

from data.audiostream import audio_stream
from train.optim import optimizer
from util import playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2**14
total_steps = 100
variance = 0.2
samplerate = zounds.SR22050()

init_weights = make_initializer(0.05)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding)
        # self.norm = nn.BatchNorm1d(out_channels)
        # self.norm = nn.InstanceNorm1d(out_channels)
        self.nl = nn.LeakyReLU(0.2)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.conv(x)
        # x = self.norm(x)
        x = self.nl(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, size, do_nl=True, do_bn=True):
        super().__init__()

        self.size = size
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size, stride, padding)

        # self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        # self.norm = nn.BatchNorm1d(out_channels)
        # self.norm = nn.InstanceNorm1d(out_channels)
        self.nl = nn.LeakyReLU(0.2)
        self.do_nl = do_nl
        self.do_bn = do_bn

    def forward(self, x):
        x = self.up(x)

        x = self.conv(x)
        # if self.do_bn:
        #     x = self.norm(x)
        if self.do_nl:
            x = self.nl(x)
        return x


class Film(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.weight = nn.Linear(1, out_channels)
        self.bias = nn.Linear(1, out_channels)
    
    def forward(self, x, cond):
        w = torch.sigmoid(self.weight(cond)[..., None])
        b = self.bias(cond)[..., None]
        x = (x * w) + b
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_step = nn.Linear(1, 16)

        self.embed_samples = nn.Conv1d(1, 16, 7, 2, 3)  # 8192

        self.downsample = nn.Sequential(
            DownsamplingBlock(16, 32, 7, 2, 3),  # (4096)
            DownsamplingBlock(32, 64, 7, 2, 3),  # (2048)
            DownsamplingBlock(64, 128, 7, 2, 3),  # (1024)
            DownsamplingBlock(128, 256, 7, 2, 3),  # (512)
            DownsamplingBlock(256, 512, 7, 2, 3),  # (256)
            DownsamplingBlock(512, 512, 7, 2, 3),  # (128)
            DownsamplingBlock(512, 512, 7, 2, 3),  # (64)
            DownsamplingBlock(512, 512, 7, 2, 3),  # (32)
        )

        self.cond_down = nn.Sequential(
            Film(32),
            Film(64),
            Film(128),
            Film(256),
            Film(512),
            Film(512),
            Film(512),
            Film(512),
        )

        self.upsample = nn.Sequential(

            UpsamplingBlock(512, 512, 7, 1, 3, 256),  # (64)
            UpsamplingBlock(512, 512, 7, 1, 3, 256),  # (128)
            UpsamplingBlock(512, 512, 7, 1, 3, 256),  # (256)

            UpsamplingBlock(512, 256, 7, 1, 3, 256),  # (512)
            UpsamplingBlock(256, 128, 7, 1, 3, 512),  # (1024)
            UpsamplingBlock(128, 64, 7, 1, 3, 1024),  # (2048)
            UpsamplingBlock(64, 32, 7, 1, 3, 2048),  # (4096)
            UpsamplingBlock(32, 16, 7, 1, 3, 4096),  # (8192)
            UpsamplingBlock(16, 16, 7, 1, 3, 8192)  # (16384),
        )

        self.final = nn.Conv1d(16, 1, 7, 1, 3)

        self.cond_up = nn.Sequential(
            Film(512),
            Film(512),
            Film(512),
            Film(256),
            Film(128),
            Film(64),
            Film(32),
            Film(16),
            Film(16),
        )

        self.apply(init_weights)

    def forward(self, x, step):
        x = x.view(-1, 1, n_samples)
        # step = self.embed_step(step)  # (batch, 16)
        x = self.embed_samples(x)  # (batch, 16, 8192)
        # x = x + step[:, :, None]


        residual = {}

        # print('=====================')
        for film, layer in zip(self.cond_down, self.downsample):
            x = layer(x)
            x = film(x, step)
            residual[x.shape[-1]] = x
            # print(x.std().item())

        for film, layer in zip(self.cond_up, self.upsample):
            ds = residual.get(x.shape[-1])
            if ds is not None:
                x = x + ds
            x = layer(x)
            x = film(x, step)
            # print(x.std().item())

        x = self.final(x)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def add_noise(x, variance=variance):
    # mean is the current sample
    # n = torch.zeros_like(x).normal_(x, variance)
    n = torch.normal(0, variance, x.shape).to(device)
    return n, x + n


def forward_process(x, n_steps, total_steps, variance=variance):
    for i in range(n_steps):
        n, x = add_noise(x, variance=variance)
    return n, x, i / total_steps


def backward_process(step_size=1e-2):
    x = torch.zeros(1, 1, n_samples).normal_(0, 1).to(device)
    for i in torch.arange(1, -step_size, -step_size):
        step = torch.zeros(1, 1).fill_(i).to(device)
        pred = model.forward(x, step)
        x = x - pred
    return x


def train_model(batch):
    optim.zero_grad()
    # pick a number of forward_steps, emphasizing earlier
    # steps (with more signal)
    n_steps = 1 - ((np.cos(np.random.uniform(0, 1)) + 1) / 2)
    n_steps = int(total_steps * n_steps)
    n_steps = max(1, n_steps)

    noise, signal, step = forward_process(
        batch, n_steps, total_steps, variance=variance)

    steps = torch.zeros(batch.shape[0], 1).fill_(step).to(device)

    pred = model(signal, steps).view(-1, n_samples)

    loss = torch.abs(pred - noise).sum()
    loss.backward()
    optim.step()
    print('STEPS', n_steps, 'LOSS', loss.item())


@readme
class DiffusionExperiment(object):

    def __init__(self, overfit, batch_size):
        super().__init__()
        self.overfit = overfit
        self.batch_size = batch_size

        self.batch = None

    def listen(self):
        with torch.no_grad():
            # 100 steps
            x = backward_process(step_size=1e-2)
        return playable(x, samplerate)

    def spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def check_forward(self, steps):
        n, s, _ = forward_process(
            self.batch, steps, total_steps, variance=variance)
        return playable(s, samplerate)

    def run(self):

        stream = audio_stream(
            self.batch_size,
            n_samples,
            self.overfit,
            normalize=True,
            as_torch=True)

        for item in stream:
            self.batch = item
            train_model(item)
