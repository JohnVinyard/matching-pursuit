
import numpy as np
import torch
from modules.diffusion import DiffusionProcess
from modules.phase import AudioCodec, MelScale
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from torch import nn
from util import playable
import zounds

from util.weight_init import make_initializer

scale = MelScale()
codec = AudioCodec(scale)

diff_process = DiffusionProcess(total_steps=100, variance_per_step=1)

init_weights = make_initializer(0.1)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.nl = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.nl(x)
        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.nl = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.nl(x)
        return x


class FilmLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Linear(1, channels)
        self.bias = nn.Linear(1, channels)
    
    def forward(self, x, conditioning):
        w = self.weight(conditioning)[:, :, None, None]
        b = self.bias(conditioning)[:, :, None, None]
        return (x * w) + b

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            DownsamplingBlock(2, 16, (3, 3), (1, 2), (1, 1)),
            DownsamplingBlock(16, 32, (3, 3), (1, 2), (1, 1)),  # (batch, channels, 64, 64)
            DownsamplingBlock(32, 64, (3, 3), (2, 2), (1, 1)),  # (32, 32)
            DownsamplingBlock(64, 128, (3, 3), (2, 2), (1, 1)),  # (16, 16)
            DownsamplingBlock(128, 256, (3, 3), (2, 2), (1, 1)),  # (8, 8)
            DownsamplingBlock(256, 512, (3, 3), (2, 2), (1, 1)),  # (4, 4)
        )

        self.down_film = nn.Sequential(
            FilmLayer(16),
            FilmLayer(32),
            FilmLayer(64),
            FilmLayer(128),
            FilmLayer(256),
            FilmLayer(512),
        )

        self.up = nn.Sequential(
            UpsamplingBlock(512, 256, (4, 4), (2, 2), (1, 1)), # (8, 8)
            UpsamplingBlock(256, 128, (4, 4), (2, 2), (1, 1)), # (16, 16)
            UpsamplingBlock(128, 64, (4, 4), (2, 2), (1, 1)), # (32, 32)
            UpsamplingBlock(64, 32, (4, 4), (2, 2), (1, 1)), # (64, 64)
            UpsamplingBlock(32, 16, (3, 4), (1, 2), (1, 1)), # (64, 128)
            UpsamplingBlock(16, 2, (3, 4), (1, 2), (1, 1)), # (64, 256)
        )

        self.up_film = nn.Sequential(
            FilmLayer(256),
            FilmLayer(128),
            FilmLayer(64),
            FilmLayer(32),
            FilmLayer(16),
            FilmLayer(2),
        )

        self.apply(init_weights)

    def forward(self, x, current_step):
        
        residual = {}

        for film, layer in zip(self.down_film, self.down):
            x = layer(x)
            x = film(x, current_step)
            residual[x.shape[-2:]] = x
        
        for film, layer in zip(self.up_film, self.up):
            r = residual.get(x.shape[-2:])
            if r is not None:
                x = x + r
            x = layer(x)
            x = film(x, current_step)
        
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e4)


def train_model(batch):
    optim.zero_grad()

    batch = batch.permute(0, 3, 1, 2)
    steps = torch.rand(1)

    steps = (1 - torch.cos(steps * np.pi)) / 2

    steps = torch.zeros((batch.shape[0], 1)).fill_(float(steps))
    x, step, noise = diff_process.forward_process(batch, steps)
    noise_pred = diff_process.backward_process(x, step, model)

    loss = torch.abs(noise_pred - noise).sum()
    loss.backward()
    print(steps[0, 0].item(), loss.item(), x.std().item())
    return loss.item()


@readme
class SpectrogramDiffusion(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.spec = None
    
    def listen(self):
        with torch.no_grad():
            start = torch.normal(0, 1, (1, 2, 64, 256))
            for i in range(diff_process.total_steps):
                curr = 1 - torch.zeros((1, 1)).fill_(i / diff_process.total_steps).to(device)
                noise_pred = diff_process.backward_process(start, curr, model)
                start = start - noise_pred
            
            start = start.permute(0, 2, 3, 1)
            spec = start.data.cpu().numpy().squeeze()
            audio = codec.to_time_domain(start)
            return spec, playable(audio, zounds.SR22050())
    
    def check_forward(self):
        spec = self.spec[:1].permute(0, 3, 1, 2)
        steps = torch.ones((1, 1)).to(device)
        x, step, noise = diff_process.forward_process(spec, steps)
        return x.data.cpu().numpy().squeeze()

    def run(self):
        for item in self.stream:
            spec = codec.to_frequency_domain(item)
            self.spec = spec
            train_model(spec)
