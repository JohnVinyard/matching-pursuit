
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

diff_process = DiffusionProcess(total_steps=100, variance_per_step=0.1)

init_weights = make_initializer(0.02)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
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

        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
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
        w = self.weight(conditioning)[:, :, None]
        b = self.bias(conditioning)[:, :, None]
        return (x * w) + b

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            DownsamplingBlock(512, 128, 3, 2, 1), # 32
            DownsamplingBlock(128, 128, 3, 2, 1), # 16
            DownsamplingBlock(128, 128, 3, 2, 1), # 8
            DownsamplingBlock(128, 128, 3, 2, 1), # 4
            DownsamplingBlock(128, 128, 3, 2, 1), # 2
            DownsamplingBlock(128, 128, 3, 1, 1), # 2
        )

        self.down_film = nn.Sequential(
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(128),
        )

        self.up = nn.Sequential(
            UpsamplingBlock(128, 128, 4, 2, 1), # 4
            UpsamplingBlock(128, 128, 4, 2, 1), # 8
            UpsamplingBlock(128, 128, 4, 2, 1), # 16
            UpsamplingBlock(128, 128, 4, 2, 1), # 32
            UpsamplingBlock(128, 512, 4, 2, 1), # 64
        )

        self.up_film = nn.Sequential(
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(128),
            FilmLayer(512),
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
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()

    batch = batch.reshape(-1, 64, 512).permute(0, 2, 1)

    steps = torch.rand(1)
    steps = torch.zeros((batch.shape[0], 1)).fill_(float(steps)).to(device)
    
    x, step, noise = diff_process.forward_process(batch, steps)
    noise_pred = diff_process.backward_process(x, step, model)
    loss = torch.abs(noise_pred - noise).sum()
    loss.backward()
    optim.step()

    print(steps[0, 0].item(), loss.item(), x.std().item())
    return loss.item()


@readme
class SpectrogramDiffusion(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.spec = None
    
    def gen(self):
        with torch.no_grad():   
            start = torch.normal(0, 1.8, (1, 512, 64)).to(device)

            for i in range(diff_process.total_steps):
                curr = 1 - torch.zeros((1, 1)).fill_(i / diff_process.total_steps).to(device)
                noise_pred = diff_process.backward_process(start, curr, model)
                start = start - noise_pred
            
            start = start.reshape(-1, 2, 256, 64).permute(0, 3, 2, 1)
            return start
    
    def listen(self):
        start = self.gen()
        audio = codec.to_time_domain(start)
        return playable(audio, zounds.SR22050())
    
    def look(self, dim=0):
        start = self.gen()
        spec = start.data.cpu().numpy().squeeze()
        return spec[..., dim]

    
    def check_forward(self, steps=1):
        spec = self.spec[:1]
        print(spec.shape)
        spec = spec.reshape(-1, 64, 512).permute(0, 2, 1)

        steps = torch.zeros((1, 1)).fill_(steps).to(device)
        x, step, noise = diff_process.forward_process(spec, steps)
        return x.data.cpu().numpy().squeeze().T.reshape((-1, 64, 256, 2)).squeeze()

    def run(self):
        for item in self.stream:
            spec = codec.to_frequency_domain(item)
            self.spec = spec
            train_model(spec)

            self.check_forward()