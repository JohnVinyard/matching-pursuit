import math
import zounds
from typing import Sequence
import torch
from torch import nn
from modules import OscillatorBank, NoiseModel
import numpy as np
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.phase import AudioCodec, MelScale
from train.optim import optimizer
from upsample import FFTUpsampleBlock
from util import device, playable
from train import gan_cycle, get_latent
from loss import least_squares_disc_loss, least_squares_generator_loss
from util.readmedocs import readme
from util.weight_init import make_initializer

init_weights = make_initializer(0.1)

scale = MelScale()
codec = AudioCodec(scale)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, in_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.in_size = in_size

        self.up = FFTUpsampleBlock(in_channels, in_channels, in_size)
        # self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)

        # self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv1d(in_channels, out_channels, 7, 1, 3)
        self.norm = nn.BatchNorm1d(out_channels)
        self.nl = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        # x = self.norm(x)
        x = self.nl(x)
        return x


class BandGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        

        self.up = nn.Sequential(
            UpsamplingBlock(512, 256, 4, 2, 1, 4),
            UpsamplingBlock(256, 128, 4, 2, 1, 8),
            UpsamplingBlock(128, 64, 4, 2, 1, 16),
            nn.Conv1d(64, 64, 3, 1, 1)
        )

        self.to_harm = nn.Conv1d(64, 64, 1, 1, 0)
        self.to_noise = nn.Conv1d(64, 64, 1, 1, 0)

        self.osc = OscillatorBank(
            64, 
            64, 
            size, 
            constrain=True, 
            log_frequency=False, 
            amp_activation=torch.relu)
        
        self.noise = NoiseModel(
            64, 32, 64, size, 32, activation=torch.relu)


    
    def forward(self, x):
        
        x = self.up(x)

        h = self.to_harm(x)
        n = self.to_noise(x)

        h = self.osc(h)
        n = self.noise(n)

        return h + n


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        self.bands = nn.ModuleDict({str(bs): BandGenerator(bs) for bs in self.band_sizes})
        self.initial = nn.Linear(64, 512 * 4)
        self.apply(init_weights)
    
    def forward(self, x):
        x = x.view(-1, 64)
        x = self.initial(x).reshape(-1, 512, 4)
        

        output = {k: model(x) for k, model in self.bands.items()}
        return output


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm = norm

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.nl = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.nl(x)
        return x



class BandDisc(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        layers = int(math.log2(size) - math.log2(32))
        channels = [2**i for i in range(layers)] + [32]

        self.net = nn.Sequential(
            *[DownsamplingBlock(channels[i], channels[i + 1], 7, 2, 3) for i in range(layers)])
        self.judge = nn.Conv1d(32, 1, 3, 1, 1)
    
    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x.view(x.shape[0], -1))
        
        j = self.judge(x)
        
        features = torch.cat(features, dim=1)
        return x, features, j


class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.band_sizes = [512, 1024, 2048, 4096, 8192, 16384]
        self.bands = nn.ModuleDict({str(bs): BandDisc(bs) for bs in self.band_sizes})
        self.apply(init_weights)

        self.final = nn.Sequential(
            DownsamplingBlock(32 * len(self.band_sizes), 128, 7, 2, 3),
            DownsamplingBlock(128, 256, 7, 2, 3),
            DownsamplingBlock(256, 512, 7, 2, 3),
            DownsamplingBlock(512, 512, 2, 2, 0, norm=False),
            nn.Conv1d(512, 1, 1, 1, 0)
        )
    
    def forward(self, x):
        features = []
        bands = []
        judgements = []

        x = {str(k): v for k, v in x.items()}


        for k, v in self.bands.items():
            z, f, j = v(x[k])
            bands.append(z)
            features.append(f)
            judgements.append(j.view(f.shape[0], -1))
        
        x = torch.cat(bands, dim=1)
        for layer in self.final:
            x = layer(x)
            features.append(x.view(x.shape[0], -1))
        
        judgements = torch.cat(judgements, dim=1)
        x = torch.cat([judgements.view(-1), x.view(-1)])
        
        features = torch.cat(features, dim=1)
        return x, features

gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-4)

disc = Disc().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(batch):
    gen_optim.zero_grad()
    x = get_latent(list(batch.values())[0].shape[0], 64)
    fake = gen(x)
    j, _ = disc(fake)
    loss = least_squares_generator_loss(j)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake

def train_disc(batch):
    disc_optim.zero_grad()
    x = get_latent(list(batch.values())[0].shape[0], 64)
    fake = gen(x)
    fj, _ = disc(fake)
    rj, _ = disc(batch)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())

@readme
class MultiBandGan(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
    
    def listen(self):
        results = fft_frequency_recompose({int(k):v for k, v in self.fake.items()}, 16384)
        return playable(results, zounds.SR22050())
    
    def orig(self):
        results = fft_frequency_recompose({int(k):v for k, v in self.real.items()}, 16384)
        return playable(results, zounds.SR22050())
    

    def fake_spec(self):
        results = fft_frequency_recompose({int(k):v for k, v in self.fake.items()}, 16384)
        results = results.view(-1, 16384)
        spec = codec.to_frequency_domain(results)
        return spec[0, ..., 0].data.cpu().numpy().squeeze()
    
    def real_spec(self):
        results = fft_frequency_recompose({int(k):v for k, v in self.real.items()}, 16384)
        results = results.view(-1, 16384)
        spec = codec.to_frequency_domain(results)
        return spec[0, ..., 0].data.cpu().numpy().squeeze()
    
    
    def run(self):
        for item in self.stream:
            bands = fft_frequency_decompose(item.view(-1, 1, 16384), min_size=512)
            self.real = bands
            step = next(gan_cycle)
            if step == 'gen':
                self.fake = train_gen(bands)
            else:
                train_disc(bands)

