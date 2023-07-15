
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.linear import LinearOutputStack
from modules.sparse import sparsify
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

d_size = 64
kernel_size = exp.n_samples

band = zounds.FrequencyBand(1, exp.samplerate.nyquist)
scale = zounds.LinearScale(band, d_size)

d = morlet_filter_bank(
    exp.samplerate, 
    kernel_size, 
    scale, 
    0.01,
    normalize=True).real.astype(np.float32)

d = torch.from_numpy(d).to(device).T


class Modulator(nn.Module):
    def __init__(self, channels, layers=3, activation=lambda x: x):
        super().__init__()
        self.channels = channels
        self.net = LinearOutputStack(
            channels, 
            layers=layers, 
            out_channels=channels, 
            activation=activation,
            norm=nn.LayerNorm((exp.n_samples, channels))
        )
        self.w = nn.Linear(channels, channels)
        self.b = nn.Linear(channels, channels)
    
    def forward(self, x):
        x = self.net(x)
        w = self.w(x)
        b = self.b(x)
        return w, b


class Stack(nn.Module):
    def __init__(self, channels, layers, sub_layers, activation):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.sub_layers = sub_layers
        self.activation = activation

        self.latent = nn.Parameter(torch.zeros(1, self.channels).uniform_(-1, 1))

        self.main = nn.Sequential(*[
            LinearOutputStack(
                channels, 
                sub_layers, 
                out_channels=channels, 
                activation=activation, 
                norm=nn.LayerNorm((exp.n_samples, channels))
            ) for _ in range(layers)
        ])

        self.mods = nn.Sequential(*[
            Modulator(channels, layers=sub_layers, activation=activation)
            for _ in range(layers)
        ])

        self.final = nn.Linear(channels, 1)

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self):
        x = self.latent
        x = x.view(-1, 1, self.channels)
        pos = d[None, ...]
        x = x + pos

        for layer, mod in zip(self.main, self.mods):
            l = layer(x)
            w, b = mod(x)
            x = (l * w) + b
        
        x = self.final(x)
        x = x.view(-1, 1, exp.n_samples)
        return x

model = Stack(d_size, layers=4, sub_layers=1, activation=lambda x: F.leaky_relu(x, 0.2)).to(device)
optim = optimizer(model)



def train(batch, i):
    optim.zero_grad()
    batch = batch.view(-1, 1, exp.n_samples)
    recon = model.forward()
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class ComplexMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    