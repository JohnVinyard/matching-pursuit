from typing import Callable, Type
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.normalization import ExampleNorm
from modules.pos_encode import pos_encoded
from modules.stft import stft
from perceptual.feature import NormalizedSpectrogram
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import Tensor, nn
from util import device, playable
from torch.nn import functional as F
import torch
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class LayerWeights(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.weights = nn.Parameter(
            torch.zeros(channels, channels).uniform_(-0.1, 0.1))

    def forward(self, x, context):
        return self.weights


class LowRankLayerWeights(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.low_rank_channels = 8

        self.a = nn.Parameter(torch.zeros(channels, channels * self.low_rank_channels).uniform_(-0.1, 0.1))
        self.b = nn.Parameter(torch.zeros(channels, channels * self.low_rank_channels).uniform_(-0.1, 0.1))
    
    def forward(self, x, context):
        a = context @ self.a
        b = context @ self.b
        a = a.view(self.channels, self.low_rank_channels)
        b = b.view(self.channels, self.low_rank_channels)
        w = a @ b.T
        return w


class Layer(nn.Module):
    def __init__(
            self,
            channels: int,
            weight: Type[LayerWeights],
            activation: Callable[[Tensor], Tensor]):
        super().__init__()
        self.weights = weight(channels)
        self.activation = activation

    def forward(self, x, context):
        w = self.weights.forward(x, context)
        x = x @ w.T
        x = self.activation(x)
        return x


class Network(nn.Module):
    def __init__(
            self, 
            channels: 
            int, n_layers: 
            int, layer: Type[Layer], 
            weight: Type[LayerWeights]):

        super().__init__()
        self.channels = channels

        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 4, norm=ExampleNorm())

        def activation(x):
            # return torch.sin(x * 30)
            return torch.tanh(x)
        
        self.norm_spec = NormalizedSpectrogram(
            512, exp.model_dim, exp.model_dim, exp.model_dim, exp.model_dim)
        
        self.embed_pos = nn.Linear(33, channels, bias=False)

        self.layers = nn.Sequential(*[
            layer(channels, weight, activation) for _ in range(n_layers)
        ])
        self.final = nn.Linear(channels, 1, bias=False)
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, pos, x):
        spec = exp.fb.forward(x, normalize=False)
        ns = self.norm_spec.forward(spec).permute(0, 2, 1)
        ns = self.encoder.forward(ns)
        ns = torch.mean(ns, dim=1)
        context = ns
        x = self.embed_pos(pos)
        for layer in self.layers:
            x = layer.forward(x, context)
        x = self.final(x)
        return x.view(-1, 1, exp.n_samples)


model = Network(exp.model_dim, 6, Layer, LowRankLayerWeights).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch):
    optim.zero_grad()
    pos = pos_encoded(batch.shape[0], exp.n_samples, 16, device=device)

    recon = model.forward(pos, batch)

    a = stft(recon, 512, 256, pad=True)
    b = stft(batch, 512, 256, pad=True)

    loss = F.mse_loss(recon, batch) + F.mse_loss(a, b)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class NERFRedux(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None
        self.pos = pos_encoded(1, 512, 16, device=device).data.cpu().numpy()[0]
