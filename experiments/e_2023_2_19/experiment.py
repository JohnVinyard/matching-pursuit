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

def mu_law(x, mu=255):
    s = np.sign(x)
    x = np.abs(x)
    return s * (np.log(1 + (mu * x)) / np.log(1 + mu))


def inverse_mu_law(x, mu=255):
    s = np.sign(x)
    x = np.abs(x)
    x *= np.log(1 + mu)
    x = (np.exp(x) - 1) / mu
    return x * s

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

        self.a = nn.Parameter(torch.zeros(channels, channels * self.low_rank_channels).uniform_(-1, 1))
        self.b = nn.Parameter(torch.zeros(channels, channels * self.low_rank_channels).uniform_(-1, 1))
    
    def forward(self, x, context):
        a = context @ self.a
        b = context @ self.b
        a = a.view(self.channels, self.low_rank_channels)
        b = b.view(self.channels, self.low_rank_channels)
        w = a @ b.T
        return w


class FilmModifier(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.to_weight = nn.Linear(channels, channels)
        self.to_bias = nn.Linear(channels, channels)
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x, context):
        w = self.to_weight(context)
        b = self.to_bias(context)
        return w, b


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

class FilmLayer(nn.Module):
    def __init__(
            self,
            channels: int,
            weight: Type[FilmModifier],
            activation: Callable[[Tensor], Tensor]):
        super().__init__()
        self.base = nn.Linear(channels, channels)
        self.weights = weight(channels)
        self.activation = activation

    def forward(self, x, context):
        w, b = self.weights.forward(x, context)
        x = self.base(x)
        x = (x * w[:, None, :]) + b[:, None, :]
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

        # encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        # self.encoder = nn.TransformerEncoder(encoder, 4, norm=ExampleNorm())

        self.reduce = nn.Conv1d(33 + channels, channels, 1, 1, 0)
        self.encoder = DilatedStack(channels, [1, 3, 9, 1])

        def activation(x):
            # return torch.sin(x * 30)
            return torch.tanh(x)
            # return F.leaky_relu(x, 0.2)
        
        # self.norm_spec = NormalizedSpectrogram(
        #     512, exp.model_dim, exp.model_dim, exp.model_dim, exp.model_dim)
        
        self.embed_pos = nn.Linear(33, channels, bias=False)

        self.layers = nn.Sequential(*[
            layer(channels, weight, activation) for _ in range(n_layers)
        ])
        self.final = nn.Linear(channels, 256, bias=False)
        self.norm = ExampleNorm()
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, pos, x):
        batch = x.shape[0]

        spec = exp.pooled_filter_bank(x)
        spec_pos = pos_encoded(x.shape[0], spec.shape[-1], 16, device=x.device).permute(0, 2, 1)

        spec = torch.cat([spec_pos, spec], dim=1)
        spec = self.reduce(spec)
        ns = self.encoder.forward(spec)
        ns = self.norm(ns)

        ns = torch.mean(ns, dim=-1)
        context = ns

        x = self.embed_pos(pos)
        for layer in self.layers:
            x = layer.forward(x, context)
        x = self.final(x)
        return x.view(batch, exp.n_samples, 256)


model = Network(exp.model_dim, 6, FilmLayer, FilmModifier).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch):
    batch = torch.clamp(batch, -1, 1)

    discrete = (mu_law(batch.data.cpu().numpy()) + 1) * 0.5
    discrete = discrete * 255
    discrete = torch.from_numpy(discrete).long().to(device)

    optim.zero_grad()
    pos = pos_encoded(batch.shape[0], exp.n_samples, 16, device=device)

    recon = model.forward(pos, batch)


    # loss = F.mse_loss(recon, batch) + F.mse_loss(a, b)
    r = recon.view(-1, 256)
    loss = F.cross_entropy(r, discrete.view(-1))

    loss.backward()
    optim.step()

    indices = torch.argmax(recon, dim=-1)
    indices = indices.float().data.cpu().numpy()
    indices = (indices / 255)
    indices = (indices * 2) - 1
    return loss, indices


@readme
class NERFRedux(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None
        self.pos = pos_encoded(1, 512, 16, device=device).data.cpu().numpy()[0]
