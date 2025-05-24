from typing import Callable
import torch
from conjure import LmdbCollection, loggers, serve_conjure
from torch import nn
from torch.optim import Adam
import numpy as np
from itertools import count
from torch.nn.utils.weight_norm import weight_norm

from modules import max_norm
from util import encode_audio, make_initializer

initializer = make_initializer(0.1)


def to_blocks(x: torch.Tensor, block_size: int) -> torch.Tensor:
    x = x.unfold(-1, block_size, block_size)
    return x


def to_samples(x: torch.Tensor, block_size: int) -> torch.Tensor:
    time, block_size = x.shape[-2:]
    return x.view(*x.shape[:-2], time * block_size)


def total_energy(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x).sum()


def cumulative_energy(x: torch.Tensor) -> torch.Tensor:
    x = x.view(x.shape[0], -1, x.shape[-1])
    x = torch.abs(x)
    x = torch.sum(x, dim=1, keepdim=True)
    x = torch.cumsum(x, dim=-1)

    divisor = torch.cumsum(torch.linspace(1, x.shape[-1], x.shape[-1], device=x.device), dim=-1)
    x = x / divisor
    return x


def compute_discontinuity(x: torch.Tensor) -> torch.Tensor:
    last_samples = x[..., 0:-1:, -1]
    first_samples = x[..., 1::, 0]
    return torch.abs(last_samples - first_samples)


class Block(nn.Module):
    def __init__(
            self,
            channels: int,
            non_linearity: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.channels = channels
        self.non_linearity = non_linearity
        self.proj = nn.Linear(channels, channels, bias=False)
        self.keys = nn.Linear(channels, channels, bias=False)
        self.values = nn.Linear(channels, channels, bias=False)
        self.queries = nn.Linear(channels, channels, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, _ = x.shape

        x = self.proj(x)
        orig = x

        k = self.keys(x)
        q = self.queries(x)

        v = self.values(x)

        m = k @ q.permute(0, 2, 1)
        m = torch.tril(m)

        x = v.permute(0, 2, 1) @ m
        x = self.non_linearity(x)
        x = x.permute(0, 2, 1)

        return x


class Interface(nn.Module):
    def __init__(self, input_channels: int, model_channels: int, block_size: int):
        super().__init__()
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.block_size = block_size

        self.to_model_dim = nn.Linear(block_size * input_channels, model_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, time = x.shape
        blocked = to_blocks(x, self.block_size)
        n_blocks = blocked.shape[-2]
        blocked = blocked.permute(0, 2, 1, 3).reshape(batch, n_blocks, -1)
        x = self.to_model_dim(blocked)
        return x


class EnergyInstrumentModel(nn.Module):
    def __init__(
            self,
            input_channels: int,
            model_channels: int,
            block_size: int,
            n_layers: int):
        super().__init__()
        self.input_channels = input_channels
        self.model_channels = model_channels
        self.block_size = block_size
        self.n_layers = n_layers

        self.interface = Interface(input_channels, model_channels, block_size)
        self.net = nn.Sequential(*[
            Block(model_channels, lambda x: torch.tanh(x))
            for _ in range(n_layers)
        ])
        self.to_samples = nn.Linear(model_channels, block_size, bias=False)

        self.apply(initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.interface(x)
        for layer in self.net:
            x = layer(x)
        x = self.to_samples(x)
        return x


def train_model(device: torch.device = torch.device('cpu'), ):
    batch_size = 8
    input_channels = 64
    samples = 2 ** 15
    block_size = 128
    model_channels = 512

    model = EnergyInstrumentModel(
        input_channels,
        model_channels,
        block_size,
        n_layers=3).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    collection = LmdbCollection('energy')

    audio, = loggers(['audio'], 'audio/wav', encode_audio, collection)

    serve_conjure([audio], port=9999, n_workers=1)

    for i in count():
        optim.zero_grad()
        input_signal = torch.zeros(batch_size, input_channels, samples, device=device).bernoulli_(p=0.0001)
        input_signal *= torch.zeros_like(input_signal).uniform_(0, 10)

        blockwise = x = model(input_signal)
        x = to_samples(x, block_size)

        audio(max_norm(x)[0, ...].view(-1))

        # total energy should be the same
        total_energy_loss = torch.abs(total_energy(x) - total_energy(input_signal)).sum()

        # cumulative energy should always be *less* at each time step in the output
        resonance = 0.01
        input_cumulative_energy = cumulative_energy(input_signal)
        output_cumulative_energy = cumulative_energy(x)

        # the constant 0.1 defines resonance and should be another input to the network
        cumulative_energy_loss = torch.abs(output_cumulative_energy - (input_cumulative_energy * resonance)).sum()

        # cumulative_energy_loss = torch.clamp(output_cumulative_energy - input_cumulative_energy, 0, np.inf).sum()


        block_loss = compute_discontinuity(blockwise)
        block_loss = torch.abs(block_loss).sum()

        loss = cumulative_energy_loss + block_loss
        loss.backward()
        optim.step()
        print(i, loss.item(), total_energy_loss.item(), cumulative_energy_loss.item())


if __name__ == '__main__':
    train_model(torch.device('cuda'))
