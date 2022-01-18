from torch import nn

from weight_init import make_initializer
import torch
from torch.nn import functional as F

init_weights = make_initializer(0.1)


class ForwardBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.ln = nn.Linear(self.n_channels, self.n_channels)
        self.apply(init_weights)

    def forward(self, x):
        shortcut = x
        x = self.ln(x)
        x = F.leaky_relu(x + shortcut, 0.2)
        return x


class FourierMixer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = torch.fft.fft(x, dim=-1, norm='ortho')
        x = torch.fft.fft(x, dim=-2, norm='ortho')
        x = x.real
        return x


class Transformer(nn.Module):
    def __init__(self, n_channels, n_layers):
        super().__init__()
        self.n_channels = n_channels
        self.n_layers = n_layers
        self.net = nn.Sequential(*[
            nn.Sequential(
                ForwardBlock(self.n_channels),
                FourierMixer()
            )
            for _ in range(self.n_layers)])

    def forward(self, x):
        x = self.net(x)
        return x
