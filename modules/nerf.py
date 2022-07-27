from torch import nn
import torch
from torch.nn import functional as F
from modules.pos_encode import pos_encoded


class NerfLayer(nn.Module):
    def __init__(self, channels, factor=30):
        super().__init__()
        self.channels = channels
        self.factor = factor

        self.ln = nn.Linear(channels, channels)
        self.w = nn.Linear(channels, channels)
        self.b = nn.Linear(channels, channels)
    
    def forward(self, x):
        x = self.ln(x)
        # w = self.w(z)
        # b = self.b(z)
        # x = (x * w) + b
        return torch.sin(x * self.factor)


class NerfStack(nn.Module):
    def __init__(self, n_samples, channels, layers, factor):
        super().__init__()

        self.to_channels = nn.Linear(33, channels)
        self.factor = factor
        self.n_samples = n_samples
        self.channels = channels
        self.net = nn.Sequential(*[
            NerfLayer(channels, factor) for _ in range(layers)
        ])

        self.to_samples = nn.Linear(channels, 1)

    def forward(self, z):
        z = z.view(-1, self.channels)
        p = pos_encoded(z.shape[0], self.n_samples, 16)

        
        x = self.to_channels(p) + z[:, None, :]

        # for layer in self.net:
        #     x = layer(x, z[:, None, :])
        x = self.net(x)
        
        x = self.to_samples(x)
        x = torch.sin(x * self.factor)
        return x