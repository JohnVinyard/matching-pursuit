from torch import nn
import torch
import numpy as np
from modules import ResidualStack


class Attention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)

    def forward(self, x):
        x = x.view(-1, self.channels)
        l = x.shape[0]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn = torch.matmul(q, k.T)
        attn = attn / np.sqrt(attn.numel())
        attn = torch.softmax(attn.view(-1), dim=0).view(l, l)
        x = torch.matmul(attn, v)
        return x


class LinearOutputStack(nn.Module):
    def __init__(self, channels, layers, out_channels=None):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.out_channels = out_channels or channels
        self.net = nn.Sequential(
            ResidualStack(channels, layers),
            nn.Linear(channels, self.out_channels)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.seq = nn.Sequential(
            LinearOutputStack(channels, 3),
            Attention(channels),
            LinearOutputStack(channels, 3)
        )
        self.length = nn.Sequential(
            nn.Linear(1, channels),
            LinearOutputStack(channels, 2)
        )
        self.judge = LinearOutputStack(channels * 2, 1, 1)

    def forward(self, x, l):
        n = x.shape[0]
        l = self.length(l.view(1, 1)).repeat(n, 1)
        x = x.view(-1, self.channels)
        x = self.seq(x)
        x = torch.cat([l, x], dim=1)
        x = self.judge(x)
        return x


if __name__ == '__main__':
    network = Discriminator(128)
    embeddings = torch.FloatTensor(700, 128).normal_(0, 1)
    x = network(embeddings, torch.FloatTensor([0.2]))
    print(x.shape)
