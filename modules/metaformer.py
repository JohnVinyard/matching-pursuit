from torch import nn
from torch.nn import functional as F

from modules.atoms import unit_norm


class PoolMixer(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.avg_pool1d(x, self.window_size, 1, padding=self.window_size // 2)
        x = x.permute(0, 2, 1)
        return x


class MetaFormerBlock(nn.Module):
    def __init__(self, channels, make_mixer, make_norm):
        super().__init__()
        self.norm1 = make_norm(channels)
        self.mixer = make_mixer(channels)
        self.norm2 = make_norm(channels)
        self.ln = nn.Linear(channels, channels)

    def forward(self, x):
        orig = x
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.mixer(x)
        x = orig + x
        if self.norm2 is not None:
            x = self.norm2(x)
        x = self.ln(x)
        return x


class MetaFormer(nn.Module):
    def __init__(self, channels, layers, make_mixer, make_norm, return_features=False):
        super().__init__()
        self.net = nn.Sequential(
            *[MetaFormerBlock(channels, make_mixer, make_norm) for _ in range(layers)])
        self.return_features = return_features

    def forward(self, x):
        if not self.return_features:
            x = self.net(x)
            return x
        
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        
        return x, features
