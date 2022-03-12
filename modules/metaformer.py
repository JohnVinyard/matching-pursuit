from torch import nn
from torch.nn import functional as F


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
    def __init__(self, channels, make_mixer):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.mixer = make_mixer(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ln = nn.Linear(channels, channels)

    def forward(self, x):
        orig = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = orig + x
        x = self.norm2(x)
        x = self.ln(x)
        return x


class MetaFormer(nn.Module):
    def __init__(self, channels, layers, make_mixer):
        super().__init__()
        self.net = nn.Sequential(
            *[MetaFormerBlock(channels, make_mixer) for _ in range(layers)])

    def forward(self, x):
        x = self.net(x)
        return x
