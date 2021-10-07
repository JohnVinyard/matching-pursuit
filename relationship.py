import torch
from torch import nn
import numpy as np

def unit_norm(x):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
    else:
        n = torch.norm(x, dim=-1, keepdim=True)
    return x / (n + 1e-12)

class Relationship(nn.Module):
    def __init__(self, in_channels, channels):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels

        self.transform = nn.Linear(in_channels, channels)
        self.attend = nn.Linear(in_channels, 1)
        

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, -1, self.in_channels)

        rel = x[:, None, :, :] - x[:, :, None, :]
        # (batch, time, time, channels)
        attn = torch.sigmoid(self.attend(rel))

        x = self.transform(rel)

        x = attn * x
        # x = x.mean(dim=1)
        x, _ = x.max(dim=1)
        # x = unit_norm(x) * 3.2
        return x


if __name__ == '__main__':
    t = torch.FloatTensor(16, 128, 18).normal_(0, 1)
    r = Relationship(18, 128)
    x = r.forward(t)
    print(x.shape)
