import torch
from torch import nn

from modules.linear import LinearOutputStack

class ModulatedLayer(nn.Module):
    def __init__(self, channels, forward_layers, conditioning_layers, activation):
        super().__init__()
        self.f = LinearOutputStack(channels, forward_layers, activation=activation)
        self.weight = LinearOutputStack(channels, conditioning_layers, activation=activation)
        self.bias = LinearOutputStack(channels, conditioning_layers, activation=activation)

    def forward(self, x, conditioning):
        x = self.f(x)
        w = self.weight(conditioning)
        b = self.bias(conditioning)

        return (x * torch.sigmoid(w)) + b


class ModulatedStack(nn.Module):
    def __init__(self, channels, layers, freq_bins, pos_encoder, activation):
        super().__init__()

        self.pos_encoder = pos_encoder
        self.initial = LinearOutputStack(channels, 1, in_channels=33, activation=activation)
        self.net = nn.Sequential(
            *[ModulatedLayer(channels, 2, 2) for _ in range(layers)])
        self.mag = LinearOutputStack(channels, 3, out_channels=freq_bins, activation=activation)
        self.phase = LinearOutputStack(channels, 3, out_channels=freq_bins, activation=activation)

    def forward(self, latent):
        pos = self.pos_encoder(latent.shape[0], 128, 16, latent.device)
        x = self.initial(pos)

        for layer in self.net:
            x = layer(x, latent[:, None, :])

        mag = self.mag(x)
        phase = self.phase(x)
        x = torch.cat([mag[..., None], phase[..., None]], dim=-1)
        return x
