from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class ResidualBlock(nn.Module):
    def __init__(
            self,
            channels,
            bias=True,
            activation=None,
            shortcut=True,
            do_weight_norm: bool = False,
            norm=lambda channels: None):

        super().__init__()
        self.channels = channels
        self.do_weight_norm = do_weight_norm

        if self.do_weight_norm:
            l1 = weight_norm(nn.Linear(channels, channels, bias))
            l2 = weight_norm(nn.Linear(channels, channels, bias))
            self.l1 = l1
            self.l2 = l2
        else:
            self.l1 = nn.Linear(channels, channels, bias)
            self.l2 = nn.Linear(channels, channels, bias)

        self.activation = activation or nn.LeakyReLU(0.2)
        self.shortcut = shortcut
        self.norm = norm(channels)

    def forward(self, x):
        shortcut = x
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        if self.shortcut:
            x = self.activation(shortcut + x)
        else:
            x = self.activation(x)
        
        if self.norm:
            x = self.norm(x)
        return x




class ResidualStack(nn.Module):
    def __init__(
            self,
            channels, 
            layers, 
            bias=True, 
            activation=lambda x: F.leaky_relu(x, 0.2), 
            shortcut=True,
            do_weight_norm: bool = False,
            norm=lambda channels: None):

        super().__init__()
        self.activation = activation
        self.channels = channels
        self.layers = layers
        self.net = nn.Sequential(
            *[
                ResidualBlock(channels, bias, self.activation, shortcut, norm=norm, do_weight_norm=do_weight_norm)
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        return self.net(x)



class LinearOutputStack(nn.Module):
    def __init__(
            self, channels,
            layers,
            out_channels=None,
            in_channels=None,
            activation=lambda x: F.leaky_relu(x, 0.2),
            bias=True,
            shortcut=True,
            do_weight_norm: bool = False,
            norm=lambda channels: None):

        super().__init__()
        self.channels = channels
        self.layers = layers
        self.out_channels = out_channels or channels

        core = [
            ResidualStack(
                channels, layers, activation=activation, bias=bias, shortcut=shortcut, norm=norm, do_weight_norm=do_weight_norm),
            nn.Linear(channels, self.out_channels, bias=self.out_channels > 1)
        ]

        inp = [] if in_channels is None else [
            nn.Linear(in_channels, channels, bias=bias)]

        self.net = nn.Sequential(*[
            *inp,
            *core
        ])

    def forward(self, x):
        x = self.net(x)
        return x
