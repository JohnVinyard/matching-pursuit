from torch import nn
from torch.nn import functional as F

class MixerBlock(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size

        self.l1 = nn.Linear(channels, size)
        self.l1_gate = nn.Linear(channels, size)


        self.l2 = nn.Linear(size, channels)
        self.l2_gate = nn.Linear(channels, size)
    
    def forward(self, x):
        residual = x
        x = x.view(-1, self.size, self.channels)
        x = self.l1(x) # (-1, size, size)
        x = x.permute(0, 2, 1)
        x = self.l2(x) # (-1, size, channels)
        x = x + residual
        return x


class Mixer(nn.Module):

    def __init__(self, channels, size, layers, return_features=False):
        super().__init__()
        self.channels = channels
        self.size = size
        self.layers = layers
        self.return_features = return_features

        self.net = nn.Sequential(*[
            MixerBlock(channels, size)
            for _ in range(layers)
        ])
    
    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
            x = F.leaky_relu(x, 0.2)
        
        if self.return_features:
            return x, features
        else:
            return x
        
        
            