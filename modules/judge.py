from torch import nn

from modules.linear import LinearOutputStack

class Judge(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = LinearOutputStack(channels, 5, out_channels=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1, self.channels)
        return self.net(x)
