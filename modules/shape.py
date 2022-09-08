from torch import nn


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape
    
    def forward(self, x):
        x = x.reshape(x.shape[0], *self.new_shape)
        return x