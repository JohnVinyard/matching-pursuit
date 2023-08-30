import torch
from torch import nn


class HyperNetworkLayer(nn.Module):
    """
    Hyper-layer that produces layer weights via a factorized matrix
    """

    def __init__(self, latent_channels, layer_latent, layer_in_channels, layer_out_channels):
        super().__init__()
        self.latent_channels = latent_channels
        self.layer_latent = layer_latent
        self.layer_in_channels = layer_in_channels
        self.layer_out_channels = layer_out_channels

        self.a = nn.Linear(latent_channels, layer_latent * layer_in_channels)
        self.b = nn.Linear(latent_channels, layer_latent * layer_out_channels)

    def forward(self, x):
        a = self.a(x).view(-1, self.layer_in_channels, self.layer_latent)
        b = self.b(x).view(-1, self.layer_latent, self.layer_out_channels)
        weights = a @ b

        def forward(z):
            if len(z.shape) != 3:
                z = z[:, None, :]
            z = torch.bmm(z, weights)
            return z

        return weights, forward
    
    


if __name__ == '__main__':

    latent = torch.zeros(7, 13).normal_(0, 1)

    model = HyperNetworkLayer(13, 5, 128, 128)

    weights, forward = model.forward(latent)
    print(weights.shape)

    inp = torch.zeros(7, 128)
    result = forward(inp)
    print(result.shape)

    # expected_shape = (7, 128, 128)
    # assert weights.shape == expected_shape
