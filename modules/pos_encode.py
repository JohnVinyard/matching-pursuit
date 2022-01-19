import torch
from torch import nn


def pos_encode_feature(x, domain, n_samples, n_freqs):
    # batch, time, _ = x.shape
    x = torch.clamp(x, -domain, domain)
    output = [x]
    for i in range(n_freqs):
        output.extend([
            torch.sin((2 ** i) * x),
            torch.cos((2 ** i) * x)
        ])

    x = torch.cat(output, dim=-1)
    return x


def n_features_for_freq(n_freqs):
    return n_freqs * 2 + 1

def pos_encoded(batch_size, time_dim, n_freqs, device=None):
    """
    Return positional encodings with shape (batch_size, time_dim, n_features)
    """
    n_features = n_features_for_freq(n_freqs)
    pos = pos_encode_feature(torch.linspace(-1, 1, time_dim).view(-1, 1), 1, time_dim, n_freqs)\
        .view(1, time_dim, n_features)\
        .repeat(batch_size, 1, 1)\
        .view(batch_size, time_dim, n_features)\

    if device:
        pos = pos.to(device)
    return pos



class ExpandUsingPosEncodings(nn.Module):
    def __init__(self, channels, time_dim, n_freqs, latent_dim, multiply=False):
        super().__init__()
        self.time_dim = time_dim
        self.n_freqs = n_freqs
        self.latent_dim = latent_dim
        self.channels = channels
        n_freqs = n_features_for_freq(n_freqs)
        self.embed_pos = nn.Linear(n_freqs, channels)
        self.embed_latent = nn.Linear(latent_dim, channels)
        self.multiply = multiply
    
    def forward(self, x):
        batch_size = x.shape[0]

        pos = pos_encoded(batch_size, self.time_dim, self.n_freqs, x.device)
        pos = self.embed_pos(pos)

        x = self.embed_latent(x).view(batch_size, 1, self.channels).repeat(1, self.time_dim, 1)

        if self.multiply:
            x = x * pos
        else:
            x = x + pos
        
        return x

