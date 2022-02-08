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
    def __init__(
            self,
            channels,
            time_dim,
            n_freqs,
            latent_dim,
            multiply=False,
            learnable_encodings=False):

        super().__init__()
        self.learnable_encodings = learnable_encodings
        self.time_dim = time_dim
        self.n_freqs = n_freqs
        self.latent_dim = latent_dim
        self.channels = channels
        n_freqs = n_features_for_freq(n_freqs)
        self.embed_pos = nn.Linear(n_freqs, channels)
        self.embed_latent = nn.Linear(latent_dim, channels)
        self.multiply = multiply
        self.pos_encodings = None
    
    def _get_pos_encodings(self, batch_size, device):
        if not self.learnable_encodings:
            return pos_encoded(batch_size, self.time_dim, self.n_freqs, device)
        
        if self.pos_encodings is None:
            self.pos_encodings = nn.Parameter(
                pos_encoded(1, self.time_dim, self.n_freqs, device))
        
        return self.pos_encodings.repeat(batch_size, 1, 1)

    def forward(self, x):
        batch_size = x.shape[0]

        pos = self._get_pos_encodings(batch_size, x.device)
        pos = self.embed_pos(pos)

        factor = self.time_dim // x.shape[1]
        x = self.embed_latent(x).view(
            batch_size, -1, self.channels).repeat(1, factor, 1)
        

        if self.multiply:
            x = x * pos
        else:
            x = x + pos

        return x
