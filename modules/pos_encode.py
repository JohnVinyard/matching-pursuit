import torch
from torch import nn
import zounds

from modules.stft import morlet_filter_bank

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


class LearnedPosEncodings(nn.Module):
    def __init__(self, n_freqs, out_channels):
        super().__init__()
        self.n_freqs = n_freqs
        self.out = nn.Linear(n_features_for_freq(n_freqs), out_channels)
    
    def forward(self, x):
        pos = pos_encoded(x.shape[0], x.shape[1], self.n_freqs, x.device)
        learned = self.out(pos)
        return x + learned


class ExpandUsingPosEncodings(nn.Module):
    def __init__(
            self,
            channels,
            time_dim,
            n_freqs,
            latent_dim,
            multiply=False,
            learnable_encodings=False,
            concat=False):

        super().__init__()
        self.learnable_encodings = learnable_encodings
        self.time_dim = time_dim
        self.n_freqs = n_freqs
        self.latent_dim = latent_dim
        self.channels = channels
        # n_freqs = n_features_for_freq(n_freqs)
        n_freqs = 256
        self.embed_pos = nn.Linear(n_freqs, channels)
        self.embed_latent = nn.Linear(latent_dim, channels)
        self.multiply = multiply
        self.pos_encodings = None
        self.concat = concat


        if self.concat:
            self.cat = nn.Linear(channels * 2, channels)
    
    def _get_pos_encodings(self, batch_size, device):
        samplerate = zounds.SR22050()
        band = zounds.FrequencyBand(0.001, samplerate.nyquist)
        scale = zounds.MelScale(band, 128)
        bank = torch.from_numpy(morlet_filter_bank(samplerate, 2**15, scale, 0.25, normalize=False).real)\
            .float().view(1, 128, 2**15).permute(0, 2, 1).repeat(batch_size, 1, 1)
        
        bank = bank * (torch.linspace(1, 0, 128)[None, None, :]  ** 2)
        bank = torch.cat([bank, -bank], dim=-1)
        return bank
        
        if not self.learnable_encodings:
            self.pos_encodings = pos_encoded(batch_size, self.time_dim, self.n_freqs, device)
            return self.pos_encodings
        
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
        elif self.concat:
            x = torch.cat([x, pos], dim=-1)
            x = self.cat(x)
        else:
            x = x + pos

        return x
