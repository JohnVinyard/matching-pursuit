from torch import nn
import torch
from torch.nn import functional as F
import numpy as np
from torch.nn.utils.weight_norm import weight_norm as wnorm

from util.weight_init import make_initializer


init_weights = make_initializer(0.1)


def upsample_with_holes(low_sr: torch.Tensor, desired_size: int) -> torch.Tensor:
    """Upsample signal by simply placing existing samples at fixed
    intervals, with zeros in between
    """
    factor = desired_size // low_sr.shape[-1]
    upsampled = torch.zeros(*low_sr.shape[:-1], desired_size)
    upsampled[..., ::factor] = low_sr
    return upsampled


def interpolate_last_axis(low_sr: torch.Tensor, desired_size) -> torch.Tensor:
    """A convenience wrapper around `torch.nn.functional.interpolate` to allow
    for an arbitrary number of leading dimensions
    """
    orig_shape = low_sr.shape
    new_shape = orig_shape[:-1] + (desired_size,)
    last_dim = low_sr.shape[-1]
    
    reshaped = low_sr.reshape(-1, 1, last_dim)
    upsampled = F.interpolate(reshaped, mode='linear', size=desired_size)
    upsampled = upsampled.reshape(*new_shape)
    return upsampled

    

class UpsampleBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            mode='nearest',
            weight_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if weight_norm:
            self.conv = wnorm(nn.Conv1d(in_channels, out_channels, 3, 1, 1))
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, 3, 1, 1)
        
        self.mode = mode

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode=self.mode)
        x = self.conv(x)
        return x


class Nearest(UpsampleBlock):
    def __init__(self, in_channels, out_channels, weight_norm=False):
        super().__init__(in_channels, out_channels, 'nearest', weight_norm=weight_norm)


class Linear(UpsampleBlock):
    def __init__(self, in_channels, out_channels, weight_norm=False):
        super().__init__(in_channels, out_channels, 'linear', weight_norm=weight_norm)


class LearnedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, step_size=2, padding=1, weight_norm=False):
        super().__init__()
        if weight_norm:
            self.conv = wnorm(nn.ConvTranspose1d(in_channels, out_channels, kernel_size, step_size, padding))
        else:
            self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, step_size, padding)

    def forward(self, x):
        x = self.conv(x)
        return x


class FFTUpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, factor=2, infer=False, weight_norm=False):
        super().__init__()
        self.channels = in_channels
        self.size = size
        self.factor = factor
        self.new_time = int(self.size * self.factor)
        self.orig_coeffs = self.size // 2 + 1
        self.n_coeffs = self.new_time // 2 + 1

        inferred = self.n_coeffs - self.orig_coeffs

        self.infer = infer
        r = torch.zeros(self.orig_coeffs, inferred).uniform_(-0.01, 0.01)
        i = torch.zeros(self.orig_coeffs, inferred).uniform_(-0.01, 0.01)
        c = torch.complex(r, i)

        self.inferred = nn.Parameter(c)
        self.final = nn.Conv1d(in_channels, out_channels, 3, 1, 1)

    def upsample(self, x):
        batch = x.shape[0]

        x = x.reshape(-1, self.channels, self.size)

        coeffs = torch.fft.rfft(x, axis=-1, norm='ortho')

        r = torch.zeros(batch, self.channels, self.n_coeffs).to(x.device)
        i = torch.zeros(batch, self.channels, self.n_coeffs).to(x.device)

        new_coeffs = torch.complex(r, i)

        new_coeffs[:, :, :self.orig_coeffs] = coeffs

        if self.infer:
            inferred = coeffs @ self.inferred
            new_coeffs[:, :, self.orig_coeffs:] = inferred

        x = torch.fft.irfft(new_coeffs, n=self.new_time, norm='ortho')
        # x = self.final(x)
        return x

    def forward(self, x):
        x = self.upsample(x)
        x = self.final(x)
        return x


def iter_layers(start_size, end_size):
    for i in range(int(np.log2(start_size)), int(np.log2(end_size))):
        yield i, 2**i



class ConvUpsample(nn.Module):
    def __init__(
            self,
            latent_dim,
            channels,
            start_size,
            end_size,
            mode,
            out_channels,
            from_latent=True,
            batch_norm=False,
            layer_norm=False,
            weight_norm=False,
            norm=nn.Identity(),
            activation_factory=lambda: nn.LeakyReLU(0.2)):

        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.mode = mode
        self.start_size = start_size
        self.end_size = end_size
        self.n_layers = int(np.log2(end_size) - np.log2(start_size))
        self.from_latent = from_latent
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm

        self.begin = nn.Linear(
            self.latent_dim, self.channels * self.start_size)
        self.out_channels = out_channels

        if mode == 'learned':
            def layer(channels, size, weight_norm): return LearnedUpsampleBlock(
                channels, channels, weight_norm=weight_norm)
        elif mode == 'nearest':
            def layer(channels, size, weight_norm): return Nearest(channels, channels, weight_norm=weight_norm)
        elif mode == 'linear':
            def layer(channels, size, weight_norm): return Linear(channels, channels, weight_norm=weight_norm)
        elif mode == 'fft':
            def layer(channels, size, weight_norm): return FFTUpsampleBlock(
                channels, channels, size, weight_norm=weight_norm)
        elif mode == 'fft_learned':
            def layer(channels, size, weight_norm): return FFTUpsampleBlock(
                channels, channels, size, infer=True, weight_norm=weight_norm)

        self.net = nn.Sequential(*[nn.Sequential(
            layer(channels, size, weight_norm),
            # nn.BatchNorm1d(channels) if batch_norm else norm,
            self._build_norm_layer(channels, size),
            activation_factory(),
        ) for _, size in iter_layers(start_size, end_size)])

        self.final = nn.Conv1d(channels, self.out_channels, 3, 1, 1)

        self.apply(init_weights)
    
    def _build_norm_layer(self, channels, size):
        if self.batch_norm:
            return nn.BatchNorm1d(channels)
        elif self.layer_norm:
            return nn.LayerNorm((channels, size * 2), elementwise_affine=False)
        else:
            return nn.Identity()

    def __iter__(self):
        return iter(self.net)

    def forward(self, x):
        if self.from_latent:
            x = x.reshape(-1, self.latent_dim)
            x = self.begin(x)
            x = x.view(-1, self.channels, self.start_size)

        x = self.net(x)
        x = self.final(x)
        return x


class SimpleEncoder(nn.Module):
    def __init__(self, in_channels, channels, input_size, latent_dim):
        super().__init__()
        self.channels = channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        layers = int(np.log2(input_size))

        self.start = nn.Conv1d(in_channels, self.channels, 3, 1, 1)

        self.net = nn.Sequential(*[nn.Sequential(
            nn.Conv1d(channels, channels, 3, 2, 1),
            nn.LeakyReLU(0.2)
        ) for _ in range(layers)])
        self.final = nn.Conv1d(channels, latent_dim, 1, 1, 0)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, self.in_channels, self.input_size)
        x = self.start(x)
        x = self.net(x)
        x = self.final(x)
        return x.view(-1, self.latent_dim)
