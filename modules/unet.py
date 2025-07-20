from torch import nn
import torch
import numpy as np
from torch.nn import Identity

from modules.stft import stft
from torch.nn.utils import weight_norm

class DownsamplingBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.1),
            weight_norm(nn.Conv1d(channels, channels, 3, 2, 1)),
            nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownsamplingDiscriminator(nn.Module):

    def __init__(
            self,
            window_size: int,
            step_size: int,
            n_samples: int,
            channels: int,
            complex_valued: bool = False):

        super().__init__()
        self.window_size = window_size
        self.step_size = step_size
        self.n_samples = n_samples
        self.n_frames = n_samples // step_size
        self.channels = channels
        self.n_coeffs = self.window_size // 2 + 1
        self.complex_valued = complex_valued

        self.input_channels = self.n_coeffs * 2 if self.complex_valued else self.n_coeffs

        self.proj = nn.Conv1d(self.input_channels, channels, 1, 1, 0)
        self.n_layers = int(np.log2(self.n_frames)) - 2
        self.downsample = nn.Sequential(*[DownsamplingBlock(channels) for i in range(self.n_layers)])
        self.judge = nn.Conv1d(channels, 1, 4, 4, 0)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time = x.shape
        assert time == self.n_samples

        x = stft(
            x,
            ws=self.window_size,
            step=self.step_size,
            pad=True,
            return_complex=self.complex_valued).view(batch, -1, self.input_channels).permute(0, 2, 1)

        print('DISC', x.shape)
        x = self.proj(x)
        x = self.downsample(x)
        x = self.judge(x)
        return x



class UNet(nn.Module):
    def __init__(self, channels, is_disc: bool = False, norm: bool = True):
        super().__init__()
        self.channels = channels
        self.is_disc = is_disc
        self.norm = norm

        def norm_layer():
            if norm:
                return nn.BatchNorm1d(self.channels)
            else:
                return Identity()

        if self.is_disc:
            self.disc = nn.Conv1d(channels, 1, kernel_size=4, stride=4, padding=0)

        self.down = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),
            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),

            # 128
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                norm_layer()
            ),
        )

        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)

    def encode(self, x):
        # Input will be (batch, 1024, 128)
        context = {}

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x

        return x

    
    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x

        if self.is_disc:
            x = self.disc(x)
            return x
        
        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]
            
        x = self.proj(x)
        return x
