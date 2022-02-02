from torch import nn
from modules.ddsp import DDSP
from modules.linear import LinearOutputStack

from util import make_initializer
import torch
import numpy as np
from torch.nn import functional as F


gen_init_weights = make_initializer(0.175)
disc_init_weights = make_initializer(0.1)


class BandEncoder(nn.Module):
    def __init__(
            self,
            channels,
            periodicity_feature_size,
            band_size,
            periodicity_channels=8,
            return_features=False):

        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.periodicity_feature_size = periodicity_feature_size
        self.period = LinearOutputStack(
            channels,
            layers=3,
            in_channels=periodicity_feature_size,
            out_channels=periodicity_channels)
        self.return_features = return_features

    def forward(self, x):
        batch_size = x.shape[0]
        # (batch, 64, 32, N)
        x = x.view(batch_size, 64, 32, self.periodicity_feature_size)
        x = self.period(x)
        # (batch, 64, 32, 8)
        x = x.permute(0, 3, 1, 2)
        # (batch, 8, 64, 32)
        x = x.reshape(batch_size, -1, 32)
        return x


class EncoderShell(nn.Module):
    def __init__(
            self,
            channels,
            make_band_encoder,
            make_summarizer,
            feature,
            compact=True,
            use_pos_encoding=False):

        super().__init__()
        self.channels = channels
        self.compact = compact
        self.use_pos_encoding = use_pos_encoding

        bands = {str(k): make_band_encoder(v, k)
                 for k, v in feature.kernel_sizes.items()}
        self.bands = nn.ModuleDict(bands)

        self.summarizer = make_summarizer()

        self.apply(disc_init_weights)

    def forward(self, x):
        encodings = [self.bands[str(k)](v) for k, v in x.items()]
        encodings = torch.cat(encodings, dim=-1)
        x = self.summarizer(encodings)
        return x


class ConvBandDecoder(nn.Module):
    def __init__(
            self,
            channels,
            band_size,
            activation,
            feature,
            use_filters=False,
            use_ddsp=False,
            constrain_ddsp=False):

        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.band_specific = LinearOutputStack(channels, 3)
        self.n_layers = int(np.log2(band_size) - np.log2(32))
        self.use_filters = use_filters
        self.use_ddsp = use_ddsp
        self.feature = feature

        self.upsample = nn.Sequential(*[
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels, channels, 7, 1, 3),
                activation
            )
            for _ in range(self.n_layers)])
        self.to_samples = nn.Conv1d(
            channels, 64 if (use_filters or use_ddsp) else 1, 7, 1, 3)

        if self.use_ddsp:
            self.ddsp = DDSP(channels, band_size, constrain=constrain_ddsp)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 32, self.channels)
        x = self.band_specific(x)
        x = x.permute(0, 2, 1)

        x = self.upsample(x)
        x = self.to_samples(x)


        if self.use_ddsp:
            x = self.ddsp(x)
            return x

        if not self.use_filters:
            return x

        x = F.pad(x, (0, 1))
        x = self.feature.banks[self.band_size][0].transposed_convolve(x) * 0.1
        return x


class ConvExpander(nn.Module):
    def __init__(self, channels, activation):
        super().__init__()
        self.channels = channels

        self.expand = LinearOutputStack(channels, 3, out_channels=channels * 4)

        self.n_layers = int(np.log2(32) - np.log2(4))

        self.upsample = nn.Sequential(*[
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels, channels, 7, 1, 3),
                activation
            )
            for _ in range(self.n_layers)])
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.expand(x).view(batch_size, self.channels, 4)
        x = self.upsample(x)
        return x

class DecoderShell(nn.Module):
    def __init__(
            self,
            channels,
            make_decoder,
            make_expander,
            feature):

        super().__init__()
        self.channels = channels
        self.expander = make_expander()
        bands = {str(k): make_decoder(k)
                 for k, v in zip(feature.band_sizes, feature.kernel_sizes)}

        self.bands = nn.ModuleDict(bands)
        self.apply(gen_init_weights)


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.expander(x)
        x = x.permute(0, 2, 1).reshape(batch_size, 32, self.channels)
        return {int(k): decoder(x) for k, decoder in self.bands.items()}
