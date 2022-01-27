import zounds
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from .decompose import fft_frequency_decompose

samplerate = zounds.SR22050()
n_samples = 16384

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PsychoacousticFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.banks = self._build_filter_banks()
        self.weights = {
            512: 1,
            1024: 1,
            2048: 1,
            4096: 1,
            8192: 1,
            16384: 1
        }

    def _build_filter_banks(self):
        spans = [
            (20, 344),
            (344, 689),
            (689, 1378),
            (1378, 2756),
            (2756, 5512),
            (5512, 11025)
        ]
        kernel_sizes = [
            32,
            64,
            128,
            256,
            512,
            1024
        ]
        self.kernel_sizes = kernel_sizes
        keys = [
            512,
            1024,
            2048,
            4096,
            8192,
            16384
        ]
        bank_dict = {}
        self.kernel_sizes = {}
        for span, size, key in zip(spans, kernel_sizes, keys):
            self.kernel_sizes[key] = size // 2 + 1

            band = zounds.FrequencyBand(*span)
            scale = zounds.MelScale(band, 64)
            samples_per_second = span[1] * 2
            freq = zounds.Picoseconds(1e12) / samples_per_second
            sr = zounds.SampleRate(freq, freq)
            fb = zounds.learn.FilterBank(
                sr,
                size,
                scale,
                np.geomspace(0.25, 0.9, num=64),
                normalize_filters=True,
                a_weighting=False).to(device)
            bank_dict[key] = (fb, span, size)
        return bank_dict

    def chroma_basis(self, size):
        fb, span, size = self.banks[size]
        band = zounds.FrequencyBand(*span)
        chr = zounds.ChromaScale(band)
        basis = chr._basis(fb.scale, zounds.OggVorbisWindowingFunc())
        x = np.array(basis)
        return x
    
    @property
    def n_bands(self):
        return len(self.band_sizes)

    @property
    def band_sizes(self):
        return list(self.banks.keys())

    def decompose(self, x):
        return fft_frequency_decompose(x, 512)

    def compute_feature_dict(self, x):

        # TODO: Don't perform the decomposition if x is already
        # a dictionary
        if not isinstance(x, dict):
            batch_size = x.shape[0]

            x = x.view(batch_size, 1, -1)

            x = self.decompose(x)

        bands = dict()

        for size, data in self.banks.items():
            fb, span, kernel_size = data
            band = x[size]

            spec = fb.forward(band, normalize=False)

            spec = F.pad(spec, (kernel_size // 4, kernel_size // 4))
            spec = spec.unfold(-1, kernel_size, kernel_size // 2)

            import warnings
            warnings.warn('Remember that youre doing windowing here!')
            spec = spec * torch.hamming_window(spec.shape[-1])[None, None, None, :]
            
            spec = torch.abs(torch.fft.rfft(spec, dim=-1))
            bands[size] = spec

        return bands

    def forward(self, x):

        if isinstance(x, dict):
            batch_size = list(x.values())[0].shape[0]
        else:
            batch_size = x.shape[0]

        # Don't perform the decomposition if x is already
        # a dictionary
        if not isinstance(x, dict):

            x = x.view(batch_size, 1, -1)
            x = self.decompose(x)

        results = []
        specs = []

        for size, data in self.banks.items():
            fb, span, kernel_size = data
            band = x[size]

            spec = fb.forward(band, normalize=False)

            spec = F.pad(spec, (kernel_size // 4, kernel_size // 4))
            spec = spec.unfold(-1, kernel_size, kernel_size // 2)

            import warnings
            warnings.warn('Remember that youre doing windowing here!')
            spec = spec * torch.hamming_window(spec.shape[-1])[None, None, None, :]

            if span[0] > 99999:
                spec = torch.norm(spec, dim=-1)
            else:
                freq = torch.fft.rfft(spec, dim=-1)
                spec = torch.abs(freq)
                specs.append(spec.data.cpu().numpy().squeeze())

            results.append(spec.reshape(batch_size, -1))

        x = torch.cat(results, dim=1)
        # print('RATIO', x.shape, x.shape[0] / n_samples)
        return x, specs
