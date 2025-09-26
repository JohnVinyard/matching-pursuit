import argparse
from typing import Generator, Tuple

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure, SupportedContentType, NumpySerializer, \
    NumpyDeserializer
from modules import stft, max_norm
from modules.infoloss import CorrelationLoss
from modules.overlap_add import overlap_add
from util import encode_audio, make_initializer, device
from torch.nn.utils.parametrizations import weight_norm

collection = LmdbCollection('funcsong')

DatasetBatch = Tuple[torch.Tensor, torch.Tensor, int]

init = make_initializer(0.05)


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transform(x: torch.Tensor) -> torch.Tensor:
    return stft(x, 2048, 256, pad=True)


def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r = transform(recon)
    t = transform(target)
    l = torch.abs(r - t).sum()
    return l


class Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        ln = weight_norm(nn.Linear(in_channels, out_channels, bias=True))
        self.mn = ln

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.mn(x)
        x = torch.selu(x)
        x = x + skip
        return x


class Network(nn.Module):
    def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            n_layers: int,
            n_samples: int,
            window_size: int = 1024):
        super().__init__()

        self.window_size = window_size
        self.step_size = window_size // 2
        self.n_coeffs = window_size // 2 + 1
        self.n_samples = n_samples

        self.input = nn.Linear(in_channels, hidden_channels, bias=True)
        self.network = nn.Sequential(*[Layer(hidden_channels, hidden_channels) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_channels, self.n_coeffs * 2, bias=False)

        self.apply(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, dim, frames = x.shape

        x = x.permute(0, 2, 1)
        x = self.input(x)
        x = self.network(x)
        x = self.output(x)
        x = x.view(batch, frames, self.n_coeffs, 2)
        x = torch.view_as_complex(x)
        x = torch.fft.irfft(x, dim=-1, norm='ortho')
        x = overlap_add(x[:, None, :, :], apply_window=True, trim=self.n_samples)
        return x


@numpy_conjure(collection)
def get_samples(path: str, samplerate: int) -> np.ndarray:
    samples, sr = librosa.load(path, sr=samplerate, mono=True)
    return samples


def pos_encoding(
        start_sample: int,
        stop_sample: int,
        total_samples: int,
        n_channels: int,
        device: torch.device) -> torch.Tensor:
    start = start_sample / total_samples
    end = stop_sample / total_samples
    n_samples = stop_sample - start_sample

    factor = np.pi * 2

    t = torch.linspace(start * factor, end * factor, n_samples, device=device)[None, :]
    freqs = torch.linspace(1, total_samples // 2, n_channels // 2, device=device)[:, None]

    s = torch.sin(t * freqs)
    c = torch.cos(t * freqs)
    encoding = torch.cat([s, c], dim=0)
    return encoding


def dataset(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        window_size: int = 1024,
        n_pos_encoding_channels: int = 64,
        batch_size: int = 8) -> Generator[DatasetBatch, None, None]:
    samples = get_samples(path, 22050)
    n_samples = len(samples)

    step_size = window_size // 2
    n_frames = n_samples // step_size
    n_segment_frames = n_segment_samples // step_size

    print(
        f'operating on {n_samples} samples {n_samples / 22050} seconds with batch size {batch_size} and {n_frames} frames')

    while True:
        batch = torch.zeros(batch_size, 1, n_segment_samples, device=device)
        pos = torch.zeros(batch_size, n_pos_encoding_channels, n_segment_frames, device=device)

        for i in range(batch_size):
            start_index = np.random.randint(n_frames - (n_segment_frames - 1))
            end_index = start_index + n_segment_frames

            chunk = torch.from_numpy(samples[start_index * step_size:end_index * step_size]).to(device)
            batch[i, 0, :] = chunk
            pos[i, :, :] = pos_encoding(start_index, end_index, n_frames, n_pos_encoding_channels, device)

        yield pos, batch, n_samples


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def train(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        window_size: int = 1024,
        n_pos_encoding_channels: int = 64,
        batch_size: int = 8,
        hidden_channels: int = 128,
        n_layers: int = 4):
    recon_audio, orig_audio = loggers(
        ['recon', 'orig'],
        'audio/wav',
        encode_audio,
        collection)

    encoding, = loggers(
        ['encoding'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    serve_conjure([
        orig_audio,
        recon_audio,
        encoding
    ], port=9999, n_workers=1)

    model = Network(
        in_channels=n_pos_encoding_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        n_samples=n_segment_samples,
        window_size=window_size).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    model_params = count_parameters(model)

    for i, pair in enumerate(dataset(
            path=path,
            device=device,
            n_segment_samples=n_segment_samples,
            window_size=window_size,
            n_pos_encoding_channels=n_pos_encoding_channels,
            batch_size=batch_size)):
        pos, samples, total_samples = pair

        encoding(pos[0])

        # log original audio
        orig_audio(max_norm(samples))

        optim.zero_grad()
        recon = model.forward(pos)

        # log recon audio
        recon_audio(max_norm(recon))

        l = loss(recon, samples)
        l.backward()
        optim.step()
        print(i, l.item(), f'Compression Ratio: {(model_params / total_samples):.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    train(
        args.path,
        torch.device('cuda'),
        n_segment_samples=2 ** 16,
        window_size=1024,
        n_pos_encoding_channels=4096,
        hidden_channels=512,
        n_layers=6,
        batch_size=32)
