import argparse
from typing import Generator, Tuple

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure
from modules import stft, max_norm, flattened_multiband_spectrogram
from spiking import SpikingModel
from util import encode_audio, make_initializer
from torch.nn import functional as F

collection = LmdbCollection('funcsong')

DatasetBatch = Tuple[torch.Tensor, torch.Tensor]

init = make_initializer(0.02)


loss_model = SpikingModel(64, 64, 64, 64, 64).to(torch.device('cuda'))


# def transform(x: torch.Tensor) -> torch.Tensor:
#     # x = loss_model.forward(x)
#     # return x
#     return flattened_multiband_spectrogram(x, {'small': (64, 16)})
#     # return stft(x, 2048, 256, pad=True)


def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # r = transform(recon)
    # t = transform(target)
    # l = torch.abs(r - t).sum()
    l = loss_model.compute_multiband_loss(target, recon)
    return l


class Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mn = nn.Linear(in_channels, out_channels, bias=True)
        # self.factor = nn.Parameter(torch.zeros(1).fill_(30))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.mn(x)
        x = torch.relu(x)
        x = x + skip
        return x


class Network(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, n_layers: int):
        super().__init__()
        self.input = nn.Linear(in_channels, hidden_channels, bias=True)
        self.network = nn.Sequential(*[Layer(hidden_channels, hidden_channels) for _ in range(n_layers)])
        self.output = nn.Linear(hidden_channels, out_channels, bias=True)

        # self.apply(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.input(x)
        x = self.network(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)
        return x


@numpy_conjure(collection)
def get_samples(path: str, samplerate: int) -> np.ndarray:
    samples, sr = librosa.load(path, sr=samplerate, mono=True)
    return samples
    # return samples[2 ** 18:2 ** 18 + 2 ** 14]


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
        n_pos_encoding_channels: int = 64,
        batch_size: int = 8) -> Generator[DatasetBatch, None, None]:
    samples = get_samples(path, 22050)
    n_samples = len(samples)
    print(f'operating on {n_samples} samples {n_samples / 22050} seconds')

    while True:
        batch = torch.zeros(batch_size, 1, n_segment_samples, device=device)
        pos = torch.zeros(batch_size, n_pos_encoding_channels, n_segment_samples, device=device)

        for i in range(batch_size):
            start_index = np.random.randint(n_samples - (n_segment_samples - 1))
            end_index = start_index + n_segment_samples
            chunk = torch.from_numpy(samples[start_index:end_index]).to(device)
            batch[i, 0, :] = chunk
            pos[i, :, :] = pos_encoding(start_index, end_index, n_samples, n_pos_encoding_channels, device)

        yield pos, batch


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def train(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        n_pos_encoding_channels: int = 64,
        batch_size: int = 8,
        hidden_channels: int = 128,
        n_layers: int = 4):
    recon_audio, orig_audio = loggers(
        ['recon', 'orig'],
        'audio/wav',
        encode_audio,
        collection)

    # encoding, = loggers(
    #     ['encoding'],
    #     SupportedContentType.Spectrogram.value,
    #     to_numpy,
    #     collection,
    #     serializer=NumpySerializer(),
    #     deserializer=NumpyDeserializer())

    serve_conjure([
        orig_audio,
        recon_audio,
        # encoding
    ], port=9999, n_workers=1)

    model = Network(
        n_pos_encoding_channels,
        hidden_channels=hidden_channels,
        out_channels=1,
        n_layers=n_layers).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    for i, pair in enumerate(dataset(path, device, n_segment_samples, n_pos_encoding_channels, batch_size)):
        pos, samples = pair

        # encoding(pos[0])

        # log original audio
        orig_audio(max_norm(samples))

        optim.zero_grad()
        recon = model.forward(pos)

        # log recon audio
        recon_audio(max_norm(recon))

        l = loss(recon, samples)
        l.backward()
        optim.step()
        print(i, l.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    train(
        args.path,
        torch.device('cuda'),
        n_segment_samples=2 ** 14,
        n_pos_encoding_channels=4096,
        hidden_channels=256,
        n_layers=6,
        batch_size=4)
