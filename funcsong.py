import argparse
from typing import Generator, Tuple, Union

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure
from modules import stft, max_norm, fft_frequency_recompose, flattened_multiband_spectrogram
from util import encode_audio, make_initializer, device
from torch.nn import functional as F

collection = LmdbCollection('funcsong')

DatasetBatch = Tuple[torch.Tensor, torch.Tensor, int]

init = make_initializer(0.01)


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(x, { 'xs': (64, 16)})
    # return stft(x, 2048, 256, pad=True)


def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r = transform(recon)
    t = transform(target)
    l = torch.abs(r - t).sum()
    return l


class Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # ln = weight_norm(nn.Linear(in_channels, out_channels, bias=True))
        # self.mn = ln
        self.mn = nn.Linear(in_channels, out_channels)
        # self.mn = nn.Parameter(torch.zeros(in_channels, out_channels))

    # def perturbed_weights(self) -> torch.Tensor:
    #     return torch.zeros_like(self.mn).normal_(self.mn.mean().item(), self.mn.std().item())

    def forward(self, x: torch.Tensor, perturb_weights: bool = False) -> torch.Tensor:
        skip = x
        x = self.mn(x)

        # if perturb_weights:
        #     pw = self.perturbed_weights()
        #     x = pw @ x
        # else:
        #     x = self.mn @ x

        # if  is None:
        #     x = self.mn @ x
        # else:
        #     x = alternate_weights @ x
        # x = torch.selu(x)
        x = F.leaky_relu(x, 0.2)
        # x = torch.sin(x * 30)
        # x = torch.tanh(x)
        x = x + skip
        return x


class Network(nn.Module):
    def __init__(
            self,
            segment_size: int,
            in_channels: int,
            hidden_channels: int,
            n_layers: int,
            n_samples: int,
            smallest_band: int = 512):
        super().__init__()

        self.segment_size = segment_size
        self.n_samples = n_samples
        self.n_layers = n_layers

        self.sm = int(np.log2(smallest_band))
        self.lg = int(np.log2(segment_size))

        print(segment_size, n_samples, )

        self.transforms = (nn.ModuleDict
                           ({str(2 ** size): nn.Linear(hidden_channels, hidden_channels) for size in range(self.sm, self.lg + 1)}))
        self.amp_transforms = (nn.ModuleDict
                           ({str(2 ** size): nn.Linear(hidden_channels, hidden_channels) for size in range(self.sm, self.lg + 1)}))


        self.input = nn.Linear(in_channels, hidden_channels, bias=True)
        self.network = nn.ModuleList([Layer(hidden_channels, hidden_channels) for _ in range(n_layers)])
        self.apply(init)

    def forward(self, x: torch.Tensor, perturbed_layer: int = None) -> torch.Tensor:

        x = x.permute(0, 2, 1)
        x = self.input(x)

        for i, layer in enumerate(self.network):
            x = layer(x, i == perturbed_layer)


        total_size = x.shape[1]
        bands = {}

        pool_size = 1

        for k, layer in self.transforms.items():
            size = int(k)
            step = total_size // size
            z = x[:, ::step, :]
            a = self.amp_transforms[k].forward(z).permute(0, 2, 1) ** 2
            a = F.avg_pool1d(a, pool_size, stride=1, padding=pool_size // 2)[..., :a.shape[-1]]
            z = layer(z).permute(0, 2, 1)
            bands[k] = torch.sum(z * a, dim=1, keepdim=True)
            pool_size *= 2

        x = fft_frequency_recompose(bands, desired_size=self.segment_size)


        return x


@numpy_conjure(collection)
def get_samples(path: str, samplerate: int) -> np.ndarray:
    samples, sr = librosa.load(path, sr=samplerate, mono=True)
    print('SAMPLES', samples.shape)
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
        n_pos_encoding_channels: int = 64,
        batch_size: int = 8) -> Generator[DatasetBatch, None, None]:
    samples = get_samples(path, 22050)
    n_samples = len(samples)


    print(
        f'operating on {n_samples} samples {n_samples / 22050} seconds with batch size {batch_size} and {n_segment_samples} frames')

    while True:
        batch = torch.zeros(batch_size, 1, n_segment_samples, device=device)
        pos = torch.zeros(batch_size, n_pos_encoding_channels, n_segment_samples, device=device)

        for i in range(batch_size):
            start_index = np.random.randint(0, n_samples - n_segment_samples)
            end_index = start_index + n_segment_samples

            chunk = torch.from_numpy(samples[start_index : end_index]).to(device)
            batch[i, 0, :] = chunk
            pos[i, :, :] = pos_encoding(
                start_sample=start_index,
                stop_sample=end_index,
                total_samples=n_samples,
                n_channels=n_pos_encoding_channels,
                device=device)

        yield pos, batch, n_samples


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
    #     NumpySerializer(),
    #     NumpyDeserializer())

    serve_conjure([
        orig_audio,
        recon_audio,
        # encoding
    ], port=9999, n_workers=1, web_components_version='0.0.101')

    model = Network(
        segment_size=n_segment_samples,
        n_samples=n_segment_samples,
        in_channels=n_pos_encoding_channels,
        hidden_channels=hidden_channels,
        n_layers=n_layers).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    model_params = count_parameters(model)

    # loss_model = CorrelationLoss().to(device)

    for i, pair in enumerate(dataset(
            path=path,
            device=device,
            n_segment_samples=n_segment_samples,
            n_pos_encoding_channels=n_pos_encoding_channels,
            batch_size=batch_size)):
        pos, samples, total_samples = pair

        # encoding(pos[0])

        # log original audio
        orig_audio(max_norm(samples))

        optim.zero_grad()
        recon = model.forward(pos)

        # with torch.no_grad():
        #     pert = model.forward(pos, perturbed_layer=np.random.randint(0, model.n_layers))
        #     perturbed(max_norm(pert))

        # log recon audio
        recon_audio(max_norm(recon))

        l = loss(recon, samples)
        # l = loss_model.multiband_noise_loss(samples, recon, 64, 16)
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
        n_segment_samples=2 ** 15,
        n_pos_encoding_channels=4096,
        hidden_channels=128,
        n_layers=4,
        batch_size=8)
