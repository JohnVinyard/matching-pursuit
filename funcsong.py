import argparse
from typing import Generator, Tuple, Union

import librosa
import numpy as np
import torch
from torch import nn
from torch.nn.utils import weight_norm
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure
from modules import stft, max_norm, fft_frequency_recompose, flattened_multiband_spectrogram
from util import encode_audio, make_initializer, device
from torch.nn import functional as F

collection = LmdbCollection('funcsong')

DatasetBatch = Tuple[torch.Tensor, torch.Tensor, int]

init = make_initializer(0.02)


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def transform(x: torch.Tensor) -> torch.Tensor:
    # return flattened_multiband_spectrogram(x, { 'xs': (64, 16)})
    return stft(x, 2048, 256, pad=True)


def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r = transform(recon)
    t = transform(target)
    l = torch.abs(r - t).sum()
    return l


class DampedHarmonicOscillatorBlock(nn.Module):
    def __init__(
            self,
            n_samples: int,
            n_oscillators: int,
            n_resonances: int,
            expressivity: int):
        super().__init__()
        self.n_samples = n_samples
        self.n_oscillators = n_oscillators
        self.n_resonances = n_resonances
        self.expressivity = expressivity

        self.damping = nn.Parameter(
            torch.zeros(1, n_oscillators, n_resonances, expressivity) \
                .uniform_(0.5, 1.5))

        self.mass = nn.Parameter(
            torch.zeros(1, n_oscillators, n_resonances, expressivity) \
                .uniform_(-2, 2))

        self.tension = nn.Parameter(
            torch.zeros(1, n_oscillators, n_resonances, expressivity) \
                .uniform_(4, 9))

        self.initial_displacement = nn.Parameter(
            torch.zeros(1, n_oscillators, n_resonances, expressivity) \
                .uniform_(-1, 2))

        self.amplitudes = nn.Parameter(
            torch.zeros(1, n_oscillators, n_resonances, expressivity, 1) \
                .uniform_(-1, 1))


    def _materialize_resonances(self, energy: torch.Tensor, device: torch.device, tension_modifier: torch.Tensor = None, scaling: torch.Tensor = None):
        time = torch.linspace(0, 10, self.n_samples, device=device) \
            .view(1, 1, 1, self.n_samples)

        t = self.tension[..., None]

        if tension_modifier is not None:
            t = t + (tension_modifier[0] * scaling)

        x = damped_harmonic_oscillator(
            energy=energy,
            time=time,
            mass=torch.sigmoid(self.mass[..., None]) * 2,
            damping=torch.sigmoid(self.damping[..., None]) * 30,
            tension=10 ** t,
            initial_displacement=self.initial_displacement[..., None],
        )

        x = x.view(-1, self.n_oscillators, self.n_resonances, self.expressivity, self.n_samples)
        x = x * self.amplitudes


        x = torch.sum(x, dim=1)
        x = x.view(-1, 1, self.n_resonances, self.expressivity, self.n_samples)

        return x

class DampedHarmonicOscillatorStack(nn.Module):
    def __init__(
            self,
            n_samples: int,
            n_oscillators: int,
            n_resonances: int,
            expressivity: int):

        super().__init__()
        self.dho1 = DampedHarmonicOscillatorBlock(n_samples, n_oscillators, n_resonances, expressivity)
        self.dho2 = DampedHarmonicOscillatorBlock(n_samples, n_oscillators, n_resonances, expressivity)
        self.dho3 = DampedHarmonicOscillatorBlock(n_samples, n_oscillators, n_resonances, expressivity)
        self.influence = nn.Parameter(torch.zeros(n_oscillators, n_resonances, expressivity, 1).uniform_(-0.01, 0.01))
        self.influence2 = nn.Parameter(torch.zeros(n_oscillators, n_resonances, expressivity, 1).uniform_(-0.01, 0.01))


        self.mix = nn.Parameter(torch.zeros(1, 1, n_resonances, expressivity, 1, 3).uniform_(-1, 1))

    def forward(self, energy: torch.Tensor):

        outputs = []

        x = self.dho1._materialize_resonances(energy, self.influence.device)
        outputs.append(x)
        x = self.dho2._materialize_resonances(energy, self.influence.device, x, self.influence)
        outputs.append(x)
        x = self.dho3._materialize_resonances(energy, self.influence.device, x, self.influence2)
        outputs.append(x)

        outputs = torch.stack(outputs, dim=-1)

        x = (outputs * torch.softmax(self.mix, dim=-1)).sum(dim=-1)
        # x = outputs @ torch.softmax(self.mix, dim=-1)

        return x

class Layer(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        ln = weight_norm(nn.Linear(in_channels, out_channels, bias=True))
        self.mn = ln



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        x = self.mn(x)
        x = torch.selu(x)
        # x = F.leaky_relu(x, 0.2)
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
            n_resonances: int = 64):

        super().__init__()

        self.segment_size = segment_size
        self.n_samples = n_samples
        self.n_layers = n_layers
        self.n_resonances = n_resonances

        self.input = nn.Linear(in_channels, hidden_channels, bias=True)
        self.network = nn.ModuleList([Layer(hidden_channels, hidden_channels) for _ in range(n_layers)])
        self.to_energy = nn.Linear(hidden_channels, n_resonances)
        self.dho = DampedHarmonicOscillatorStack(segment_size, n_oscillators=2, n_resonances=n_resonances, expressivity=1)
        self.apply(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]

        x = x.permute(0, 2, 1)
        x = self.input(x)

        for i, layer in enumerate(self.network):
            x = layer(x)

        e = torch.abs(self.to_energy(x)).permute(0, 2, 1).view(batch, 1, self.n_resonances, 1, self.segment_size)
        # e = e.view(batch, -1, self.segment_size)
        # e = F.avg_pool1d(e, kernel_size=128, stride=1, padding=64)[..., :self.segment_size]
        # e = e.view(batch, 1, self.n_resonances, 1, self.segment_size)

        d = self.dho(e)

        d = d.view(batch, self.n_resonances, self.segment_size)
        d = torch.sum(d, dim=1, keepdim=True)

        return d




@torch.jit.script
def damped_harmonic_oscillator(
        energy: torch.Tensor,
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor
) -> torch.Tensor:
    x = (damping / (2 * mass))

    omega = torch.sqrt(torch.abs(tension - (x ** 2)))

    phi = torch.atan2(
        (x * initial_displacement),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    z = a * energy * torch.cos(omega * time - phi)
    return z

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
        hidden_channels=256,
        n_layers=4,
        batch_size=4)
