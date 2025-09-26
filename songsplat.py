import argparse
from typing import Generator, Tuple, Union

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure, SupportedContentType, NumpySerializer, \
    NumpyDeserializer
from modules import stft, max_norm, sparse_softmax
from modules.overlap_add import overlap_add
from modules.transfer import fft_convolve
from modules.upsample import upsample_with_holes
from util import encode_audio, make_initializer, device
from torch.nn.utils.parametrizations import weight_norm
from torch.nn import functional as F

collection = LmdbCollection('songsplat')

DatasetBatch = Tuple[torch.Tensor, torch.Tensor, int, int, int]

init = make_initializer(0.05)


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_vector(t: torch.Tensor, total_samples: int, n_channels: int, device: torch.device) -> torch.Tensor:
    freqs = torch.linspace(1, total_samples // 2, n_channels // 2, device=device)[:, None]
    s = torch.sin(t * freqs)
    c = torch.cos(t * freqs)
    encoding = torch.cat([t.view(1, -1), s, c], dim=0)
    return encoding


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

    return time_vector(t, total_samples, n_channels, device)


def transform(x: torch.Tensor) -> torch.Tensor:
    return stft(x, 2048, 256, pad=True)

def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r = transform(recon)
    t = transform(target)
    l = torch.abs(r - t).sum()
    return l


class HalfLappedWindowParams:

    def __init__(self, window_size: int):
        super().__init__()
        self.window_size = window_size

    @property
    def step_size(self):
        return self.window_size // 2

    @property
    def n_coeffs(self):
        return self.window_size // 2 + 1

    def total_samples(self, n_frames: int) -> int:
        return n_frames * self.step_size

    def group_delay(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(0, np.pi, self.n_coeffs, device=device)

    def materialize_phase(
            self, n_frames: int,
            dither: Union[torch.Tensor, None],
            device: torch.device) -> torch.Tensor:

        gd = self.group_delay(device)
        gd = gd.view(1, 1, self.n_coeffs).repeat(1, n_frames, 1)

        if dither is not None:
            noise = torch.zeros_like(gd).uniform_(-1, 1)
            gd = gd + (dither * self.group_delay(device)[None, None, :] * noise)

        gd = torch.cumsum(gd, dim=1)
        return gd

class EventGenerator(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            n_frames: int,
            window_size: int,
            hidden_channels: int,
            n_layers: int,
            n_pos_encoding_channels: int):

        super().__init__()
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.window_size = window_size
        self.n_pos_encoding_channels = n_pos_encoding_channels
        self.window = HalfLappedWindowParams(window_size)
        self.hidden_channels = hidden_channels

        # learnable event pos encodings
        pos_encoding = torch.zeros(1, n_frames, n_pos_encoding_channels).uniform_(-1, 1)
        decay = torch.linspace(1, 0, n_frames)
        p = torch.zeros(n_pos_encoding_channels).uniform_(10, 100)
        self.register_buffer('pos_encoding', pos_encoding * (decay[None, :] ** p[:, None]))

        ep = nn.Linear(n_pos_encoding_channels, hidden_channels)
        self.embed_pos = weight_norm(ep)
        ee = nn.Linear(latent_dim, hidden_channels)
        self.embed_event = weight_norm(ee)

        tm = nn.Linear(hidden_channels, self.window.n_coeffs)
        self.to_magnitude = weight_norm(tm)
        td = nn.Linear(hidden_channels, self.window.n_coeffs)
        self.to_dither = weight_norm(td)

        layers = []
        for _ in range(n_layers):
            layer = nn.Linear(hidden_channels, hidden_channels)
            layer = weight_norm(layer)
            layers.append(layer)

        self.network = nn.ParameterList(layers)

    @property
    def total_samples(self):
        return self.window.total_samples(self.n_frames)

    def forward(self, events: torch.Tensor) -> torch.Tensor:

        batch, n_events, dim = events.shape

        e = self.embed_event(events).view(batch, n_events, 1, self.hidden_channels)
        p = self.embed_pos(self.pos_encoding).view(1, 1, self.n_frames, self.hidden_channels)

        x = e + p
        x = x.view(batch, n_events, self.n_frames, self.hidden_channels)
        for layer in self.network:
            skip = x
            x = layer(x)
            x = torch.selu(x)
            x = x + skip

        m = torch.abs(self.to_magnitude(x))
        p = self.to_dither(x)
        p = self.window.materialize_phase(self.n_frames, p, device=x.device)

        spec = m * torch.exp(1j * p)
        windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
        samples = overlap_add(windowed, apply_window=True, trim=self.total_samples)
        samples = samples.view(batch, n_events, 1, -1)
        return samples


def schedule_events(
        events: torch.Tensor,
        times: torch.Tensor,
        pos_encodings: torch.Tensor,
        temperature: float) -> torch.Tensor:


    batch, n_events, n_samples = events.shape
    n_events, pos_encoding_dim = times.shape
    _, n_frames, pos_encoding_dim = pos_encodings.shape

    pos_encodings = pos_encodings.view(pos_encoding_dim, -1)

    sim = times @ pos_encodings.T
    # sched = F.gumbel_softmax(sim, tau=temperature, hard=True, dim=-1)
    sched = sparse_softmax(sim, normalize=True)
    sched = upsample_with_holes(sched, desired_size=n_samples)
    scheduled = fft_convolve(events, sched)
    return scheduled

class Model(nn.Module):

    def __init__(
            self,
            total_samples: int,
            n_frames: int,
            samplerate: int,
            event_latent_dim: int,
            window_size: int,
            hidden_channels: int,
            n_layers: int,
            pos_encoding_channels: int,
            n_segment_samples: int,
            events_per_second: float):

        super().__init__()

        self.n_segment_samples = n_segment_samples
        self.n_frames = n_frames
        self.window_size = window_size
        self.window_params = HalfLappedWindowParams(window_size)
        self.pos_encoding_channels = pos_encoding_channels
        self.event_latent_dim = event_latent_dim
        self.total_samples = total_samples
        self.samplerate = samplerate
        self.total_seconds = total_samples / samplerate
        self.events_per_second = events_per_second
        self.total_events = int(self.total_seconds * self.events_per_second)
        print('TOTAL EVENTS', self.total_events)

        self.events = nn.Parameter(torch.zeros(self.total_events, self.event_latent_dim).uniform_(-1, 1))
        times = torch.zeros(self.total_events, device=device).uniform_(-1, 1)
        self.times = nn.Parameter(time_vector(times, self.total_samples, pos_encoding_channels, device))


        n_event_frames = n_segment_samples // self.window_params.step_size

        self.generator = EventGenerator(
            event_latent_dim, n_event_frames, window_size, hidden_channels, n_layers, 128)

        self.apply(init)

    def forward(
            self,
            pos_encoding: torch.Tensor,
            start_frame: int,
            end_frame: int,
            temperature: float) -> torch.Tensor:

        n_frames = end_frame - start_frame
        n_samples = self.window_params.total_samples(n_frames)
        early_frame = start_frame - n_frames

        start_rel = early_frame / self.n_frames
        stop_rel = end_frame / self.n_frames

        # use the linear [0-1] channel to locate events that will affect this interval
        t = self.times[0, :]

        mask = (t > start_rel) & (t < stop_rel)

        events = self.events[mask]
        times = self.times.T[mask]

        if events.shape[0] == 0:
            raise ValueError('no events')

        samples = self.generator.forward(events[None, ...])
        samples = samples.view(1, -1, n_samples)
        padding = torch.zeros_like(samples)
        samples = torch.cat([samples, padding], dim=-1)

        scheduled = schedule_events(samples, times, pos_encoding, temperature=temperature)

        scheduled = torch.sum(scheduled, dim=1, keepdim=True)
        return scheduled[:, :, n_samples:]


@numpy_conjure(collection)
def get_samples(path: str, samplerate: int) -> np.ndarray:
    samples, sr = librosa.load(path, sr=samplerate, mono=True)
    return samples


def dataset(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        window_size: int = 1024,
        n_pos_encoding_channels: int = 64) -> Generator[DatasetBatch, None, None]:

    samples = get_samples(path, 22050)
    n_samples = len(samples)

    step_size = window_size // 2
    n_frames = n_samples // step_size
    n_segment_frames = n_segment_samples // step_size

    print(
        f'operating on {n_samples} samples {n_samples / 22050} seconds with and {n_frames} frames')

    while True:
        batch = torch.zeros(1, 1, n_segment_samples, device=device)

        # we'll return positions that are twice as long as the segment, beginning one full segment
        # earlier than the samples
        pos = torch.zeros(1, n_pos_encoding_channels + 1, n_segment_frames * 2, device=device)

        start_index = np.random.randint(n_frames - (n_segment_frames - 1))
        end_index = start_index + n_segment_frames
        early_index = start_index - n_segment_frames

        chunk = torch.from_numpy(samples[start_index * step_size:end_index * step_size]).to(device)
        batch[:, 0, :] = chunk
        pos[:, :, :] = pos_encoding(early_index, end_index, n_frames, n_pos_encoding_channels, device)

        yield pos, batch, n_samples, start_index, end_index


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def train(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        window_size: int = 1024,
        n_pos_encoding_channels: int = 64,
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

    iterator = dataset(
            path=path,
            device=device,
            n_segment_samples=n_segment_samples,
            window_size=window_size,
            n_pos_encoding_channels=n_pos_encoding_channels)

    _, _, total_samples, _, _ = next(iterator)

    model = Model(
        total_samples=total_samples,
        n_frames=total_samples // (window_size // 2),
        samplerate=22050,
        event_latent_dim=8,
        events_per_second=4,
        window_size=window_size,
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        pos_encoding_channels=n_pos_encoding_channels + 1,
        n_segment_samples=n_segment_samples
    ).to(device)
    optim = Adam(model.parameters(), lr=1e-3)

    model_params = count_parameters(model)

    tmp_schedule = torch.linspace(1, 1e-4, 10000)

    for i, pair in enumerate(iterator):

        pos, samples, total_samples, start_frame, end_frame = pair

        encoding(pos[0])

        # log original audio
        orig_audio(max_norm(samples))

        optim.zero_grad()
        try:
            tmp = tmp_schedule[i]
        except IndexError:
            tmp = 1e-4

        try:
            recon = model.forward(pos, start_frame, end_frame, temperature=tmp)
        except ValueError:
            continue

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
        hidden_channels=128,
        n_layers=4)
