import argparse
import os.path
from random import choice
from typing import Generator, Tuple, Union

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure, SupportedContentType, NumpySerializer, \
    NumpyDeserializer
from modules import max_norm, sparse_softmax, interpolate_last_axis, iterative_loss, \
    stft
from modules.transfer import fft_convolve, damped_harmonic_oscillator
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import encode_audio, make_initializer, device

collection = LmdbCollection('songsplat')

DatasetBatch = Tuple[torch.Tensor, int, int, int, int]

init = make_initializer(0.05)

# TODO: try this with scalar FFT-based scheduling

# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def transform(x: torch.Tensor) -> torch.Tensor:
    # return flattened_multiband_spectrogram(x, {'sm': (64, 16)})
    return stft(x, 2048, 256, pad=True)


def reconstruction_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = transform(a)
    b = transform(b)
    return torch.abs(a - b).sum()

def nearest_power_of_two(n: float) -> int:
    x = np.log2(n)
    x = np.ceil(x)
    return x


class SimpleLookup(nn.Module):
    def __init__(
            self,
            latent: int,
            n_items: int,
            n_samples: int,
            expressivity: int):
        super().__init__()
        self.latent = latent
        self.n_items = n_items
        self.n_samples = n_samples
        self.expressivity = expressivity

        self.from_latent = nn.Linear(latent, n_items * expressivity)
        self.items = nn.Parameter(torch.zeros(n_items, n_samples).uniform_(-0.02, 0.02))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_items, dim = x.shape
        x = self.from_latent(x)
        x = torch.relu(x)
        x = x.view(batch, n_items, self.expressivity, -1)
        x = x @ self.items
        return x


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


class SpectralResonanceBlock(nn.Module):

    def __init__(self, n_samples: int, n_resonances: int, expressivity: int):
        super().__init__()
        self.n_samples = n_samples
        self.n_resonances = n_resonances
        self.expressivity = expressivity

        self.n_coeffs = n_samples // 2 + 1
        self.total_coeffs = self.n_coeffs * 2

        self.res = nn.Parameter(torch.zeros(n_resonances, expressivity, self.n_coeffs, 2).uniform_(-1, 1))


    def forward(self):
        x = torch.view_as_complex(self.res)
        x = torch.fft.irfft(x, dim=-1, norm='ortho')
        return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples) #* ramp[None, None, None, None, :]


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

        self.mass = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-2, 2))

        self.damping = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(0.5, 1.5))

        self.tension = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(4, 9))

        self.initial_displacement = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-1, 2))

        self.amplitudes = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity, 1) \
                .uniform_(-1, 1))

    def _materialize_resonances(self, device: torch.device):
        time = torch.linspace(0, 10, self.n_samples, device=device) \
            .view(1, 1, 1, self.n_samples)

        x = damped_harmonic_oscillator(
            time=time,
            mass=torch.sigmoid(self.mass[..., None]) * 2,
            damping=torch.sigmoid(self.damping[..., None]) * 30,
            tension=10 ** self.tension[..., None],
            initial_displacement=self.initial_displacement[..., None],
            initial_velocity=0
        )

        x = x.view(self.n_oscillators, self.n_resonances, self.expressivity, self.n_samples)
        x = torch.tanh(x * self.amplitudes)
        x = torch.sum(x, dim=0)

        return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples) #* ramp[None, None, None, None, :]

    def forward(self) -> torch.Tensor:
        return self._materialize_resonances(self.damping.device)


class SampleEventGenerator(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            n_frames: int,
            window_size: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.window_size = window_size
        self.window = HalfLappedWindowParams(window_size)
        self.n_samples = self.window.total_samples(n_frames)

        self.to_samples = nn.Linear(latent_dim, self.n_samples, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.to_samples(x)

class EventGenerator(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            n_frames: int,
            window_size: int):

        super().__init__()
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.window_size = window_size
        self.window = HalfLappedWindowParams(window_size)
        self.n_samples = self.window.total_samples(n_frames)

        expressivity = 4
        n_coeffs = self.window.n_coeffs
        self.expressivity = expressivity
        self.n_coeffs = n_coeffs

        self.base_resonance = 0.02
        self.max_resonance = 0.99
        self.resonance_span = self.max_resonance - self.base_resonance

        n_envelopes = 64
        n_resonances = 128
        n_deformations = 32

        self.n_resonances = n_resonances

        # self.resonances = DampedHarmonicOscillatorBlock(
        #     n_samples=self.n_samples,
        #     n_oscillators=32,
        #     n_resonances=n_resonances,
        #     expressivity=1)

        self.resonances = SpectralResonanceBlock(n_samples=self.n_samples, n_resonances=self.n_resonances, expressivity=1)

        self.to_resonance = nn.Linear(latent_dim, n_resonances * self.expressivity)

        # note: the noise envelope is 4x the frame rate, as it is upsampled
        # to 1/4 the overall number of frames
        self.to_noise = SimpleLookup(latent_dim, n_envelopes, n_frames, 1)
        # self.to_noise = nn.Linear(latent_dim, n_frames, bias=False)

        self.to_deformation = SimpleLookup(latent_dim, n_deformations, n_frames, expressivity)

        self.to_mix = nn.Linear(latent_dim, 2)
        self.to_loudness = nn.Linear(latent_dim, 1)

    @property
    def total_samples(self):
        return self.window.total_samples(self.n_frames)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        batch, n_events, dim = events.shape

        # materialize resonances into sample space (n_resonances, n_samples)
        resonances = self.resonances.forward().view(self.n_resonances, -1)
        # produce a selection of the resonances
        x = self.to_resonance(events)
        x = x.view(batch, n_events * self.expressivity, self.n_resonances)
        resonances = x @ resonances

        resonances = resonances.view(batch, n_events, self.expressivity, self.n_samples)

        n = self.to_noise(events).view(batch, n_events, 1, self.n_frames)

        n = interpolate_last_axis(n, desired_size=self.n_samples // 4)
        n = ensure_last_axis_length(n, desired_size=self.n_samples)
        n = n * torch.zeros_like(n).uniform_(-1, 1)

        resonances = fft_convolve(resonances, n)

        deformations = self.to_deformation(events).view(batch, n_events, self.expressivity, self.n_frames)
        deformations = torch.cumsum(deformations, dim=-1)
        deformations = torch.softmax(deformations, dim=2)
        deformations = interpolate_last_axis(deformations, desired_size=self.n_samples)

        x = resonances * deformations
        x = torch.sum(x, dim=2)

        mx = self.to_mix(events)

        stacked = torch.stack([n.view(1, -1, self.n_samples), x], dim=-1)

        x = stacked * torch.softmax(mx.view(1, -1, 1, 2), dim=-1)

        x = torch.sum(x, dim=-1)

        a = self.to_loudness(events)

        x = x * torch.abs(a)

        return x


def schedule_events(
        events: torch.Tensor,
        times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, n_events, n_samples = events.shape

    sched = times

    mx, indices = torch.max(sched, dim=-1)
    print(sched.shape, indices.max().item(), mx.mean().item())

    # OPERATION: schedule
    sched = upsample_with_holes(sched, desired_size=n_samples)
    scheduled = fft_convolve(events, sched)
    return scheduled, mx


class Model(nn.Module):

    def __init__(
            self,
            total_samples: int,
            n_frames: int,
            samplerate: int,
            event_latent_dim: int,
            window_size: int,
            n_segment_samples: int,
            events_per_second: float):
        super().__init__()

        self.n_segment_samples = n_segment_samples
        self.n_frames = n_frames
        self.window_size = window_size
        self.window_params = HalfLappedWindowParams(window_size)
        self.event_latent_dim = event_latent_dim
        self.total_samples = total_samples
        self.samplerate = samplerate
        self.total_seconds = total_samples / samplerate
        self.events_per_second = events_per_second

        self.total_events = int(self.total_seconds * self.events_per_second)

        self.events = nn.Parameter(torch.zeros(self.total_events, self.event_latent_dim).uniform_(-0.01, 0.01))

        # TODO: Try out with a hierarchical build-up
        # a hierarchical buildup will require that the number of samples (and by extension, frames)
        # must be a power of 2
        self.times = nn.Parameter(torch.zeros(self.total_events, self.n_frames).uniform_(-0.01, 0.01))

        n_event_frames = n_segment_samples // self.window_params.step_size

        self.generator = EventGenerator(
            event_latent_dim, n_event_frames, window_size)

        self.apply(init)

    def materialize_times(self):
        return self.times


    @property
    def compression_ratio(self):
        # we only need a single float value, representing event time
        n_params = \
            (self.total_events * self.event_latent_dim) \
            + self.total_events + count_parameters(self.generator)
        return n_params / self.total_samples

    def generate_random(self, n_events):
        # events = torch.zeros(n_events, self.event_latent_dim, device=device) \
        #     .normal_(self.events.mean().item(), self.events.std().item())

        indices = np.random.permutation(self.total_events)[:n_events, ...]
        events = self.events[indices]

        samples = self.generator.forward(events[None, ...])

        samples = samples.view(1, -1, self.n_segment_samples)
        samples = ensure_last_axis_length(samples, desired_size=self.n_segment_samples * 2)

        times = torch.zeros(n_events, (self.n_segment_samples * 2) // self.window_size, device=device).uniform_(-1, 1)
        times = sparse_softmax(times, dim=-1, normalize=True)

        scheduled, _ = schedule_events(samples, times)
        scheduled = scheduled[:, :, :self.n_segment_samples]
        return scheduled

    def forward(
            self,
            start_frame: int,
            end_frame: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_frames = end_frame - start_frame
        n_samples = self.window_params.total_samples(n_frames)
        early_frame = start_frame - n_frames

        start_rel = early_frame / self.n_frames
        stop_rel = end_frame / self.n_frames

        if start_rel < 0:
            raise ValueError('skipping too-early segment')

        # OPERATION: range query
        times = self.materialize_times()
        t = sparse_softmax(times, dim=-1, normalize=True)

        oh = t
        with torch.no_grad():
            t = torch.argmax(t, dim=-1) / self.n_frames
            mask = (t > start_rel) & (t < stop_rel)

        events = self.events[mask]
        times = times[mask]

        if events.shape[0] == 0:
            raise ValueError('no events')

        samples = self.generator.forward(events[None, ...])

        samples = samples.view(1, -1, n_samples)
        samples = ensure_last_axis_length(samples, desired_size=n_samples * 2)

        # OPERATION: offset/shift/translate
        scheduled, mx = schedule_events(samples, times[:, early_frame: end_frame])

        return scheduled[:, :, n_samples:], mx, oh


@numpy_conjure(collection)
def get_samples(path: str, samplerate: int) -> np.ndarray:
    samples, sr = librosa.load(path, sr=samplerate, mono=True)
    return samples[:2**19]


def dataset(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        window_size: int = 1024) -> Generator[DatasetBatch, None, None]:
    samples = get_samples(path, 22050)
    n_samples = len(samples)

    step_size = window_size // 2
    n_frames = n_samples // step_size
    n_segment_frames = n_segment_samples // step_size

    print(
        f'operating on {n_samples} samples {n_samples / 22050} seconds with and {n_frames} frames')

    while True:
        batch = torch.zeros(1, 1, n_segment_samples, device=device)

        start_index = np.random.randint(n_frames - (n_segment_frames - 1))
        end_index = start_index + n_segment_frames

        chunk = torch.from_numpy(samples[start_index * step_size:end_index * step_size]).to(device)
        batch[:, 0, :] = chunk

        yield batch, n_samples, start_index, end_index, n_frames


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def train(
        path: str,
        device: torch.device,
        n_segment_samples: int = 2 ** 15,
        window_size: int = 1024):

    recon_audio, orig_audio, rnd = loggers(
        ['recon', 'orig', 'rnd'],
        'audio/wav',
        encode_audio,
        collection)

    events, = loggers(
        ['events'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    serve_conjure([
        orig_audio,
        recon_audio,
        events,
        # times,
        rnd
    ], port=9999, n_workers=1, web_components_version='0.0.101')

    if os.path.isdir(path):
        fn = choice(os.listdir(path))
        path = os.path.join(path, fn)
        print(f'Chose {path}')

    iterator = dataset(
        path=path,
        device=device,
        n_segment_samples=n_segment_samples,
        window_size=window_size)

    _, total_samples, _, _, _ = next(iterator)


    model = Model(
        total_samples=total_samples,
        n_frames=total_samples // (window_size // 2),
        samplerate=22050,
        event_latent_dim=32,
        events_per_second=4,
        window_size=window_size,
        n_segment_samples=n_segment_samples
    ).to(device)

    optim = Adam(model.parameters(), lr=1e-3)


    for i, pair in enumerate(iterator):

        samples, total_samples, start_frame, end_frame, n_frames = pair

        events(max_norm(model.events.data))

        # log original audio
        orig_audio(max_norm(samples))

        optim.zero_grad()

        try:
            recon, mx, positions = model.forward(start_frame, end_frame)
        except ValueError as e:
            print(e)
            continue

        # times(positions)
        recon_summed = torch.sum(recon, dim=1, keepdim=True)

        # log recon audio
        recon_audio(max_norm(recon_summed))


        # iterative loss seems to be important for producing
        # playable events
        # l = iterative_loss(samples, recon, transform, ratio_loss=False, sort_channels=True)
        l = reconstruction_loss(samples, recon)
        l.backward()
        optim.step()
        print(i, l.item(),
              f'N Frames: {n_frames}, Compression Ratio: {(model.compression_ratio):.2f}')

        with torch.no_grad():
            r = model.generate_random(n_events=6)
            r = torch.sum(r, dim=1, keepdim=True)
            r = max_norm(r)
            rnd(r)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    train(
        args.path,
        torch.device('cuda'),
        n_segment_samples=2 ** 16,
        window_size=2048)
