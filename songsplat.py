import argparse
from typing import Generator, Tuple, Union

import librosa
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, numpy_conjure, loggers, serve_conjure, SupportedContentType, NumpySerializer, \
    NumpyDeserializer
from modules import stft, max_norm, sparse_softmax, interpolate_last_axis, flattened_multiband_spectrogram, \
    iterative_loss
from modules.atoms import unit_norm
from modules.infoloss import CorrelationLoss
from modules.transfer import fft_convolve, freq_domain_transfer_function_to_resonance
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import encode_audio, make_initializer, device
from torch.nn import functional as F

collection = LmdbCollection('songsplat')

DatasetBatch = Tuple[torch.Tensor, torch.Tensor, int, int, int]

init = make_initializer(0.05)



def decaying_noise(n_items: int, n_samples: int, low_exp: int, high_exp: int, device: torch.device):
    t = torch.linspace(1, 0, n_samples, device=device)
    pos = torch.zeros(n_items, device=device).uniform_(low_exp, high_exp)
    noise = torch.zeros(n_items, n_samples, device=device).uniform_(-1, 1)
    return (t[None, :] ** pos[:, None]) * noise


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def time_vector(t: torch.Tensor, total_samples: int, n_channels: int, device: torch.device) -> torch.Tensor:
    freqs = torch.linspace(1, total_samples // 2, n_channels // 2, device=device)[:, None]
    scaled = torch.linspace(1, 0, n_channels // 2, device=device)

    s = torch.sin(t * freqs) * scaled[:, None]
    c = torch.cos(t * freqs) * scaled[:, None]

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
    return flattened_multiband_spectrogram(x, { 'sm': (64, 16)})
    # return stft(x, 2048, 256, pad=True)

def loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    r = transform(recon)
    t = transform(target)
    l = torch.abs(r - t).sum()
    return l


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


class EventGenerator(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            n_frames: int,
            window_size: int,
            raw_sample_mode: bool = False):

        super().__init__()
        self.raw_sample_mode = raw_sample_mode
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.window_size = window_size
        self.window = HalfLappedWindowParams(window_size)
        self.n_samples = self.window.total_samples(n_frames)

        if raw_sample_mode:
            n_items = 512
            self.net = nn.Linear(latent_dim, n_items, bias=False)
            self.items = nn.Parameter(
                decaying_noise(n_items, self.n_samples, 10, 100, device=device) * 1e-3)
        else:
            expressivity = 2
            n_coeffs = self.window.n_coeffs
            self.expressivity = expressivity
            self.n_coeffs = n_coeffs

            n_envelopes = 32
            n_dither = 16
            n_resonances = 64
            n_phase = 8
            n_deformations = 16

            # self.to_noise = nn.Linear(latent_dim, n_frames // 4)
            # self.to_dither = nn.Linear(latent_dim, n_coeffs * expressivity)
            # self.to_coeffs = nn.Linear(latent_dim, n_coeffs * expressivity)
            # self.to_phase = nn.Linear(latent_dim, n_coeffs * expressivity)
            # self.to_deformation = nn.Linear(latent_dim, n_frames * expressivity)

            self.to_noise = SimpleLookup(latent_dim, n_envelopes, n_frames // 4, 1)
            self.to_dither = SimpleLookup(latent_dim, n_dither, n_coeffs, expressivity)
            self.to_coeffs = SimpleLookup(latent_dim, n_resonances, n_coeffs, expressivity)
            self.to_phase = SimpleLookup(latent_dim, n_phase, n_coeffs, expressivity)
            self.to_deformation = SimpleLookup(latent_dim, n_deformations, n_frames, expressivity)

            self.to_amps = nn.Linear(latent_dim, n_coeffs * expressivity)
            self.to_mix = nn.Linear(latent_dim, 2)


    @property
    def total_samples(self):
        return self.window.total_samples(self.n_frames)

    def forward(self, events: torch.Tensor) -> torch.Tensor:
        batch, n_events, dim = events.shape

        if self.raw_sample_mode:
            x = self.net(events)
            x = torch.relu(x)
            x = x @ self.items
            return x
        else:
            dither = self.to_dither(events).view(batch, n_events * self.expressivity, self.n_coeffs)
            decay = self.to_coeffs(events).view(batch, n_events * self.expressivity, self.n_coeffs)
            amps = self.to_amps(events).view(batch, n_events * self.expressivity, self.n_coeffs)
            phase = self.to_phase(events).view(batch, n_events * self.expressivity, self.n_coeffs)

            resonances = freq_domain_transfer_function_to_resonance(
                window_size=self.window_size,
                coeffs=torch.sigmoid(decay) * 0.9999,
                n_frames=self.n_frames,
                apply_decay=True,
                start_phase=torch.tanh(phase) *  np.pi,
                start_mags=amps ** 2,
                log_space_scan=True,
                phase_dither=torch.tanh(dither.view(-1, 1, self.n_coeffs))
            )

            resonances = resonances.view(batch, n_events, self.expressivity, self.n_samples)

            n = self.to_noise(events).view(batch, n_events, 1, self.n_frames // 4)
            n = n ** 2
            n = interpolate_last_axis(n, desired_size=self.n_samples // 4)
            n = ensure_last_axis_length(n, desired_size=self.n_samples)
            n = n * torch.zeros_like(n).uniform_(-1, 1)

            resonances = fft_convolve(resonances, n)

            deformations = self.to_deformation(events).view(batch, n_events, self.expressivity, self.n_frames)
            deformations = torch.cumsum(deformations, dim=-1)
            deformations = torch.relu(deformations)
            deformations = interpolate_last_axis(deformations, desired_size=self.n_samples)


            x = resonances * deformations
            x = torch.sum(x, dim=2)

            mx = self.to_mix(events)

            stacked = torch.stack([n.view(1, -1, self.n_samples), x], dim=-1)

            x = stacked * torch.softmax(mx.view(1, -1, 1, 2), dim=-1)

            x = torch.sum(x, dim=-1)
            return x




def schedule_events(
        events: torch.Tensor,
        times: torch.Tensor,
        pos_encodings: torch.Tensor,
        temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:

    batch, n_events, n_samples = events.shape
    n_events, pos_encoding_dim = times.shape
    _, n_frames, pos_encoding_dim = pos_encodings.shape

    # print(pos_encodings.shape, times.shape)

    pos_encodings = pos_encodings.view(pos_encoding_dim, -1)


    sim = times @ pos_encodings.T

    # indices = torch.argmax(sim, dim=-1)
    # print(indices)

    # sched = F.gumbel_softmax(sim, tau=temperature, hard=True, dim=-1)
    sched = sparse_softmax(sim, normalize=True)
    # sched = torch.softmax(sim, dim=-1)

    mx, indices = torch.max(sched, dim=-1)

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
        times = torch.zeros(self.total_events, device=device).uniform_(0, 1)

        orig_times = time_vector(times, self.total_samples, pos_encoding_channels, device)
        self.register_buffer('orig_times', orig_times)
        self.times = nn.Parameter(orig_times.clone())

        n_event_frames = n_segment_samples // self.window_params.step_size

        self.generator = EventGenerator(
            event_latent_dim, n_event_frames, window_size)

        self.apply(init)

    @property
    def time_change(self):
        return torch.abs(self.orig_times - self.times).sum()

    def forward(
            self,
            pos_encoding: torch.Tensor,
            start_frame: int,
            end_frame: int,
            temperature: float) -> Tuple[torch.Tensor, torch.Tensor]:

        n_frames = end_frame - start_frame
        n_samples = self.window_params.total_samples(n_frames)
        early_frame = start_frame - n_frames

        start_rel = early_frame / self.n_frames
        stop_rel = end_frame / self.n_frames

        if start_rel < 0:
            raise ValueError('skipping too-early segment')

        # print(n_frames, n_samples, start_frame, end_frame, early_frame, start_rel, stop_rel)

        # use the linear [0-1] channel to locate events that will affect this interval
        t = self.times[0, :]

        mask = (t > start_rel) & (t < stop_rel)

        events = self.events[mask]
        times = self.times.T[mask]

        if events.shape[0] == 0:
            raise ValueError('no events')

        samples = self.generator.forward(events[None, ...])

        # print('EVENTS', events.shape, times.shape, samples.shape)

        samples = samples.view(1, -1, n_samples)
        padding = torch.zeros_like(samples)
        samples = torch.cat([samples, padding], dim=-1)

        scheduled, mx = schedule_events(samples, times, pos_encoding, temperature=temperature)

        return scheduled[:, :, n_samples:], mx


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
        n_pos_encoding_channels: int = 64):
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
        event_latent_dim=32,
        events_per_second=8,
        window_size=window_size,
        pos_encoding_channels=n_pos_encoding_channels + 1,
        n_segment_samples=n_segment_samples
    ).to(device)

    optim = Adam(model.parameters(), lr=1e-4)

    model_params = count_parameters(model)

    tmp_schedule = torch.linspace(1, 1e-4, 10000)

    for i, pair in enumerate(iterator):

        pos, samples, total_samples, start_frame, end_frame = pair

        # encoding(model.times)

        # log original audio
        orig_audio(max_norm(samples))

        optim.zero_grad()
        try:
            tmp = tmp_schedule[i]
        except IndexError:
            tmp = 1e-4

        try:
            recon, mx = model.forward(pos, start_frame, end_frame, temperature=tmp)
        except ValueError as e:
            print(e)
            continue

        recon_summed = torch.sum(recon, dim=1, keepdim=True)

        # log recon audio
        recon_audio(max_norm(recon_summed))


        # print(mx)
        confidence_loss = torch.abs(0.99 - mx).sum()
        l = iterative_loss(samples, recon, transform, ratio_loss=False) + confidence_loss

        # l = loss(recon_summed, samples) + confidence_loss
        l.backward()
        optim.step()
        print(i, l.item(), f'Compression Ratio: {(model_params / total_samples):.2f}, change: {model.time_change.item():.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    train(
        args.path,
        torch.device('cuda'),
        n_segment_samples=2 ** 16,
        window_size=1024,
        n_pos_encoding_channels=4096)
