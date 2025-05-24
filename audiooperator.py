from typing import Literal, Tuple
from conjure import loggers, LmdbCollection, NumpyDeserializer, SupportedContentType, \
    NumpySerializer, serve_conjure
from torch import nn
import torch
import numpy as np
from modules import LinearOutputStack, unit_norm, max_norm
from modules.normal_pdf import gamma_pdf
from modules.reds import interpolate_last_axis
from util import make_initializer, encode_audio
from itertools import count
from torch.nn import functional as F

import matplotlib

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

BandType = Literal['linear', 'geometric']

initializer = make_initializer(0.02)

n_pos_encoding_bands = 64
min_pos_encoding_freq = 0.01
max_pos_encoding_freq = 128

device = torch.device('cpu')


def generate_envelope(n_envelopes: int, resolution: int, device: torch.device):
    a = 1e-12 + torch.zeros(n_envelopes, 1, device=device).uniform_(0, 10)
    b = 1e-12 + torch.zeros(n_envelopes, 1, device=device).uniform_(0, 10)
    env = gamma_pdf(a, b, resolution, normalize=True)
    return env


class PosEncoder(nn.Module):

    def __init__(
            self,
            n_bands: int,
            band_type: BandType = 'linear',
            max_freq: float = 128,
            min_freq: float = 0.01):

        super().__init__()
        self.n_bands = n_bands

        if band_type == 'linear':
            freqs = np.linspace(min_freq, max_freq, num=n_bands)
        else:
            freqs = np.geomspace(max_freq, max_freq, num=n_bands)

        freqs = torch.from_numpy(freqs).float()
        self.register_buffer('freqs', freqs.view(1, 1, n_bands, 1))

    @property
    def total_bands(self):
        return self.n_bands * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_events, time = x.shape
        s = torch.sin(x[:, :, None, :] * self.freqs)
        c = torch.cos(x[:, :, None, :] * self.freqs)
        x = torch.zeros(batch, n_events, self.total_bands, time, device=x.device)
        x[:, :, ::2, :] = s
        x[:, :, 1::2, :] = c
        return x


pos_encoding_model = PosEncoder(
    n_bands=n_pos_encoding_bands,
    min_freq=min_pos_encoding_freq,
    max_freq=max_pos_encoding_freq,
    band_type='linear',
).to(device)


def generate_training_batch(
        n_examples: int,
        resolution: int,
        envelope_resolution: int,
        device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Each entry in the infinite/synthetic dataset will consist of the following
    pairs:

        inputs:  event_time, event_duration, event_envelope, timespan
        targets:
    """
    start_times = torch.zeros(n_examples, device=device).uniform_(0, 1)
    durations = torch.zeros(n_examples, device=device).uniform_(1e-3, 1)
    envelopes = generate_envelope(n_examples, envelope_resolution, device)

    start_samples = torch.floor(start_times * resolution).int()
    duration_samples = torch.floor(durations * resolution).int()
    end_samples = start_samples + duration_samples

    target = torch.zeros(n_examples, 1, resolution, device=device)

    for i in range(n_examples):
        rasterized_envelope = interpolate_last_axis(
            envelopes[i: i + 1, ...], desired_size=duration_samples[i])

        last_sample = end_samples[i]
        diff = max(0, last_sample.item() - resolution)
        target_size = target[i, :, start_samples[i].item(): end_samples[i].item() - diff].shape[-1]
        target[i, :, start_samples[i].item(): end_samples[i].item() - diff] \
            = rasterized_envelope[..., :target_size]

    return target, start_times, durations, envelopes


class UnitNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return unit_norm(x)

class Model(nn.Module):

    def __init__(
            self,
            envelope_resolution: int,
            latent_dim: int,
            pos_encoding_dim: int,
            model_dim: int):
        super().__init__()
        self.envelope_resolution = envelope_resolution
        self.pos_encoding_dim = pos_encoding_dim
        self.model_dim = model_dim

        self.embed_envelope = nn.Linear(envelope_resolution, model_dim)
        self.embed_start = nn.Linear(pos_encoding_dim, model_dim)
        self.embed_duration = nn.Linear(pos_encoding_dim, model_dim)
        self.embed_properties = nn.Linear(latent_dim, model_dim)
        self.embed_positions = nn.Linear(pos_encoding_dim, model_dim)

        self.weighting = nn.Parameter(torch.zeros(1).fill_(1))

        self.network = LinearOutputStack(
            channels=model_dim,
            layers=5,
            out_channels=1,
            in_channels=model_dim,
            activation=lambda x: torch.selu(x),
            # norm=lambda channels: nn.LayerNorm([channels,])
        )

        self.apply(initializer)

    def forward(
            self,
            start: torch.Tensor,
            duration: torch.Tensor,
            envelope: torch.Tensor,
            event_properties: torch.Tensor,
            pos: torch.Tensor) -> torch.Tensor:

        resolution = pos.shape[-1]

        batch, n_events = start.shape[:2]
        start = start.view(batch, n_events, self.pos_encoding_dim)
        duration = duration.view(batch, n_events, self.pos_encoding_dim)

        batch, n_events, _ = start.shape

        start = self.embed_start(start)
        duration = self.embed_duration(duration)
        envelope = self.embed_envelope(envelope)
        props = self.embed_properties(event_properties)
        pos = self.embed_positions(pos.permute(0, 1, 3, 2))\
            .view(batch, resolution, self.model_dim)


        x = start + duration + envelope + props + pos

        x = self.network(x)
        x = x.view(batch, n_events, resolution)
        return x


def train_model(
        batch_size: int,
        n_samples: int,
        samplerate: int = 22050,
        device: torch.device = torch.device('cpu'),
        overfit: bool = False,
        envelope_window_size: int = 512,
        envelope_step_size: int = 512):

    n_events = 1
    n_bands = 256
    envelope_resolution = 128
    latent_dim = 64

    model_dim = 128

    pos_encoding_model = PosEncoder(
        n_bands=n_bands, band_type='linear', max_freq=2048).to(device)

    model = Model(
        envelope_resolution=envelope_resolution,
        latent_dim=latent_dim,
        pos_encoding_dim=pos_encoding_model.total_bands,
        model_dim=model_dim).to(device)


    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.data.cpu().numpy()

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    collection = LmdbCollection(path='audiooperator')

    log_target, log_recon = loggers(
        ['target', 'recon'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    log_audio, = loggers(
        ['audio'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure(
        [
            log_target,
            log_recon,
            log_audio
        ], port=9999, n_workers=1)


    for i in count():
        optim.zero_grad()
        

        with torch.no_grad():

            if not overfit or i == 0:
                print(f'Generating batch for iteration {i}')
                target, start_times, durations, envelopes = generate_training_batch(
                    n_examples=batch_size,
                    resolution=n_samples,
                    envelope_resolution=envelope_resolution,
                    device=device)

                # generate random instrument "latents"
                latents = torch.zeros(batch_size, n_events, latent_dim, device=device).uniform_(-1, 1)

                # generate times
                times = torch\
                    .linspace(0, 1, n_samples, device=device)\
                    .view(1, 1, n_samples)\
                    .repeat(batch_size, 1, 1)

                times = pos_encoding_model.forward(times)

                embedded_starts = pos_encoding_model.forward(start_times.view(batch_size, 1, 1))
                embedded_durations = pos_encoding_model.forward(durations.view(batch_size, 1, 1))


        recon = model.forward(
            embedded_starts,
            embedded_durations,
            envelopes,
            latents,
            times)

        log_audio(max_norm(torch.sum(recon, dim=0).view(-1)))

        pool_window_size = envelope_window_size
        pool_step_size = envelope_step_size
        target_downsampled = F.avg_pool1d(
            torch.abs(target), pool_window_size, pool_step_size, pool_step_size)
        recon_downsampled = F.avg_pool1d(
            torch.abs(recon), pool_window_size, pool_step_size, pool_step_size)


        log_target(target_downsampled[:, 0, :])
        log_recon(recon_downsampled[:, 0, :])

        target_on = target_downsampled > 0
        target_off = target_downsampled == 0

        off_count = target_off.sum()
        on_count = target_on.sum()

        off = torch.abs((target_downsampled * target_off) - (recon_downsampled * target_off)).sum() / off_count
        on = torch.abs((target_downsampled * target_on) - (recon_downsampled * target_on)).sum() / on_count

        loss = off + on

        loss.backward()
        optim.step()

        print(i, loss.item())



if __name__ == '__main__':
    train_model(
        batch_size=8,
        n_samples=2**15,
        samplerate=22050,
        envelope_window_size=512,
        envelope_step_size=256,
        overfit=True,
        device=torch.device('cuda'))
