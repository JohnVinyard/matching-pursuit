import torch
import numpy as np
from conjure import LmdbCollection, loggers, serve_conjure
from torch import nn
from torch.nn import functional as F

from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import unit_norm, iterative_loss, flattened_multiband_spectrogram, sparse_softmax, max_norm, \
    gammatone_filter_bank, stft
from itertools import count
from torch.optim import Adam
from typing import Tuple

from modules.eventgenerators.splat import SplattingEventGenerator
from modules.infoloss import CorrelationLoss
from modules.multiheadtransform import MultiHeadTransform
from modules.transfer import hierarchical_dirac, fft_convolve, make_waves
from util import device, make_initializer
from util.music import musical_scale_hz

initializer = make_initializer(0.05)


class OverfitHierarchicalEvents(nn.Module):
    def __init__(
            self,
            n_samples: int,
            samplerate: int,
            n_events: int,
            context_dim: int):
        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.n_events = n_events
        self.context_dim = context_dim

        event_levels = int(np.log2(n_events))
        total_levels = int(np.log2(n_samples))

        self.event_levels = event_levels

        starting_time_bits = total_levels - event_levels

        self.event_generator = SplattingEventGenerator(
            n_samples=n_samples,
            samplerate=samplerate,
            n_resonance_octaves=64,
            n_frames=n_samples // 256,
            hard_reverb_choice=False,
            hierarchical_scheduler=True,
            wavetable_resonance=True
        )
        self.transform = MultiHeadTransform(
            self.context_dim, hidden_channels=128, shapes=self.event_generator.shape_spec, n_layers=1)

        self.event_time_dim = int(np.log2(self.n_samples))

        self.event_vectors = nn.Parameter(torch.zeros(1, 2, self.context_dim).uniform_(-1, 1))
        self.hierarchical_event_vectors = nn.ParameterDict(
            {str(i): torch.zeros(1, 2, self.context_dim).uniform_(-1, 1) for i in range(event_levels - 1)})

        self.times = nn.Parameter(
            torch.zeros(1, 2, total_levels, 2).uniform_(-1, 1))
        self.hierarchical_time_vectors = nn.ParameterDict(
            {str(i): torch.zeros(1, (2 ** (i + 2)), total_levels, 2).uniform_(-1, 1) for i in range(event_levels - 1)})

        self.apply(initializer)

    @property
    def normalized_atoms(self):
        return unit_norm(self.atoms, dim=-1)

    def forward(self) -> torch.Tensor:
        events = self.event_vectors.clone()
        times = self.times.clone()

        for i in range(self.event_levels - 1):
            events = \
                events.view(1, -1, 1, self.context_dim) \
                + self.hierarchical_event_vectors[str(i)].view(1, 1, 2, self.context_dim)
            events = events.view(1, -1, self.context_dim)
            batch, n_events, n_bits, _ = times.shape
            times = times.view(batch, n_events, 1, n_bits, 2).repeat(1, 1, 2, 1, 1).view(batch, n_events * 2, n_bits, 2)
            times = times + self.hierarchical_time_vectors[str(i)]
            # times = torch.cat([times, self.hierarchical_time_vectors[str(i)]], dim=2)

        params = self.transform.forward(events)
        events = self.event_generator.forward(**params, times=times)
        return events


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def reconstruction_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    target_spec = loss_transform(target)
    recon_spec = loss_transform(recon)
    loss = torch.abs(target_spec - recon_spec).sum()
    return loss


def overfit():
    n_samples = 2 ** 15
    samplerate = 22050
    n_events = 64
    event_dim = 256

    # Begin: this would be a nice little helper to wrap up
    collection = LmdbCollection(path='hierarchical')
    collection.destroy()
    collection = LmdbCollection(path='hierarchical')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)

    audio = get_one_audio_segment(n_samples, samplerate, device='cpu')
    target = audio.view(1, 1, n_samples).to(device)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=9999, n_workers=1)
    # end proposed helper function

    model = OverfitHierarchicalEvents(
        n_samples, samplerate, n_events, context_dim=event_dim).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    loss_model = CorrelationLoss(n_elements=512).to(device)

    for i in count():
        optim.zero_grad()
        recon = model.forward()

        recon_summed = torch.sum(recon, dim=1, keepdim=True)
        recon_audio(max_norm(recon_summed))

        # loss = iterative_loss(target, recon, loss_transform, ratio_loss=True) #+ loss_model.forward(target, recon_summed)

        loss = loss_model.noise_loss(target, recon_summed)
        # loss = reconstruction_loss(target, recon_summed)
        sparsity_loss = torch.abs(model.event_vectors).sum()
        loss = loss + sparsity_loss

        loss.backward()
        optim.step()
        print(i, loss.item())


if __name__ == '__main__':
    overfit()
