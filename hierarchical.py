import torch
import numpy as np
from conjure import LmdbCollection, loggers, serve_conjure, SupportedContentType, NumpySerializer, NumpyDeserializer
from torch import nn
from torch.nn import functional as F

from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import unit_norm, iterative_loss, flattened_multiband_spectrogram, sparse_softmax, max_norm, \
    gammatone_filter_bank, stft, positional_encoding
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

        self.n_frames = n_samples // 256

        event_levels = int(np.log2(n_events))
        total_levels = int(np.log2(n_samples))

        self.event_levels = event_levels

        self.event_generator = SplattingEventGenerator(
            n_samples=n_samples,
            samplerate=samplerate,
            n_resonance_octaves=64,
            n_frames=n_samples // 256,
            hard_reverb_choice=False,
            hierarchical_scheduler=True,
            wavetable_resonance=True,
        )
        self.transform = MultiHeadTransform(
            self.context_dim, hidden_channels=128, shapes=self.event_generator.shape_spec, n_layers=1)

        self.event_time_dim = int(np.log2(self.n_samples))

        rng = 0.1

        self.event_vectors = nn.Parameter(torch.zeros(1, 2, self.context_dim).uniform_(-rng, rng))
        self.hierarchical_event_vectors = nn.ParameterDict(
            {str(i): torch.zeros(1, 2, self.context_dim).uniform_(-rng, rng) for i in range(event_levels - 1)})

        self.times = nn.Parameter(
            torch.zeros(1, 2, total_levels, 2).uniform_(-rng, rng))
        self.hierarchical_time_vectors = nn.ParameterDict(
            {str(i): torch.zeros(1, (2 ** (i + 2)), total_levels, 2).uniform_(-rng, rng) for i in
             range(event_levels - 1)})

        self.apply(initializer)

    @property
    def normalized_atoms(self):
        return unit_norm(self.atoms, dim=-1)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        events = self.event_vectors.clone()
        times = self.times.clone()

        for i in range(self.event_levels - 1):
            scale = 1 / (i + 1)
            # scale = 1

            # TODO: consider bringing back scaling as we approach the leaves of the tree
            events = \
                events.view(1, -1, 1, self.context_dim) \
                + (self.hierarchical_event_vectors[str(i)].view(1, 1, 2, self.context_dim) * scale)
            events = events.view(1, -1, self.context_dim)

            # TODO: Consider masking lower bits as we approach the leaves of the tree, so that
            # new levels can only _refine_, and not completely move entire branches
            batch, n_events, n_bits, _ = times.shape
            times = times.view(batch, n_events, 1, n_bits, 2).repeat(1, 1, 2, 1, 1).view(batch, n_events * 2, n_bits, 2)
            times = times + (self.hierarchical_time_vectors[str(i)] * scale)

        event_vectors = events

        params = self.transform.forward(events)
        events = self.event_generator.forward(**params, times=times)
        return events, event_vectors, times


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


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def overfit():
    n_samples = 2 ** 16
    samplerate = 22050
    n_events = 64
    event_dim = 16

    # Begin: this would be a nice little helper to wrap up
    collection = LmdbCollection(path='hierarchical')
    collection.destroy()
    collection = LmdbCollection(path='hierarchical')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)

    eventvectors, eventtimes = loggers(
        ['eventvectors', 'eventtimes'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    audio = get_one_audio_segment(n_samples, samplerate, device='cpu')
    target = audio.view(1, 1, n_samples).to(device)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
        eventvectors,
        eventtimes
    ], port=9999, n_workers=1)
    # end proposed helper function

    model = OverfitHierarchicalEvents(
        n_samples, samplerate, n_events, context_dim=event_dim).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    loss_model = CorrelationLoss(n_elements=512).to(device)

    for i in count():
        optim.zero_grad()
        recon, vectors, times = model.forward()

        times = sparse_softmax(times, dim=-1)
        weights = torch.from_numpy(np.array([0, 1])).to(device).float()

        eventvectors(max_norm(vectors[0]))
        t = times[0] @ weights
        eventtimes((t > 0).float())

        recon_summed = torch.sum(recon, dim=1, keepdim=True)
        recon_audio(max_norm(recon_summed))

        # loss = iterative_loss(target, recon, loss_transform, ratio_loss=False) #+ loss_model.forward(target, recon_summed)

        # loss = loss_model.forward(target, recon_summed)
        loss = loss_model.noise_loss(target, recon_summed)
        loss = loss + reconstruction_loss(target, recon_summed)
        sparsity_loss = torch.abs(model.event_vectors).sum()
        loss = loss + sparsity_loss

        loss.backward()
        optim.step()
        print(i, loss.item())


if __name__ == '__main__':
    overfit()
