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
            context_dim: int,
            n_layers: int):

        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.n_events = n_events
        self.context_dim = context_dim
        self.n_layers = n_layers

        self.total_levels = int(np.log2(n_samples))

        layers = self.total_levels - self.n_layers
        base_scheduling_bits = layers

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

        self.event_vectors = nn.Parameter(torch.zeros(1, n_events, self.context_dim).uniform_(-1, 1))
        self.times = nn.Parameter(torch.zeros(1, n_events, self.event_time_dim, 2).uniform_(-1, 1))

        self.apply(initializer)

    @property
    def normalized_atoms(self):
        return unit_norm(self.atoms, dim=-1)

    def forward(self) -> torch.Tensor:


        params = self.transform.forward(self.event_vectors)

        events = self.event_generator.forward(**params, times=self.times)

        # first, a hard choice of atom for this event, and pad to the correct size
        # if self.soft:
        #     ac = torch.softmax(self.atom_choice, dim=-1) @ self.normalized_atoms
        # else:
        #     ac = sparse_softmax(self.atom_choice, dim=-1, normalize=True) @ self.normalized_atoms
        #
        # ac = ac.view(-1, self.n_events, self.atom_samples)
        # ac = F.pad(ac, (0, self.n_samples - self.atom_samples))

        # apply amplitudes to the unit norm atoms
        # amps = self.amplitudes ** 2
        # with_amps = ac * amps

        # produce the times from the hierarchical time vectors
        # times = hierarchical_dirac(self.times, soft=self.soft)
        # scheduled = fft_convolve(with_amps, times)
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
        n_samples, samplerate, n_events, context_dim=event_dim, n_layers=6).to(device)
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
        # sparsity_loss = torch.abs(model.event_vectors).sum()
        # loss = loss + sparsity_loss

        loss.backward()
        optim.step()
        print(i, loss.item())


if __name__ == '__main__':
    overfit()