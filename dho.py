from typing import Tuple, Callable, Any, Union
from itertools import count
import numpy as np
import torch

from conjure import serve_conjure, SupportedContentType, NumpyDeserializer, NumpySerializer, Logger, MetaData
from torch import nn
from torch.optim import Adam

import conjure
from data import get_one_audio_segment
from modules import max_norm, interpolate_last_axis, sparsify, unit_norm, flattened_multiband_spectrogram, \
    fft_frequency_recompose, stft, HyperNetworkLayer
from modules.eventgenerators.overfitresonance import Lookup, flatten_envelope
from modules.infoloss import CorrelationLoss
from modules.transfer import freq_domain_transfer_function_to_resonance, fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import device, encode_audio, make_initializer
from base64 import b64encode
from sklearn.decomposition import DictionaryLearning


init_weights = make_initializer(0.02)





def damped_harmonic_oscillator(
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor,
        initial_velocity: float,
) -> torch.Tensor:
    x = (damping / (2 * mass))
    if torch.isnan(x).sum() > 0:
        print('x first appearance of NaN')

    omega = torch.sqrt(torch.clamp(tension - (x ** 2), 1e-12, np.inf))
    if torch.isnan(omega).sum() > 0:
        print('omega first appearance of NaN')

    phi = torch.atan2(
        (initial_velocity + (x * initial_displacement)),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    return z


class DampedHarmonicOscillatorController(nn.Module):
    def __init__(self, n_samples: int, control_rate: int, n_oscillators: int):
        super().__init__()
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_oscillators = n_oscillators
        self.n_frames = int(n_samples / control_rate)

        self.mass = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-6, 6))

        self.base_damping = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(0, 1))
        self.damping = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-0.01, 0.01))

        self.base_tension = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(0, 1))
        self.tension = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-0.01, 0.01))

        self.initial_displacement = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-1, 1))

        self.times = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-1, 1))

        self.max_time = 10
        self.time_step = 10 // n_samples
        self.register_buffer('base_time', torch.zeros(n_oscillators, 1, n_samples).fill_(self.time_step))

    def forward(self):
        time_modifier = upsample_with_holes(self.times,  desired_size=self.n_samples)
        t = self.base_time + time_modifier
        t = torch.cumsum(t, dim=-1)
        t = torch.clamp(t, 0, np.inf)


        # print(self.base_damping.shape, self.damping.shape)
        damping = interpolate_last_axis(self.base_damping + self.damping, desired_size=self.n_samples)
        tension = interpolate_last_axis(self.base_tension + self.tension, desired_size=self.n_samples)

        # print(t.shape, self.mass.shape, damping.shape, tension.shape, self.initial_displacement.shape)
        x = damped_harmonic_oscillator(
            time=t,
            mass=torch.sigmoid(self.mass),
            damping=torch.sigmoid(damping) * 20,
            tension=10 ** torch.abs(tension),
            initial_displacement=self.initial_displacement,
            initial_velocity=0
        )

        x = torch.sum(x, dim=0, keepdim=True)

        x = max_norm(x)
        return x


# class DampedHarmonicOscillatorBlock(nn.Module):
#     def __init__(
#             self,
#             n_samples: int,
#             n_oscillators: int,
#             n_resonances: int,
#             expressivity: int):
#         super().__init__()
#         self.n_samples = n_samples
#         self.n_oscillators = n_oscillators
#         self.n_resonances = n_resonances
#         self.expressivity = expressivity
#
#         self.mass = nn.Parameter(
#             torch.zeros(n_oscillators, n_resonances, expressivity) \
#                 .uniform_(-2, 2))
#
#         self.damping = nn.Parameter(
#             torch.zeros(n_oscillators, n_resonances, expressivity) \
#                 .uniform_(0.5, 1.5))
#
#         self.tension = nn.Parameter(
#             torch.zeros(n_oscillators, n_resonances, expressivity) \
#                 .uniform_(4, 9))
#
#         self.initial_displacement = nn.Parameter(
#             torch.zeros(n_oscillators, n_resonances, expressivity) \
#                 .uniform_(-1, 2))
#
#         self.amplitudes = nn.Parameter(
#             torch.zeros(n_oscillators, n_resonances, expressivity, 1) \
#                 .uniform_(-1, 1))
#
#     def _materialize_resonances(self, device: torch.device):
#         time = torch.linspace(0, 10, self.n_samples, device=device) \
#             .view(1, 1, 1, self.n_samples)
#
#         x = damped_harmonic_oscillator(
#             time=time,
#             mass=torch.sigmoid(self.mass[..., None]),
#             # mass=0.2,
#             damping=torch.sigmoid(self.damping[..., None]) * 20,
#             tension=10 ** self.tension[..., None],
#             initial_displacement=self.initial_displacement[..., None],
#             initial_velocity=0
#         )
#
#         x = x.view(self.n_oscillators, self.n_resonances, self.expressivity, self.n_samples)
#         x = x * self.amplitudes ** 2
#         x = torch.sum(x, dim=0)
#
#         ramp = torch.ones(self.n_samples, device=device)
#         ramp[:10] = torch.linspace(0, 1, 10, device=device)
#         return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples) * ramp[None, None, None, None, :]
#
#     def forward(self) -> torch.Tensor:
#         return self._materialize_resonances(self.damping.device)
#

LossFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

def overfit_model(
        n_samples: int,
        model: nn.Module,
        loss_func: LossFunc,
        collection_name: str):

    target = get_one_audio_segment(n_samples)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    collection = conjure.LmdbCollection(path=collection_name)

    t, r = conjure.loggers(
        ['target', 'recon',],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure(
        [t, r], port=9999, n_workers=1)

    t(max_norm(target))

    for i in count():
        optimizer.zero_grad()
        recon = model.forward()
        r(max_norm(recon))
        loss = loss_func(recon, target)
        loss.backward()
        optimizer.step()
        print(i, loss.item())




# def overfit_model():
#     n_samples = 2 ** 17
#     resonance_window_size = 2048
#     step_size = 1024
#     n_frames = n_samples // step_size
#
#     # KLUDGE: control_plane_dim and n_resonances
#     # must have the same value
#     control_plane_dim = 16
#     n_resonances = 16
#     expressivity = 2
#
#     target = get_one_audio_segment(n_samples)
#     model = OverfitResonanceStack(
#         n_layers=1,
#         n_samples=n_samples,
#         resonance_window_size=resonance_window_size,
#         control_plane_dim=control_plane_dim,
#         n_resonances=n_resonances,
#         expressivity=expressivity,
#         base_resonance=0.01,
#         n_frames=n_frames
#     ).to(device)
#     optimizer = Adam(model.parameters(), lr=1e-3)
#     collection = conjure.LmdbCollection(path='resonancemodel')
#
#
#     t, r, rand, res = conjure.loggers(
#         ['target', 'recon', 'random', 'resonance'],
#         'audio/wav',
#         encode_audio,
#         collection)
#
#     def to_numpy(x: torch.Tensor) -> np.ndarray:
#         return x.data.cpu().numpy()
#
#     c, deformations, routing, attack = conjure.loggers(
#         ['control', 'deformations', 'routing', 'attack'],
#         SupportedContentType.Spectrogram.value,
#         to_numpy,
#         collection,
#         serializer=NumpySerializer(),
#         deserializer=NumpyDeserializer())
#
#     serve_conjure(
#         [t, r, c, rand, res, deformations, routing, attack], port=9999, n_workers=1)
#
#     t(max_norm(target))
#
#     # loss_model = CorrelationLoss(n_elements=256).to(device)
#
#     def train():
#         iteration = 0
#
#         while True:
#             optimizer.zero_grad()
#             recon = model.forward()
#             r(max_norm(recon))
#             c(model.control_signal[0, 0])
#
#             # x = flattened_multiband_spectrogram(target, {'s': (64, 16)})
#             # y = flattened_multiband_spectrogram(recon, {'s': (64, 16)})
#             x = stft(target, 2048, 256, pad=True)
#             y = stft(recon, 2048, 256, pad=True)
#             loss = torch.abs(x - y).sum()
#
#             # loss = loss_model.multiband_noise_loss(target, recon, 64, 16)
#
#             loss.backward()
#             optimizer.step()
#             print(iteration, loss.item(), model.compression_ratio(n_samples))
#
#             deformations(model.flattened_deformations)
#             routing(torch.abs(model.get_router(0)))
#             attack(max_norm(model.get_attack_envelopes(0)))
#
#             with torch.no_grad():
#                 rand(max_norm(model.random(use_learned_deformations=False)))
#                 rz = model.get_materialized_resonance(0).view(-1, n_samples)
#                 res(max_norm(rz[np.random.randint(0, n_resonances * expressivity - 1)]))
#
#             iteration += 1
#
#
#     train()


def compute_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    t = stft(target, 2048, 256, pad=True)
    r = stft(recon, 2048, 256, pad=True)
    return torch.abs(t - r).sum()

if __name__ == '__main__':
    n_samples = 2 ** 17

    overfit_model(
        n_samples=n_samples,
        model=DampedHarmonicOscillatorController(
            n_samples=n_samples,
            control_rate=512,
            n_oscillators=64),
        loss_func=compute_loss,
        collection_name='dho'
    )
