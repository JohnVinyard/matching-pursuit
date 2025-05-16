from typing import Tuple

import numpy as np
import torch

from conjure import serve_conjure, SupportedContentType, NumpyDeserializer, NumpySerializer
from torch import nn
from torch.optim import Adam

import conjure
from data import get_one_audio_segment
from modules import max_norm, interpolate_last_axis, sparsify, unit_norm, flattened_multiband_spectrogram
from modules.transfer import freq_domain_transfer_function_to_resonance, fft_convolve
from util import device, encode_audio


def execute_layer(
        control_signal: torch.Tensor,
        routing: torch.Tensor,
        resonances: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        deformations: torch.Tensor,
        gains: torch.Tensor,
        n_samples: int,
        window_size: int,
        base_resonance: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    amp, phase, decay = resonances

    res_span = 1 - base_resonance
    res_factor = 0.9

    n_resonances, n_coeffs, expressivity = amp.shape
    batch, n_events, control_plane_dim, frames = control_signal.shape
    batch, n_events, expressivity, def_frames = deformations.shape
    cpd, nr = routing.shape

    # TODO: control signal sparsity enforced here?

    # first determine routing
    # TODO: einops
    routed = (control_signal.permute(0, 1, 3, 2) @ routing).permute(0, 1, 3, 2)

    before_upsample = routed

    step_size = window_size // 2
    n_resonance_frames = n_samples // step_size

    amp = amp.permute(0, 2, 1)
    phase = phase.permute(0, 2, 1)
    decay = decay.permute(0, 2, 1)
    # materialize resonances
    res = freq_domain_transfer_function_to_resonance(
        window_size,
        base_resonance + ((torch.sigmoid(decay) * res_span) * res_factor),
        n_resonance_frames,
        apply_decay=True,
        start_phase=torch.tanh(phase) * np.pi,
        start_mags=torch.abs(amp),
        log_space_scan=True)
    res = res.view(1, 1, n_resonances, expressivity, n_samples)
    res = unit_norm(res)

    # "route" energy from control plane to resonances
    routed = routed.view(batch, n_events, n_resonances, 1, frames)
    # routed = upsample_with_holes(routed, n_samples)
    routed = interpolate_last_axis(routed, n_samples)
    routed = routed * torch.zeros_like(routed).uniform_(-1, 1)

    # convolve control plane with resonances
    conv = fft_convolve(routed, res)

    # interpolate between variations on each resonance
    base_deformation = torch.zeros_like(deformations)
    base_deformation[:, :, 0:1, :] = 1
    d = base_deformation + deformations
    d = torch.softmax(d, dim=-2)
    d = d.view(batch, n_events, 1, expressivity, def_frames)
    d = interpolate_last_axis(d, n_samples)

    x = d * conv
    x = torch.sum(x, dim=-2)

    summed = torch.tanh(x * torch.abs(gains.view(1, 1, n_resonances, 1)))

    summed = torch.sum(summed, dim=-2, keepdim=True)

    return summed, before_upsample


class ResonanceLayer(nn.Module):

    def __init__(
            self,
            n_samples: int,
            resonance_window_size: int,
            control_plane_dim: int,
            n_resonances: int,
            expressivity: int,
            base_resonance: float = 0.5):
        super().__init__()
        self.expressivity = expressivity
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.resonance_window_size = resonance_window_size
        self.n_samples = n_samples
        self.base_resonance = base_resonance

        resonance_coeffs = resonance_window_size // 2 + 1

        self.router = nn.Parameter(
            torch.zeros((self.control_plane_dim, self.n_resonances)).uniform_(-1, 1))

        def init_resonance() -> torch.Tensor:
            # base resonance
            res = torch.zeros((n_resonances, resonance_coeffs, 1)).uniform_(0.01, 1)
            # variations or deformations of the base resonance
            deformation = torch.zeros((1, resonance_coeffs, expressivity)).uniform_(-0.02, 0.02)
            # expand into (n_resonances, n_deformations)
            return res + deformation

        self.resonances = nn.ParameterDict(dict(
            amp=init_resonance(),
            phase=init_resonance(),
            decay=init_resonance(),
        ))

        self.gains = nn.Parameter(torch.zeros((n_resonances, 1)).uniform_(0.01, 1.1))

    def forward(self, control_signal: torch.Tensor, deformations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, fwd = execute_layer(
            control_signal,
            self.router,
            tuple(self.resonances.values()),
            deformations,
            self.gains,
            self.n_samples,
            self.resonance_window_size,
            self.base_resonance
        )
        return output, fwd


class ResonanceStack(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_samples: int,
            resonance_window_size: int,
            control_plane_dim: int,
            n_resonances: int,
            expressivity: int,
            base_resonance: float = 0.5):
        super().__init__()

        self.expressivity = expressivity
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.resonance_window_size = resonance_window_size
        self.n_samples = n_samples
        self.base_resonance = base_resonance

        self.mix = nn.Parameter(torch.zeros(n_layers))

        self.layers = nn.ModuleList([ResonanceLayer(
            n_samples,
            resonance_window_size,
            control_plane_dim,
            n_resonances,
            expressivity,
            base_resonance
        ) for _ in range(n_layers)])

    def forward(self, control_signal: torch.Tensor, deformations: torch.Tensor) -> torch.Tensor:
        batch_size, n_events, cpd, frames = control_signal.shape

        outputs = []
        cs = control_signal

        for layer in self.layers:
            output, cs = layer(cs, deformations)
            outputs.append(output)

        final = torch.stack(outputs, dim=-1)
        mx = torch.softmax(self.mix, dim=-1)

        final = final @ mx[:, None]
        return final.view(batch_size, n_events, self.n_samples)


class OverfitResonanceStack(nn.Module):

    def __init__(
            self,
            n_layers: int,
            n_samples: int,
            resonance_window_size: int,
            control_plane_dim: int,
            n_resonances: int,
            expressivity: int,
            n_frames: int,
            base_resonance: float = 0.5):
        super().__init__()
        self.expressivity = expressivity
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.resonance_window_size = resonance_window_size
        self.n_samples = n_samples
        self.base_resonance = base_resonance
        self.n_frames = n_frames

        control_plane = torch.zeros(
            (1, 1, control_plane_dim, n_frames)) \
            .uniform_(-0.01, 0.01)

        self.control_plane = nn.Parameter(control_plane)

        deformations = torch.zeros(
            (1, 1, expressivity, n_frames)).uniform_(-0.01, 0.01)
        self.deformations = nn.Parameter(deformations)

        self.network = ResonanceStack(
            n_layers=n_layers,
            n_samples=n_samples,
            resonance_window_size=resonance_window_size,
            control_plane_dim=control_plane_dim,
            n_resonances=n_resonances,
            expressivity=expressivity,
            base_resonance=base_resonance
        )

    def _process_control_plane(
            self,
            cp: torch.Tensor,
            n_to_keep: int = 256) -> torch.Tensor:

        cp = cp.view(1, self.control_plane_dim, self.n_frames)
        cp = sparsify(cp, n_to_keep=n_to_keep)
        cp = cp.view(1, 1, self.control_plane_dim, self.n_frames)
        return cp

    @property
    def control_signal(self):
        cp = self.control_plane
        cp = self._process_control_plane(cp)
        return cp

    def random(self):
        rcp = torch \
            .zeros_like(self.control_plane) \
            .uniform_(
            self.control_plane.min().item(),
            self.control_plane.max().item())

        rcp = self._process_control_plane(rcp, n_to_keep=32)
        x = self.forward(rcp, torch.zeros_like(self.deformations))
        return x

    def forward(
            self,
            cp: torch.Tensor = None,
            deformations: torch.Tensor = None):
        cp = cp if cp is not None else self.control_signal  # / self.control_plane.sum()
        deformations = deformations if deformations is not None else self.deformations
        x = self.network.forward(cp, deformations)
        return x


def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()
    forward = mask
    backward = x
    y = backward + (forward - backward).detach()
    return y.sum()


def overfit_model():
    n_samples = 2 ** 16
    resonance_window_size = 2048
    step_size = 256
    n_frames = n_samples // step_size

    control_plane_dim = 64
    n_resonances = 64
    expressivity = 4

    target = get_one_audio_segment(n_samples)
    model = OverfitResonanceStack(
        n_layers=2,
        n_samples=n_samples,
        resonance_window_size=resonance_window_size,
        control_plane_dim=control_plane_dim,
        n_resonances=n_resonances,
        expressivity=expressivity,
        base_resonance=0.01,
        n_frames=n_frames
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    collection = conjure.LmdbCollection(path='resonancemodel')

    t, r, rand = conjure.loggers(
        ['target', 'recon', 'random'],
        'audio/wav',
        encode_audio,
        collection)

    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.data.cpu().numpy()

    c, = conjure.loggers(
        ['control'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    serve_conjure([t, r, c, rand], port=9999, n_workers=1)

    t(max_norm(target))

    def train():
        iteration = 0

        while True:
            optimizer.zero_grad()
            recon = model.forward()
            r(max_norm(recon))
            c(model.control_signal[0, 0])

            x = flattened_multiband_spectrogram(target, {'s': (64, 16)})
            y = flattened_multiband_spectrogram(recon, {'s': (64, 16)})
            loss = torch.abs(x - y).sum()

            loss.backward()
            optimizer.step()
            print(iteration, loss.item())

            with torch.no_grad():
                rand(max_norm(model.random()))

            iteration += 1

    train()


if __name__ == '__main__':
    overfit_model()
