from typing import Tuple, Union, Optional

import numpy as np
import torch
from torch import nn

from modules import max_norm, interpolate_last_axis
from modules.transfer import freq_domain_transfer_function_to_resonance, fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import playable
from util.playable import listen_to_sound


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

    routed = torch.relu(routed)
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
        start_mags=torch.sigmoid(amp),
        log_space_scan=True)
    res = res.view(1, 1, n_resonances, expressivity, n_samples)

    # "route" energy from control plane to resonances
    routed = routed.view(batch, n_events, n_resonances, 1, frames)
    routed = upsample_with_holes(routed, n_samples)

    # convolve control plane with resonances
    conv = fft_convolve(routed, res)

    # interpolate between variations on each resonance
    base_deformation = torch.zeros_like(deformations)
    base_deformation[:, :, 0:1, :] = 1
    d = base_deformation + deformations
    d = torch.softmax(d, dim=-2)
    d = d.view(batch_size, n_events, 1, expressivity, def_frames)
    d = interpolate_last_axis(d, n_samples)

    x = d * conv
    x = torch.sum(x, dim=-2)
    x = torch.tanh(x * torch.abs(gains.view(1, 1, n_resonances, 1)))

    summed = torch.sum(x, dim=-2, keepdim=True)
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

        self.router = nn.Parameter(
            torch.zeros((self.control_plane_dim, self.n_resonances)).uniform_(-1, 1))
        # self.deformations = nn.Parameter(
        #     torch.zeros((batch_size, n_events, expressivity, n_frames)).uniform_(-0.01, 0.01))

        def init_resonance() -> torch.Tensor:
            # base resonance
            res  = torch.zeros((n_resonances, resonance_coeffs, 1)).uniform_(-6, 6)
            # variations or deformations of the base resonance
            deformation = torch.zeros((1, resonance_coeffs, expressivity)).uniform_(-0.02, 0.02)
            # expand into (n_resonances, n_deformations)
            return res + deformation

        self.resonances = nn.ParameterDict(dict(
            amp=init_resonance(),
            phase=init_resonance(),
            decay=init_resonance(),
        ))

        self.gains = nn.Parameter(torch.zeros((n_resonances, 1)).uniform_(0.01, 10))

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


if __name__ == '__main__':
    n_samples = 2 ** 16
    resonance_window_size = 2048
    resonance_coeffs = resonance_window_size // 2 + 1

    control_plane_dim = 16
    n_resonances = 16
    expressivity = 1

    batch_size = 1
    n_events = 1

    # TODO: There is no need for control signal and deformations to have
    # the same sampling rate, in fact, deformations should likely be slower
    samples_per_frame = 128
    n_frames = n_samples // samples_per_frame
    device = torch.device('cuda')

    control_plane = torch.zeros(
        (batch_size, n_events, control_plane_dim, n_frames), device=device) \
        .bernoulli_(p=0.0005)

    deformations = torch.zeros(
        (batch_size, n_events, expressivity, n_frames), device=device).uniform_(-0.01, 0.01)

    network = ResonanceStack(
        n_layers=3,
        n_samples=n_samples,
        resonance_window_size=resonance_window_size,
        control_plane_dim=control_plane_dim,
        n_resonances=n_resonances,
        expressivity=expressivity,
        base_resonance=0.01
    ).to(device)

    final = network.forward(control_plane, deformations)
    print(final.shape)

    final = max_norm(final)
    p = playable(final, 22050, normalize=False, pad_with_silence=True)
    listen_to_sound(p, True)