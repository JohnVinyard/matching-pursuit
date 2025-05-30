from itertools import count
from typing import Union, List, Iterable, Optional, Tuple

import numpy as np
import torch
from scipy.signal import morlet
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, loggers, serve_conjure, SupportedContentType, NumpySerializer, NumpyDeserializer
from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import flattened_multiband_spectrogram, max_norm, stft, sparsify, \
    gammatone_filter_bank
from modules.infoloss import CorrelationLoss
from modules.overlap_add import overlap_add
from modules.transfer import fft_convolve, freq_domain_transfer_function_to_resonance
from modules.upsample import upsample_with_holes
from util import device, make_initializer, count_parameters, playable

import matplotlib

from util.playable import listen_to_sound

matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

n_samples = 2 ** 17
transform_window_size = 2048
transform_step_size = 1024
samplerate = 22050
n_frames = n_samples // transform_step_size

initializer = make_initializer(0.05)


def stft_transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, -1, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


def morlet_filter_bank(
        samplerate: int,
        kernel_size: int,
        scale: Union[List[int], np.ndarray],
        scaling_factor: Union[List, float, np.ndarray],
        normalize=True,
        device=None):
    basis_size = len(scale)
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    try:
        if len(scaling_factor) != len(scale):
            raise ValueError('scaling factor must have same length as scale')
    except TypeError:
        scaling_factor = np.repeat(float(scaling_factor), len(scale))

    sr = int(samplerate)

    for i, center_frequency in enumerate(scale):
        scaling = scaling_factor[i]
        w = center_frequency / (scaling * 2 * sr / kernel_size)
        basis[i] = morlet(
            M=kernel_size,
            w=w,
            s=scaling)

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8

    return torch.from_numpy(basis.real).float().to(device)

def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    # this is here to make the shift value interpretable
    shift = (1 - shift)

    n_samples = a.shape[-1]

    shift_samples = (shift * n_samples)

    # a = F.pad(a, (0, n_samples * 2))

    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs

    shift = torch.exp(shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    # samples = torch.relu(samples)
    return samples


def run_layer(
        control_plane: torch.Tensor,
        mapping: torch.Tensor,
        decays: torch.Tensor,
        out_mapping: torch.Tensor,
        audio_mapping: torch.Tensor,
        gains: torch.Tensor,
        shift: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

    batch_size, control_plane_dim, frames = control_plane.shape

    x = mapping @ control_plane
    orig = x
    decays = decays.view(batch_size, control_plane_dim, 1).repeat(1, 1, frames)

    decays = torch.log(1e-12 + decays)
    decays = torch.cumsum(decays, dim=-1)
    decays = torch.exp(decays)

    # decays = decays.cumprod(dim=-1)
    # decays = torch.flip(decays, dims=[-1])
    x = fft_convolve(x, decays)
    x = (out_mapping @ x) + orig

    # shift = torch.tanh(shift @ x)

    cp = torch.tanh(x * gains.view(batch_size, control_plane_dim, 1))

    audio = audio_mapping @ cp
    # shift = shift.repeat(1, audio.shape[1], 1)
    # print(audio.shape, shift.shape)
    # audio = fft_shift(audio.permute(0, 2, 1), shift.permute(0, 2, 1))
    # audio = audio.permute(0, 2, 1)

    # TODO: This should be mapped to audio outside of this layer, probably
    # each layer by a single mapping network
    audio = audio.permute(0, 2, 1)

    audio = audio.reshape(batch_size, 1, -1)

    return audio, cp


class Block(nn.Module):
    def __init__(
            self,
            block_size,
            base_resonance: float = 0.5,
            max_gain: float = 5,
            window_size: Union[int, None] = None):

        super().__init__()
        self.block_size = block_size
        self.base_resonance = base_resonance
        self.resonance_span = 1 - base_resonance
        self.max_gain = max_gain
        self.window_size = window_size or block_size

        self.w1 = nn.Parameter(torch.zeros(block_size, block_size).uniform_(-0.01, 0.01))
        self.w2 = nn.Parameter(torch.zeros(block_size, block_size).uniform_(-0.01, 0.01))
        self.audio = nn.Parameter(torch.zeros(window_size, block_size).uniform_(-0.01, 0.01))

        self.to_shift = nn.Parameter(torch.zeros(1, block_size).uniform_(-1, 1))

        self.decays = nn.Parameter(torch.zeros(block_size).uniform_(0.001, 0.99))
        self.gains = nn.Parameter(torch.zeros(block_size).uniform_(-3, 3))

    def forward(self, cp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, cp = run_layer(
            torch.relu(cp),
            self.w1,
            self.base_resonance + torch.sigmoid(self.decays) * self.resonance_span,
            self.w2,
            self.audio,
            torch.sigmoid(self.gains) * self.max_gain,
            self.to_shift)
        return output, cp


class Stack(nn.Module):
    def __init__(
            self,
            n_blocks,
            block_size,
            base_resonance:
            float = 0.5,
            max_gain: float = 5,
            window_size: Union[int, None] = None):
        super().__init__()
        self.block_size = block_size
        self.n_blocks = n_blocks
        self.window_size = window_size

        self.mix = nn.Parameter(torch.zeros(n_blocks).uniform_(-1, 1))
        self.blocks = nn.ModuleList([
            Block(
                block_size,
                base_resonance,
                max_gain,
                window_size = window_size
            ) for _ in range(n_blocks)
        ])

    def forward(self, cp):
        batch_size, channels, frames = cp.shape

        working_control_plane = cp

        total_samples = frames * self.window_size

        channels = torch.zeros(
            batch_size, self.n_blocks, total_samples, device=cp.device)

        for i, block in enumerate(self.blocks):
            output, working_control_plane = block(working_control_plane)
            channels[:, i: i + 1, :] = output

        mix = torch.softmax(self.mix, dim=-1)
        mixed = channels.permute(0, 2, 1) @ mix
        mixed = mixed.view(batch_size, 1, total_samples)
        return max_norm(mixed)


# class Block(nn.Module):
#
#     def __init__(
#             self,
#             window_size: int,
#             control_plane_dim: int,
#             transfer: torch.Tensor,
#             gain: Union[float, torch.Tensor],
#             deformation_latent_shape: int = None,
#             filter_bank: torch.Tensor = None,
#             preserve_energy: bool = False):
#
#         super().__init__()
#         self.deformation_latent_shape = deformation_latent_shape
#         self.window_size = window_size
#         self.n_coeffs = self.window_size // 2 + 1
#         self.control_plane_dim = control_plane_dim
#         self.preserve_energy = preserve_energy
#
#         self.register_buffer('group_delay', torch.linspace(0, np.pi, self.n_coeffs))
#
#         if filter_bank is not None:
#             # TODO: validation code.  Filter bank should be (n_coeffs, window_size)
#             self.register_buffer('filter_bank', torch.abs(torch.fft.rfft(filter_bank, dim=-1, norm='ortho')))
#         else:
#             self.filter_bank = None
#
#         if self.deformation_latent_shape is not None:
#             self.hyper = HyperNetworkLayer(
#                 deformation_latent_shape,
#                 deformation_latent_shape,
#                 self.n_coeffs,
#                 self.n_coeffs,
#                 bias=False,
#                 force_identity=False)
#         else:
#             self.hyper = None
#
#         self.transfer = nn.Parameter(torch.zeros((self.control_plane_dim, self.n_coeffs)))
#         self.mixer_matrix = nn.Parameter(torch.eye(self.control_plane_dim))
#
#         self.transfer.data[:] = transfer
#
#         # TODO: should there be a separate gain for each channel of control plane dim?
#         self.gain = nn.Parameter(torch.ones((control_plane_dim,)).fill_(gain))
#
#     def forward(self, x: torch.Tensor, deformations: torch.Tensor = None) -> torch.Tensor:
#
#         # route energy from the input control dim to the available
#         # transfer functions
#         x = (x.permute(0, 2, 1) @ self.mixer_matrix).permute(0, 2, 1)
#
#         batch, channels, time = x.shape
#
#         assert channels == self.control_plane_dim
#
#         windowed = windowed_audio(x, self.window_size, self.window_size // 2)
#
#         spec = torch.fft.rfft(windowed, dim=-1)
#
#         batch, channels, frames, coeffs = spec.shape
#
#         output_frames = []
#
#         identity = torch.eye(self.n_coeffs, device=x.device)
#
#         for i in range(frames):
#             current = spec[:, :, i: i + 1, :]
#             orig_norm = torch.norm(current, dim=-1, keepdim=True)
#
#             if i > 0:
#                 current = current + output_frames[i - 1]
#
#             current_transfer = self.transfer[None, :, None, :]
#
#             if deformations is not None and self.hyper is not None:
#                 # TODO: This should use the norm-perserving non-linearity
#                 current_deformation = deformations[:, :, i, :]
#                 w, func = self.hyper.forward(
#                     current_deformation.view(-1, self.deformation_latent_shape),
#                     identity[None, ...]
#                 )
#                 cdm = func(current_transfer.view(-1, self.n_coeffs))
#                 cdm = cdm.view(batch, self.control_plane_dim, 1, self.n_coeffs)
#                 current_transfer = cdm
#
#             if self.filter_bank is not None:
#                 # TODO: This should use the norm-preserving non-linearity
#                 filtered = current_transfer @ self.filter_bank
#                 filtered = filtered * current
#             else:
#                 # perform convolution in the frequency domain
#                 filtered = current * current_transfer
#
#             if self.preserve_energy:
#                 filtered = limit_norm(filtered, dim=-1, max_norm=orig_norm * 0.9999)
#
#             # TODO: as a *hack*, I could simply preserve norm here
#             # given the input norm, although this feels like a hack;
#             # ideally the operations representing the deformation and
#             # frequency mapping are norm preserving
#             output_frames.append(filtered)
#
#         output = torch.cat(output_frames, dim=2)
#         audio_windows = torch.fft.irfft(output, dim=-1)
#         samples = overlap_add(audio_windows, apply_window=True)
#         samples = samples * self.gain[None, :, None]
#         samples = torch.tanh(samples)
#         x = samples[..., :time]
#         return x


# class AudioNetwork(nn.Module):
#     def __init__(
#             self,
#             control_plane_dim: int,
#             window_size: int,
#             n_blocks: int,
#             deformation_latent_dim: int = None,
#             filter_bank: torch.Tensor = None,
#             preserve_energy: bool = False):
#         super().__init__()
#         self.window_size = window_size
#         self.n_blocks = n_blocks
#         self.mixer = nn.Parameter(torch.zeros((n_blocks + 1,)))
#         self.control_plane_dim = control_plane_dim
#         self.n_coeffs = window_size // 2 + 1
#
#         self.blocks = nn.ModuleList([
#             Block(
#                 window_size,
#                 control_plane_dim,
#                 self.init_transfer(),
#                 torch.zeros(1).uniform_(1, 2).item(),
#                 deformation_latent_shape=deformation_latent_dim,
#                 filter_bank=filter_bank,
#                 preserve_energy=preserve_energy)
#             for _ in range(self.n_blocks)
#         ])
#
#     def init_transfer(self):
#         resonances = torch.zeros(self.control_plane_dim, self.n_coeffs).uniform_(0.5, 0.9998)
#         sparse = torch.zeros_like(resonances).bernoulli_(p=0.01)
#         resonances = resonances * sparse
#         scaled_resonances = resonances
#         return scaled_resonances
#
#     def forward(self, x: torch.Tensor, deformations: Iterable[torch.Tensor]) -> torch.Tensor:
#         outputs = [x[..., None]]
#         inp = x
#
#         for i, block in enumerate(self.blocks):
#             try:
#                 deform = deformations[i]
#             except (TypeError, IndexError):
#                 deform = None
#             inp = block(inp, deform)
#             outputs.append(inp[..., None])
#
#         result = torch.cat(outputs, dim=-1)
#         mixer_values = torch.softmax(self.mixer, dim=-1)
#         mixed = (result * mixer_values[None, None, None, :]).sum(dim=-1)
#         mixed = torch.sum(mixed, dim=1, keepdim=True)
#         return mixed


class ResonanceLayer(nn.Module):
    def __init__(
            self,
            n_frames: int,
            n_samples: int,
            control_plane_dim: int,
            n_resonances: int,
            n_deformations: int,
            window_size: int,
            max_gain: float = 10,
            base_resonance: float = 0.01,
    ):

        super().__init__()
        self.n_deformations = n_deformations
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.window_size = window_size
        self.n_coeffs = self.window_size // 2 + 1
        self.resonance_frames = self.n_samples // (self.window_size // 2)
        self.max_gain = max_gain
        self.base_resonance = base_resonance
        self.resonance_range = 1 - self.base_resonance

        self.spectral_shapes = nn.Parameter(
            torch.zeros(self.n_resonances, self.n_coeffs).uniform_(-3, 3) \
            * torch.zeros(self.n_resonances, self.n_coeffs).bernoulli_(p=0.001)
        )

        self.route_energy = nn.Parameter(torch.zeros(self.control_plane_dim, self.n_resonances).bernoulli_(p=0.1))
        self.route_to_control_plane = nn.Parameter(torch.zeros(self.n_resonances, self.control_plane_dim).bernoulli_(p=0.1))

        self.gains = nn.Parameter(torch.zeros(1, self.n_resonances, 1).uniform_(-3, 3))


    def forward(self, control_plane: torch.Tensor, mix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        if control_plane.shape[-1] != self.n_samples:
            cp = upsample_with_holes(control_plane, self.n_samples)
        else:
            cp = control_plane

        n = torch.zeros(1, 1, 128, device=control_plane.device).uniform_(-1, 1)
        n = n * torch.hamming_window(128, device=n.device)

        n = torch.cat([n, torch.zeros(1, 1, self.n_samples - 128, device=n.device)], dim=-1)
        energy = fft_convolve(cp, n)

        # print(self.spectral_shapes.shape, self.gammatone.shape)

        resonances = freq_domain_transfer_function_to_resonance(
            window_size=self.window_size,
            coeffs=self.base_resonance + (torch.sigmoid(self.spectral_shapes) * self.resonance_range * 0.9),
            n_frames=self.resonance_frames,
            apply_decay=True,
        ).view(1, self.n_resonances, self.n_samples)

        energy = self.route_energy.T @ energy

        energy_with_resonance = fft_convolve(energy, resonances)

        # print(energy_with_resonance.shape, self.gains.shape)

        energy_with_resonance = torch.tanh(energy_with_resonance * torch.sigmoid(self.gains) * self.max_gain)

        mix = torch.softmax(mix, dim=-1)

        stacked = torch.stack([energy, energy_with_resonance], dim=-1)

        x = torch.sum(stacked * mix[:, None, None, :], dim=-1)

        routed = self.route_to_control_plane.T @ x

        output = torch.sum(x, dim=1, keepdim=True)

        return routed, output

class EnergyBasedEventGenerator(nn.Module):

    def __init__(self, instrument_dim: int):
        super().__init__()
        self.instrument_dim = instrument_dim

        self.block_size = 512
        self.n_coeffs = self.block_size // 2 + 1
        self.total_complex_coeffs = self.n_coeffs * 2

        self.base_damping = 0.5
        self.span = 1 - self.base_damping

        self.n_frames = 512

        transform_block_size = self.total_complex_coeffs

        n_filters = self.n_coeffs
        filters = gammatone_filter_bank(n_filters, self.block_size, device, band_spacing='geometric')
        filters = torch.fft.rfft(filters, dim=-1, norm='ortho')
        self.register_buffer('filters', filters)

        self.to_samples = nn.Parameter(torch.zeros(transform_block_size, self.instrument_dim).uniform_(-1, 1))



    def forward(
            self,
            masses: torch.Tensor,
            tensions: torch.Tensor,
            damping: torch.Tensor,
            forces: torch.Tensor) -> torch.Tensor:

        masses = 1 + (torch.abs(masses) * 100)
        tensions = 0.01 + (torch.abs(tensions) * 10)
        damping = self.base_damping + (torch.sigmoid(damping) * self.span * 0.9999)

        # forces = sparsify(forces, 32)

        batch, n_events, dim = masses.shape
        resting = torch.zeros_like(masses)

        velocity = torch.zeros_like(resting)
        acceleration = torch.zeros_like(resting)

        state = torch.zeros_like(resting)


        displacement = torch.zeros_like(state)
        blocks = []

        for i in range(self.n_frames):
            # calculate new displacement
            displacement = (state - resting)

            acceleration = acceleration + ((-displacement * tensions) / masses)

            # accumulate forces
            # apply any forces from outside the system
            acceleration = acceleration + (forces[:, :, :, i] / masses)

            # apply forces
            velocity = velocity + acceleration
            state = state + velocity

            # clear
            # apply damping (this could vary over time)
            velocity = velocity * damping
            # velocity[torch.abs(velocity) < 1e-6] = 0

            force = masses * acceleration

            # clear forces
            acceleration = acceleration * 0

            block = (force @ self.to_samples.T)  # * torch.norm(displacement)

            block = block.view(-1, n_events, self.n_coeffs, 2)
            block = torch.view_as_complex(block)

            block = block @ self.filters

            block = torch.fft.irfft(block, dim=-1)

            # print(torch.norm(displacement), torch.norm(block))
            blocks.append(block[:, :, None, :])


        # plt.matshow(np.abs(torch.stack(disp, dim=0).data.cpu().numpy().squeeze()))
        # plt.show()

        blocks = torch.cat(blocks, dim=-2)


        # print(blocks.shape)
        # samples = blocks.view(batch_size, n_events, -1)
        samples = overlap_add(blocks, apply_window=True)[..., :n_samples]


        return samples

class OverfitAudioNetwork(nn.Module):
    def __init__(
            self,
            window_size: int = 512,
            control_plane_dim: int = 16,
            low_rank_deformation_dim: int = 16,
            n_samples: int = 2 ** 15,
            n_frames: int = 128,
            n_layers: int = 3,
            impulse_decay_samples: int = 128,
            samplerate: int = 22050,
            deformations_enabled: bool = False,
            preserve_energy: bool = False,
            block_size: int = 512,
    ):

        super().__init__()

        self.block_size = block_size
        self.block_frames = n_samples // window_size

        self.n_samples = n_samples
        self.n_layers = n_layers
        self.deformations_enabled = deformations_enabled
        self.n_frames = n_frames

        self.samples_per_frame = self.n_samples // self.n_frames

        # transfer_dim = window_size // 2 + 1

        # self.control_plane = nn.Parameter(torch.zeros(1, control_plane_dim, block_frames).uniform_(0, 1e-8))
        # self.channel_decays = nn.Parameter(torch.zeros((control_plane_dim,)).uniform_(10, 50))

        # msh = musical_scale_hz(start_midi=21, stop_midi=129, n_steps=transfer_dim)

        # establish (optional) non-linear frequency space
        # fb = morlet_filter_bank(
        #     samplerate,
        #     kernel_size=window_size,
        #     scale=msh,
        #     scaling_factor=0.025,
        #     normalize=True,
        #     device=None)



        # establish deformation matrix (batch, control_plane_dim, n_frames, low_rank_deformation_dim)
        # dm = [
        #     torch.zeros(
        #         1,
        #         control_plane_dim,
        #         n_frames,
        #         low_rank_deformation_dim).uniform_(-0.1, 0.1)
        #     for _ in range(n_layers)
        # ]
        # self.deformations = nn.ParameterList(dm)

        # self.network = AudioNetwork(
        #     control_plane_dim,
        #     window_size,
        #     n_blocks=n_layers,
        #     deformation_latent_dim=low_rank_deformation_dim if self.deformations_enabled else None,
        #     filter_bank=fb,
        #     preserve_energy=preserve_energy
        # )

        # self.network = Stack(
        #     n_blocks=n_layers,
        #     block_size=self.block_size,
        #     base_resonance=0.5,
        #     max_gain=5,
        #     window_size=window_size)


        self.network = EnergyBasedEventGenerator(instrument_dim=control_plane_dim)

        control_plane = torch.zeros(1, 1, control_plane_dim, self.n_frames).uniform_(-0.01, 0.01)
        self.control_plane = nn.Parameter(control_plane)
        self.control_plane_dim = control_plane_dim

        self.masses = nn.Parameter(torch.zeros(1, 1, control_plane_dim).uniform_(0, 1))
        self.tensions = nn.Parameter(torch.zeros(1, 1, control_plane_dim).uniform_(0, 1))
        self.damping = nn.Parameter(torch.zeros(1, 1, control_plane_dim).uniform_(-6, 6))


        self.param_count = count_parameters(self.network)

    @property
    def control_signal(self):
        return torch.relu(sparsify(self.control_plane + torch.zeros_like(self.control_plane).uniform_(-1e-4, 1e-4), n_to_keep=128))

    @property
    def nonzero_count(self):
        return (self.control_signal > 0).sum().item()

    @property
    def sparsity(self):
        return self.nonzero_count / self.control_plane.numel()

    # @property
    # def all_deformations(self):
    #     x = torch.stack([d for d in self.deformations], dim=0)
    #     return x

    def _base_envelopes(self):
        ls = torch.linspace(1, 0, self.samples_per_frame, device=device).view(1, 1, self.samples_per_frame)
        ls = ls ** self.channel_decays[None, :, None]
        return ls

    def _upsampled_control_plane(self, cp: torch.Tensor):
        # noise = torch.zeros((self.n_samples,), device=cp.device).uniform_(-1, 1)
        us = upsample_with_holes(cp, self.n_samples)
        ls = self._base_envelopes()
        ls = torch.cat(
            [ls, torch.zeros(1, self.control_plane_dim, self.n_samples - self.samples_per_frame, device=cp.device)],
            dim=-1)
        us = fft_convolve(us, ls)
        us = us * torch.zeros_like(us).uniform_(-1, 1)
        return us

    def random(self):
        cp = torch.zeros_like(self.control_plane).bernoulli_(p=0.001)
        cp = cp * torch.zeros_like(cp).uniform_(0, 0.01)
        result = self.forward(sig=cp)
        return result

    def forward(self, sig=None):
        cs = sig if sig is not None else self.control_signal
        result = self.network.forward(
            masses=self.masses,
            tensions=self.tensions,
            damping=self.damping,
            forces=cs)
        return result, cs



def transform(x: torch.Tensor):
    """
    Decompose audio into sub-bands of varying sample rate, and compute spectrogram with
    varying time-frequency tradeoffs on each band.
    """
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the difference between the `recon` and `target`
    representations
    """
    fake_spec = transform(recon)
    real_spec = transform(target)
    return torch.abs(fake_spec - real_spec).sum()

def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()

    forward = mask
    backward = x

    y = backward + (forward - backward).detach()

    return y.sum()


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the control signal
    """
    # return torch.abs(c).sum() * 100
    # hinge-loss
    return torch.clamp(l0_norm(c) - 16, 0, np.inf) * 1000


def construct_experiment_model(n_samples: int) -> OverfitAudioNetwork:
    low_rank_deformation_dim = 16
    n_frames = n_samples // 256
    n_blocks = 3
    samplerate = 22050

    # ==========
    window_size = 128
    block_size = 128
    control_plane_dim = block_size


    model = OverfitAudioNetwork(
        window_size=window_size,
        control_plane_dim=control_plane_dim,
        low_rank_deformation_dim=low_rank_deformation_dim,
        n_samples=n_samples,
        n_frames=n_frames,
        n_layers=n_blocks,
        impulse_decay_samples=128,
        deformations_enabled=False,
        samplerate=samplerate,
        preserve_energy=False,
        block_size=block_size,

    ).to(device)
    return model


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()




def train_and_monitor_overfit_model(
        n_samples: int,
        samplerate: int = 22050,
        audio_path: Optional[str] = None):
    target = get_one_audio_segment(
        n_samples=n_samples, samplerate=samplerate, audio_path=audio_path)
    collection = LmdbCollection(path='freqdomain')

    print(f'overfitting to {n_samples // samplerate} seconds')

    # TODO: Just use Logger here instead
    recon_audio, orig_audio, rnd = loggers(
        ['recon', 'orig', 'rnd'],
        'audio/wav',
        encode_audio,
        collection)

    envelopes, = loggers(
        ['envelopes'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
        envelopes,
        rnd,
    ], port=9999, n_workers=1)

    loss_model = CorrelationLoss(n_elements=2048).to(device)
    # loss_model = SpikingModel(64, 64, 64, 64).to(device)

    def train(target: torch.Tensor):
        model = construct_experiment_model(n_samples=n_samples)

        optim = Adam(model.parameters(), lr=1e-3)

        for iteration in count():
            optim.zero_grad()
            recon, control_signal = model.forward()

            # windowed = recon.unfold(-1, 512, 512)
            # last = windowed[:, :, -1:]
            # first = windowed[:, :, :1]
            # window_loss = torch.abs(last - first).sum()

            recon_audio(max_norm(recon.sum(dim=1, keepdim=True)))

            # loss = loss_model.compute_multiband_loss(target, recon)
            # loss = loss_model.multiband_noise_loss(target, recon, 64, 16) #+ window_loss
            loss = reconstruction_loss(recon, target)
            # recon_loss = recon_loss + loss_model.noise_loss(target, recon)
            # loss = loss + sparsity_loss(control_signal)

            # if model.deformations_enabled:
            #     loss = loss + sparsity_loss(model.all_deformations)

            non_zero = (control_signal > 0).sum()
            sparsity = (non_zero / control_signal.numel()).item()

            loss.backward()

            envelopes(max_norm(control_signal[0]))

            encoding_samples = model.param_count + (non_zero.item() * 3)
            compression_ratio = encoding_samples / n_samples

            optim.step()
            print(iteration, loss.item(), f'{sparsity:.2f}', f'Active elements: {non_zero.item()}', compression_ratio)

            with torch.no_grad():
                # log random output from the model
                r, _ = model.random()
                rnd(max_norm(r))

    train(target)


if __name__ == '__main__':
    train_and_monitor_overfit_model(
        n_samples=2 ** 17,
        samplerate=22050,
    )
    # train_and_monitor_auto_encoder(batch_size=2)

    # control_plane_dim = 16
    # n_resonances = 16
    # n_deformations = 4
    #
    # layer = ResonanceLayer(
    #     n_frames=128,
    #     n_samples=2**16,
    #     control_plane_dim=control_plane_dim,
    #     n_resonances=n_resonances,
    #     n_deformations=n_deformations,
    #     window_size=2048,
    #     base_resonance=0.1,
    #     max_gain=2)
    #
    #
    # control_plane = torch.zeros(3, control_plane_dim, n_frames).bernoulli_(p=0.01)
    # plt.matshow(control_plane.data.cpu().numpy()[0])
    # plt.show()
    #
    # mix = torch.zeros(3, 2).uniform_(0, 1)
    #
    # x, output = layer.forward(control_plane, mix)
    #
    # print(output.shape)
    # spec = stft(output, 2048, 256, pad=True)
    # print(spec.shape)
    #
    # # print(spech.)
    # plt.matshow(spec.data.cpu().numpy()[0].reshape((-1, 1025)))
    # plt.show()
    #
    # x = playable(output, 22050, normalize=True)
    # listen_to_sound(x)
    #
    #
    # # print(x.shape, output.shape)
    #
