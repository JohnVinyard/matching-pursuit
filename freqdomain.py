from typing import Union, List, Iterable

import torch

from modules import HyperNetworkLayer, limit_norm
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio
from modules.transfer import fft_convolve
from modules.upsample import upsample_with_holes
from util.music import musical_scale_hz
from torch import nn
import numpy as np
from scipy.signal import morlet


# TODO: The option to pass either a zounds.Scale or a list of center frequencies
# should be factored in to one of the pre-existing implementations
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


class Block(nn.Module):

    def __init__(
            self,
            window_size: int,
            control_plane_dim: int,
            transfer: torch.Tensor,
            gain: Union[float, torch.Tensor],
            deformation_latent_shape: int = None,
            filter_bank: torch.Tensor = None,
            preserve_energy: bool = False):

        super().__init__()
        self.deformation_latent_shape = deformation_latent_shape
        self.window_size = window_size
        self.n_coeffs = self.window_size // 2 + 1
        self.control_plane_dim = control_plane_dim
        self.preserve_energy = preserve_energy

        if filter_bank is not None:
            # TODO: validation code.  Filter bank should be (n_coeffs, window_size)
            self.register_buffer('filter_bank', torch.abs(torch.fft.rfft(filter_bank, dim=-1, norm='ortho')))
        else:
            self.filter_bank = None

        if self.deformation_latent_shape is not None:
            self.hyper = HyperNetworkLayer(
                deformation_latent_shape,
                deformation_latent_shape,
                self.n_coeffs,
                self.n_coeffs,
                bias=False,
                force_identity=False)
        else:
            self.hyper = None

        self.transfer = nn.Parameter(torch.zeros((self.control_plane_dim, self.n_coeffs,)))
        self.mixer_matrix = nn.Parameter(torch.eye(self.control_plane_dim))

        self.transfer.data[:] = transfer

        self.gain = nn.Parameter(torch.ones((1,)).fill_(gain))

    def forward(self, x: torch.Tensor, deformations: torch.Tensor = None) -> torch.Tensor:

        # route energy from the input control dim to the available
        # transfer functions
        x = (x.permute(0, 2, 1) @ self.mixer_matrix).permute(0, 2, 1)

        batch, channels, time = x.shape

        assert channels == self.control_plane_dim

        windowed = windowed_audio(x, self.window_size, self.window_size // 2)

        spec = torch.fft.rfft(windowed, dim=-1)

        batch, channels, frames, coeffs = spec.shape

        output_frames = []

        identity = torch.eye(self.n_coeffs, device=x.device)

        for i in range(frames):
            current = spec[:, :, i: i + 1, :]
            orig_norm = torch.norm(current, dim=-1, keepdim=True)

            if i > 0:
                current = current + output_frames[i - 1]

            current_transfer = self.transfer[None, :, None, :]

            if deformations is not None and self.hyper is not None:
                # TODO: This should use the norm-perserving non-linearity
                current_deformation = deformations[:, :, i, :]
                w, func = self.hyper.forward(
                    current_deformation.view(-1, self.deformation_latent_shape),
                    identity[None, ...]
                )
                cdm = func(current_transfer.view(-1, self.n_coeffs))
                cdm = cdm.view(batch, control_plane_dim, 1, self.n_coeffs)
                current_transfer = cdm

            if self.filter_bank is not None:
                # TODO: This should use the norm-preserving non-linearity
                filtered = current_transfer @ self.filter_bank
                filtered = filtered * current
            else:
                # perform convolution in the frequency domain
                filtered = current * current_transfer


            if self.preserve_energy:
                filtered = limit_norm(filtered, dim=-1, max_norm=orig_norm * 0.9999)

            # TODO: as a *hack*, I could simply preserve norm here
            # given the input norm, although this feels like a hack;
            # ideally the operations representing the deformation and
            # frequency mapping are norm preserving
            output_frames.append(filtered)

        output = torch.cat(output_frames, dim=2)
        audio_windows = torch.fft.irfft(output, dim=-1)
        samples = overlap_add(audio_windows, apply_window=True)
        samples = samples * self.gain
        samples = torch.tanh(samples)
        x = samples[..., :time]
        return x


class AudioNetwork(nn.Module):
    def __init__(
            self,
            control_plane_dim: int,
            window_size: int,
            n_blocks: int,
            deformation_latent_dim: int = None,
            filter_bank: torch.Tensor = None,
            preserve_energy: bool = False):

        super().__init__()
        self.window_size = window_size
        self.n_blocks = n_blocks
        self.mixer = nn.Parameter(torch.zeros((n_blocks + 1,)))
        self.control_plane_dim = control_plane_dim
        self.n_coeffs = window_size // 2 + 1

        self.blocks = nn.ModuleList([
            Block(
                window_size,
                control_plane_dim,
                self.init_transfer(),
                torch.zeros(1).uniform_(1, 50).item(),
                deformation_latent_shape=deformation_latent_dim,
                filter_bank=filter_bank,
                preserve_energy=preserve_energy)
            for _ in range(self.n_blocks)
        ])

    def init_transfer(self):
        resonances = torch.zeros(self.control_plane_dim, self.n_coeffs).uniform_(0.8, 0.95)
        sparse = torch.zeros_like(resonances).bernoulli_(p=0.01)
        resonances = resonances * sparse
        scaled_resonances = resonances
        return scaled_resonances

    def forward(self, x: torch.Tensor, deformations: Iterable[torch.Tensor]) -> torch.Tensor:
        outputs = [x[..., None]]
        inp = x

        for i, block in enumerate(self.blocks):
            inp = block(inp, deformations[i])
            outputs.append(inp[..., None])

        result = torch.cat(outputs, dim=-1)
        mixer_values = torch.softmax(self.mixer, dim=-1)
        mixed = (result * mixer_values[None, None, None, :]).sum(dim=-1)
        mixed = torch.sum(mixed, dim=1, keepdim=True)
        return mixed


class OverfitAudioNetwork(nn.Module):
    def __init__(
            self,
            window_size: int = 2048,
            control_plane_dim: int = 16,
            low_rank_deformation_dim: int = 16,
            n_samples: int = 2**15,
            n_frames: int = 128,
            n_layers: int = 3):

        super().__init__()

        self.n_samples = n_samples
        self.n_layers = n_layers

        env = torch.linspace(1, 0, steps=128) ** 10
        env = torch.cat([env, torch.zeros(self.n_samples - 128)], dim=-1)
        self.register_buffer('env', env)

        msh = musical_scale_hz(start_midi=21, stop_midi=129, n_steps=transfer_dim)
        # print([f'{x:.2f}' for x in msh])

        # establish (optional) non-linear frequency space
        fb = morlet_filter_bank(
            samplerate,
            kernel_size=window_size,
            scale=msh,
            scaling_factor=0.025,
            normalize=True,
            device=None)

        control_plane = torch.zeros(1, control_plane_dim, n_frames).uniform_(-1, 1)
        self.control_plane = nn.Parameter(control_plane)

        # establish deformation matrix (batch, control_plane_dim, n_frames, low_rank_deformation_dim)
        dm = [
            torch.zeros(
                1,
                control_plane_dim,
                n_frames,
                low_rank_deformation_dim).uniform_(-0.1, 0.1)
            for _ in range(n_layers)
        ]
        self.deformations = nn.ParameterList(dm)

        self.network = AudioNetwork(
            control_plane_dim,
            window_size,
            n_blocks=n_blocks,
            deformation_latent_dim=low_rank_deformation_dim,
            filter_bank=fb,
            preserve_energy=False
        )

    def _upsampled_control_plane(self):
        noise = torch.zeros((self.n_samples,), device=self.control_plane.device).uniform_(-1, 1)
        us = upsample_with_holes(torch.relu(self.control_plane), self.n_samples)
        us = fft_convolve(us, self.env[None, None, :])
        us = us * noise
        return us

    def forward(self):
        cp = self._upsampled_control_plane()
        result = self.network.forward(cp, self.deformations)
        return result



if __name__ == '__main__':
    batch_size = 1
    window_size = 2048
    step_size = window_size // 2
    transfer_dim = window_size // 2 + 1
    control_plane_dim = 16
    low_rank_deformation_dim = 16
    n_samples = 2 ** 16
    n_frames = n_samples // step_size
    n_blocks = 3
    samplerate = 22050
    nyquist = samplerate // 2

    model = OverfitAudioNetwork(
        window_size=window_size,
        control_plane_dim=control_plane_dim,
        low_rank_deformation_dim=low_rank_deformation_dim,
        n_samples=n_samples,
        n_frames=n_frames,
        n_layers=n_blocks
    )

    result = model.forward()
    print(result.shape)

    # msh = musical_scale_hz(start_midi=21, stop_midi=129, n_steps=transfer_dim)
    # print([f'{x:.2f}' for x in msh])
    #
    # # establish (optional) non-linear frequency space
    # fb = morlet_filter_bank(
    #     samplerate,
    #     kernel_size=window_size,
    #     scale=msh,
    #     scaling_factor=0.025,
    #     normalize=True,
    #     device=None)
    #
    #
    # # establish energy inputs (batch, control_plane_dim, n_samples)
    # impulse_size = 16
    # impulse = torch.zeros(impulse_size).uniform_(-1, 1)
    # impulse = impulse * torch.hann_window(impulse_size)
    #
    # # TODO: upscale with exponential decay envelopes
    # inp = torch.zeros(1, control_plane_dim, n_samples)
    #
    # inp[:, 0, 1024: 1024 + impulse_size] += impulse * 0.1
    # inp[:, 1, 8192: 8192 + impulse_size] += impulse * 0.5
    # inp[:, 2, 16384: 16384 + impulse_size] += impulse
    #
    # # establish deformation matrix (batch, control_plane_dim, n_frames, low_rank_deformation_dim)
    # dm = [
    #     torch.zeros(
    #         1,
    #         control_plane_dim,
    #         n_frames,
    #         low_rank_deformation_dim).uniform_(-0.1, 0.1)
    #     for _ in range(n_blocks)
    # ]
    #
    # network = AudioNetwork(
    #     control_plane_dim,
    #     window_size,
    #     n_blocks=n_blocks,
    #     deformation_latent_dim=low_rank_deformation_dim,
    #     filter_bank=fb,
    #     preserve_energy=False
    # )
    #
    # samples = network.forward(inp, dm)
    #
    # print(samples.min(), samples.max())
    # samples = playable(samples, samplerate=22050, normalize=True)
    # listen_to_sound(samples)
