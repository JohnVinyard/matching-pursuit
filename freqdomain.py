from itertools import count
from typing import Union, List, Iterable, Optional

import numpy as np
import torch
from scipy.signal import morlet
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from conjure import LmdbCollection, loggers, serve_conjure, SupportedContentType, NumpySerializer, NumpyDeserializer
from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import HyperNetworkLayer, limit_norm, flattened_multiband_spectrogram, max_norm, stft, \
    sparse_softmax, iterative_loss
from modules.normal_pdf import pdf2
from modules.overlap_add import overlap_add
from modules.phase import windowed_audio
from modules.transfer import fft_convolve
from util import device, make_initializer
from util.music import musical_scale_hz

n_samples = 2 ** 17
transform_window_size = 2048
transform_step_size = 256
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

        # TODO: should there be a separate gain for each channel of control plane dim?
        self.gain = nn.Parameter(torch.ones((control_plane_dim,)).fill_(gain))

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
                cdm = cdm.view(batch, self.control_plane_dim, 1, self.n_coeffs)
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
        samples = samples * self.gain[None, :, None]
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
        resonances = torch.zeros(self.control_plane_dim, self.n_coeffs).uniform_(0.5, 0.9998)
        sparse = torch.zeros_like(resonances).bernoulli_(p=0.01)
        resonances = resonances * sparse
        scaled_resonances = resonances
        return scaled_resonances

    def forward(self, x: torch.Tensor, deformations: Iterable[torch.Tensor]) -> torch.Tensor:
        outputs = [x[..., None]]
        inp = x

        for i, block in enumerate(self.blocks):
            try:
                deform = deformations[i]
            except (TypeError, IndexError):
                deform = None
            inp = block(inp, deform)
            outputs.append(inp[..., None])

        result = torch.cat(outputs, dim=-1)
        mixer_values = torch.softmax(self.mixer, dim=-1)
        mixed = (result * mixer_values[None, None, None, :]).sum(dim=-1)
        mixed = torch.sum(mixed, dim=1, keepdim=True)
        return mixed



def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()
    forward = mask
    backward = x
    y = backward + (forward - backward).detach()
    return y.sum()


class OverfitAudioNetwork(nn.Module):
    def __init__(
            self,
            window_size: int = 2048,
            control_plane_dim: int = 16,
            low_rank_deformation_dim: int = 16,
            n_samples: int = 2 ** 15,
            n_frames: int = 128,
            n_layers: int = 3,
            impulse_decay_samples: int = 128,
            samplerate: int = 22050,
            deformations_enabled: bool = False,
            preserve_energy: bool = False,
            n_events: int = 32):

        super().__init__()

        self.n_samples = n_samples
        self.n_layers = n_layers
        self.deformations_enabled = deformations_enabled
        self.n_frames = n_frames
        self.n_events = n_events

        transfer_dim = window_size // 2 + 1

        self.event_vectors = nn.Parameter(torch.zeros(1, n_events, control_plane_dim).uniform_(0, 1e-8))
        self.event_envelopes = nn.Parameter(torch.zeros(1, n_events, 2).uniform_(0, 1))
        self.event_times = nn.Parameter(torch.zeros(1, n_events, n_samples).uniform_(-1, 1))

        self.channel_decays = nn.Parameter(torch.zeros((control_plane_dim,)))

        msh = musical_scale_hz(start_midi=21, stop_midi=129, n_steps=transfer_dim)

        # establish (optional) non-linear frequency space
        fb = morlet_filter_bank(
            samplerate,
            kernel_size=window_size,
            scale=msh,
            scaling_factor=0.025,
            normalize=True,
            device=None)

        control_plane = torch.zeros(1, control_plane_dim, n_frames).uniform_(0, 1e-8)
        self.control_plane = nn.Parameter(control_plane)
        self.control_plane_dim = control_plane_dim

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
            n_blocks=n_layers,
            deformation_latent_dim=low_rank_deformation_dim if self.deformations_enabled else None,
            filter_bank=fb,
            preserve_energy=preserve_energy
        )


    @property
    def control_signal(self):
        return torch.relu(self.control_plane)

    @property
    def all_deformations(self):
        x = torch.stack([d for d in self.deformations], dim=0)
        return x

    def _random_gaussian_events(self):

        ee = torch.zeros_like(self.event_envelopes).uniform_(0, 1)
        et = torch.zeros_like(self.event_times).uniform_(0, 1)

        means = ee[:, :, 0] * 0
        stds = torch.abs(ee[:, :, 1] + 1e-8) * 0.1
        x = pdf2(means, stds, n_elements=self.n_samples)
        x = x[:, :, None, :]

        # place the event vectors at the start of a vector, ready to be positioned
        ev = self.event_vectors[:, :, :, None]
        ev = torch.cat(
            [ev, torch.zeros((1, self.n_events, self.control_plane_dim, self.n_samples - 1), device=ev.device)], dim=-1)

        # TODO: Try gumbel softmax with a decaying temperature
        t = sparse_softmax(et, dim=-1, normalize=True)
        # t = F.gumbel_softmax(self.event_times, tau=0.1, hard=True)

        # convolve the times (dirac functions) with the pdfs, such that the PDFs
        # are now "scheduled", or shifted to the appropriate times
        x = fft_convolve(t[:, :, None, :], x)

        # next convolve the event vectors with the PDFs
        # TODO: Is this necessary, or could I just multiply with broadcasting?
        x = fft_convolve(x, ev)
        # print(x.shape, ev.shape)
        # x = x * ev

        # x = torch.sum(x, dim=1)
        return x

    def _gaussian_events(self) -> torch.Tensor:

        # get the probability density function given our mean and std parameters
        means = self.event_envelopes[:, :, 0] * 0
        stds = torch.abs(self.event_envelopes[:, :, 1] + 1e-8) * 0.1
        x = pdf2(means, stds, n_elements=self.n_samples)
        x = x[:, :, None, :]

        # place the event vectors at the start of a vector, ready to be positioned
        ev = self.event_vectors[:, :, :, None]
        ev = torch.cat(
            [ev, torch.zeros((1, self.n_events, self.control_plane_dim, self.n_samples - 1), device=ev.device)], dim=-1)

        # TODO: Try gumbel softmax with a decaying temperature
        t = sparse_softmax(self.event_times, dim=-1, normalize=True)
        # t = F.gumbel_softmax(self.event_times, tau=0.1, hard=True)

        print(t.shape, x.shape)
        # convolve the times (dirac functions) with the pdfs, such that the PDFs
        # are now "scheduled", or shifted to the appropriate times
        x = fft_convolve(t[:, :, None, :], x)

        print(x.shape, ev.shape)
        # next convolve the event vectors with the PDFs
        # TODO: Is this necessary, or could I just multiply with broadcasting?
        x = fft_convolve(x, ev)
        print(x.shape)
        # print(x.shape, ev.shape)
        # x = x * ev

        # x = torch.sum(x, dim=1)
        return x

    def forward(self, sig=None, random=False):
        if sig is not None:
            result = sig
            control_signal = sig
            uscp = result = self._upsampled_control_plane(result)
        else:
            if random:
                result = control_signal = self._random_gaussian_events()
            else:
                result = control_signal = self._gaussian_events()
            batch, n_events, cp, time = result.shape
            result = control_signal.reshape(batch * n_events, cp, time)

        result = self.network.forward(result, self.deformations)

        if sig is not None:
            control_signal = F.avg_pool1d(uscp, kernel_size=512, stride=256, padding=256)
        else:
            result = result.reshape(-1, self.n_events, result.shape[-1])
            control_signal = torch.sum(control_signal, dim=1)
            # downsample so this is viewable
            control_signal = F.avg_pool1d(control_signal, kernel_size=512, stride=256, padding=256)

        # result = torch.sum(result, dim=1, keepdim=True)

        return result, control_signal


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


def compute_spec(audio: torch.Tensor, model: nn.Module) -> torch.Tensor:
    spec = stft_transform(audio)
    vec = model.forward(spec)
    return vec


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the difference between the `recon` and `target`
    representations
    """
    fake_spec = transform(recon)
    real_spec = transform(target)
    return torch.abs(fake_spec - real_spec).sum()


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the control signal
    """
    # return torch.abs(c).sum() * 1e-2
    return l0_norm(c)


def construct_experiment_model(n_samples: int, n_events: int = 32) -> OverfitAudioNetwork:
    window_size = 2048
    control_plane_dim = 16
    low_rank_deformation_dim = 16
    n_frames = n_samples // 256
    n_blocks = 3
    samplerate = 22050

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
        n_events=n_events
    ).to(device)
    return model


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()



def train_and_monitor_overfit_model(
        n_samples: int,
        n_events: int = 32,
        samplerate: int = 22050,
        audio_path: Optional[str] = None):
    target = get_one_audio_segment(
        n_samples=n_samples, samplerate=samplerate, audio_path=audio_path)
    collection = LmdbCollection(path='freqdomain')

    print(f'overfitting to {n_samples // samplerate} seconds with {n_events} events')

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

    def train(target: torch.Tensor):
        model = construct_experiment_model(n_samples=n_samples, n_events=n_events)

        optim = Adam(model.parameters(), lr=1e-3)

        for iteration in count():
            optim.zero_grad()
            recon, control_signal = model.forward()

            recon_audio(max_norm(recon.sum(dim=1, keepdim=True)))

            loss = iterative_loss(target, recon, transform)
            # recon_loss = reconstruction_loss(recon, target)

            # loss = recon_loss + sparsity_loss(control_signal)

            if model.deformations_enabled:
                loss = loss + sparsity_loss(model.all_deformations)

            non_zero = (control_signal > 0).sum()
            sparsity = (non_zero / control_signal.numel()).item()

            loss.backward()

            envelopes(max_norm(control_signal[0]))

            optim.step()
            print(iteration, loss.item(), sparsity)

            with torch.no_grad():
                # log random output from the model
                r, _ = model.forward(sig=None, random=True)
                r = torch.sum(r, dim=1, keepdim=True)
                rnd(max_norm(r))

    train(target)


if __name__ == '__main__':
    train_and_monitor_overfit_model(
        n_samples=2 ** 16,
        samplerate=22050,
        n_events=32)
    # train_and_monitor_auto_encoder(batch_size=2)
