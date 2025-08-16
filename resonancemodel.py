from typing import Tuple, Callable, Any

import numpy as np
import torch

from conjure import serve_conjure, SupportedContentType, NumpyDeserializer, NumpySerializer
from torch import nn
from torch.optim import Adam

import conjure
from data import get_one_audio_segment
from modules import max_norm, interpolate_last_axis, sparsify, unit_norm, flattened_multiband_spectrogram, \
    fft_frequency_recompose, stft, HyperNetworkLayer
from modules.transfer import freq_domain_transfer_function_to_resonance, fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import device, encode_audio, make_initializer

MaterializeResonances = Callable[..., torch.Tensor]

init_weights = make_initializer(0.02)


def execute_layer(
        control_signal: torch.Tensor,
        routing: torch.Tensor,
        # materialize_resonances: MaterializeResonances,
        res: torch.Tensor,
        deformations: torch.Tensor,
        gains: torch.Tensor,
        window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # n_resonances, n_coeffs, expressivity = amp.shape
    batch, n_events, control_plane_dim, frames = control_signal.shape
    batch, n_events, expressivity, def_frames = deformations.shape
    cpd, nr = routing.shape

    # TODO: control signal sparsity enforced here?

    # first determine routing
    # TODO: einops
    routed = (control_signal.permute(0, 1, 3, 2) @ routing).permute(0, 1, 3, 2)

    before_upsample = routed

    step_size = window_size // 2
    # n_resonance_frames = n_samples // step_size

    # res = materialize_resonances()
    _, _, n_resonances, expressivity, n_samples = res.shape

    # ensure that resonances all have unit norm so that control
    # plane energy drives overall loudness/response
    res = unit_norm(res)

    # "route" energy from control plane to resonances
    routed = routed.view(batch, n_events, n_resonances, 1, frames)

    # convolve with noise impulse
    routed = upsample_with_holes(routed, n_samples)
    impulse = torch.hamming_window(128, device=routed.device)
    impulse = impulse * torch.zeros_like(impulse).uniform_(-1, 1)
    impulse = ensure_last_axis_length(impulse, n_samples)
    impulse = impulse.view(1, 1, 1, 1, n_samples)
    routed = fft_convolve(impulse, routed)

    # interpolate and multiply with noise
    # routed = interpolate_last_axis(routed, n_samples)
    # routed = routed * torch.zeros_like(routed).uniform_(-1, 1)

    # convolve control plane with all resonances
    conv = fft_convolve(routed, res)

    # interpolate between variations on each resonance
    base_deformation = torch.zeros_like(deformations)
    base_deformation[:, :, 0:1, :] = 1
    d = base_deformation + deformations
    # d = deformations
    d = torch.softmax(d, dim=-2)
    # d = torch.relu(d)
    d = d.view(batch, n_events, 1, expressivity, def_frames)
    d = interpolate_last_axis(d, n_samples)

    x = d * conv
    x = torch.sum(x, dim=-2)

    summed = torch.tanh(x * torch.abs(gains.view(1, 1, n_resonances, 1)))

    summed = torch.sum(summed, dim=-2, keepdim=True)

    return summed, before_upsample


class MultibandFFTResonanceBlock(nn.Module):

    def __init__(
            self,
            n_resonances: int,
            n_samples: int,
            expressivity: int,
            smallest_band_size: int = 512,
            base_resonance: float = 0.2,
            window_size: int = 64):
        super().__init__()

        # def init(x):
        #     return torch.zeros_like(x).uniform_(-6, 6) * torch.zeros_like(x).bernoulli_(p=0.01)

        step_size = window_size // 2

        full_size_log2 = int(np.log2(n_samples))
        small_size_log2 = int(np.log2(smallest_band_size))
        band_sizes = [2 ** x for x in range(small_size_log2, full_size_log2, 1)]
        n_bands = len(band_sizes)

        n_coeffs = window_size // 2 + 1
        # magnitude and phase for each band
        params_per_band = n_coeffs * 3
        total_params_per_item = params_per_band * n_bands

        def init_resonance() -> torch.Tensor:
            # base resonance
            res = torch.zeros((n_resonances, total_params_per_item, 1)).uniform_(0.01, 1)
            # variations or deformations of the base resonance
            deformation = torch.zeros((1, total_params_per_item, expressivity)).uniform_(-0.02, 0.02)
            # expand into (n_resonances, n_deformations)
            return res + deformation

        self.resonances = nn.ParameterDict(dict(
            amp=init_resonance(),
            phase=init_resonance(),
            decay=init_resonance(),
        ))
        # super().__init__(n_items, total_params_per_item, selection_type='relu', initialize=init)

        self.n_samples = n_samples
        self.n_resonances = n_resonances
        self.base_resonance = base_resonance
        self.resonance_span = 1 - base_resonance
        self.frames_per_band = [(size // step_size) for size in band_sizes]
        self.total_frames = (n_samples // step_size) * 2
        self.band_sizes = band_sizes
        self.n_bands = n_bands
        self.params_per_band = params_per_band
        self.n_coeffs = n_coeffs
        self.window_size = window_size
        # self.padded_samples = n_samples * 2
        self.expressivity = expressivity

    def forward(self) -> torch.Tensor:
        # batch, n_events, expressivity, n_params = items.shape

        bands = dict()

        for i, size in enumerate(self.band_sizes):
            start = i * self.params_per_band
            stop = start + self.params_per_band

            # band_params = items[:, :, :, start: stop]

            mag = self.resonances['decay'][:, :self.n_coeffs, :]
            phase = self.resonances['phase'][:, self.n_coeffs:self.n_coeffs * 2, :]
            start = self.resonances['amp'][:, -self.n_coeffs:, :]

            mag = self.base_resonance + ((torch.sigmoid(mag) * self.resonance_span) * 0.9999)
            phase = torch.tanh(phase) * np.pi
            start = torch.sigmoid(start)

            band = freq_domain_transfer_function_to_resonance(
                window_size=self.window_size,
                coeffs=mag,
                n_frames=self.frames_per_band[i],
                apply_decay=True,
                start_phase=phase,
                start_mags=start
            )
            bands[size] = band
            # bands[size] = ensure_last_axis_length(band, size * 2)

        full = fft_frequency_recompose(bands, desired_size=self.n_samples)
        full = full[..., :self.n_samples]
        full = full.reshape(1, 1, self.n_resonances, self.expressivity, -1)
        # full = unit_norm(full)
        return full


def damped_harmonic_oscillator(
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor,
        initial_velocity: float,
) -> torch.Tensor:
    _, n_samples = time.shape
    # mass = mass.view(n_osc, 1)
    # damping = damping.view(n_osc, 1)
    # tension = tension.view(n_osc, 1)
    # initial_displacement = initial_displacement.view(n_osc, 1)
    # initial_velocity = initial_velocity.view(n_osc, 1)

    x = (damping / (2 * mass))
    omega = torch.sqrt(tension - (x ** 2))
    phi = torch.arctan((initial_velocity + (x * initial_displacement) / (initial_displacement * omega)))
    # phi = torch.atan2(
    #     (initial_velocity + (x * initial_displacement)),
    #     (initial_displacement * omega)
    # )
    a = initial_displacement / torch.cos(phi)
    z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    return z


class DampedHarmonicOscillatorBlock(nn.Module):
    def __init__(
            self,
            n_samples: int,
            n_oscillators: int,
            n_resonances: int,
            expressivity: int):
        super().__init__()
        self.n_samples = n_samples
        self.n_oscillators = n_oscillators
        self.n_resonances = n_resonances
        self.expressivity = expressivity

        self.mass = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-2, 2))

        self.damping = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(0.1, 1.1))

        self.tension = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(4, 9))

        self.initial_displacement = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-1, 2))

    def _materialize_resonances(self, device: torch.device):
        time = torch.linspace(0, 1, self.n_samples, device=device) \
            .view(1, self.n_samples)
        x = damped_harmonic_oscillator(
            time,
            # 1e-8 + self.mass.view(-1, 1) ** 2,
            1e-8 + (10 ** self.mass.view(-1, 1)),
            1e-8 + self.damping.view(-1, 1) ** 2,
            # 1e-8 + self.tension.view(-1, 1) ** 2,
            1e-8 + (10 ** self.tension.view(-1, 1)),
            1e-8 + (10 ** self.initial_displacement.view(-1, 1)),
            # 1e-8 + self.initial_displacement.view(-1, 1) ** 2,
            0
        )
        print(x.shape)
        x = x.view(self.n_oscillators, self.n_resonances, self.expressivity, self.n_samples)
        x = torch.sum(x, dim=0)
        return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples)

    def forward(self) -> torch.Tensor:
        return self._materialize_resonances(self.damping.device)


class LatentResonanceBlock(nn.Module):
    def __init__(
            self,
            n_samples: int,
            n_resonances: int,
            expressivity: int,
            latent_dim: int):

        super().__init__()
        self.n_samples = n_samples
        self.n_resonances = n_resonances
        self.expressivity = expressivity
        self.latent_dim = latent_dim

        n_coeffs = n_samples // 2 + 1
        total_coeffs = n_coeffs * 2


        self.mapping = HyperNetworkLayer(latent_dim, latent_dim, latent_dim, total_coeffs, bias=False)
        self.layer_latents = nn.Parameter(
            torch.zeros(n_resonances, latent_dim).uniform_(-1, 1))
        self.resonances = nn.Parameter(
            torch.zeros(n_resonances, expressivity, latent_dim).uniform_(-1, 1))

    def forward(self) -> torch.Tensor:
        w, fwd = self.mapping.forward(self.layer_latents)
        mapped = fwd(self.resonances)
        mapped = mapped.view(self.n_resonances, self.expressivity, -1, 2)
        mapped = torch.view_as_complex(mapped)
        res = torch.fft.irfft(mapped)
        return res.view(1, 1, self.n_resonances, self.expressivity, self.n_samples)

class FFTResonanceBlock(nn.Module):
    def __init__(
            self,
            n_samples: int,
            resonance_window_size: int,
            n_resonances: int,
            expressivity: int,
            base_resonance: float = 0.5):
        super().__init__()
        self.n_samples = n_samples
        self.resonance_window_size = resonance_window_size
        self.n_resonances = n_resonances
        self.expressivity = expressivity
        self.base_resonance = base_resonance

        resonance_coeffs = resonance_window_size // 2 + 1

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

    def _materialize_fft_resonances(
            self,
            window_size: int,
            base_resonance: float,
            n_samples: int,
            amp: torch.Tensor,
            phase: torch.Tensor,
            decay: torch.Tensor) -> torch.Tensor:
        res_span = 1 - base_resonance
        res_factor = 0.9

        n_resonances, n_coeffs, expressivity = amp.shape

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
        return res

    def forward(self, *args) -> torch.Tensor:
        return self._materialize_fft_resonances(
            self.resonance_window_size,
            self.base_resonance,
            self.n_samples,
            self.resonances['amp'],
            self.resonances['phase'],
            self.resonances['decay']
        )


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

        # self.resonance = DampedHarmonicOscillatorBlock(
        #     n_samples, 32, n_resonances, expressivity
        # )

        # self.resonance = LatentResonanceBlock(
        #     n_samples, n_resonances, expressivity, latent_dim=16)

        self.resonance = FFTResonanceBlock(
            n_samples, resonance_window_size, n_resonances, expressivity, base_resonance)

        # self.resonance = MultibandFFTResonanceBlock(
        #     n_resonances,
        #     n_samples,
        #     expressivity,
        #     smallest_band_size=2048,
        #     base_resonance=0.00001,
        #     window_size=64)

        # def init_resonance() -> torch.Tensor:
        #     # base resonance
        #     res = torch.zeros((n_resonances, resonance_coeffs, 1)).uniform_(0.01, 1)
        #     # variations or deformations of the base resonance
        #     deformation = torch.zeros((1, resonance_coeffs, expressivity)).uniform_(-0.02, 0.02)
        #     # expand into (n_resonances, n_deformations)
        #     return res + deformation
        #
        # self.resonances = nn.ParameterDict(dict(
        #     amp=init_resonance(),
        #     phase=init_resonance(),
        #     decay=init_resonance(),
        # ))

        self.gains = nn.Parameter(torch.zeros((n_resonances, 1)).uniform_(0.01, 1.1))

    def forward(
            self,
            control_signal: torch.Tensor,
            deformations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.resonance.forward()
        print(res.shape)
        output, fwd = execute_layer(
            control_signal,
            self.router,
            res,
            deformations,
            self.gains,
            self.n_samples,
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

        self.apply(init_weights)

    def _process_control_plane(
            self,
            cp: torch.Tensor,
            n_to_keep: int = 128) -> torch.Tensor:
        cp = cp.view(1, self.control_plane_dim, self.n_frames)
        cp = sparsify(cp, n_to_keep=n_to_keep)
        cp = cp.view(1, 1, self.control_plane_dim, self.n_frames)
        return cp

    @property
    def control_signal(self):
        cp = self.control_plane
        cp = self._process_control_plane(cp)
        return cp

    def random(self, use_learned_deformations: bool = False):
        rcp = torch \
            .zeros_like(self.control_plane) \
            .uniform_(
            self.control_plane.min().item(),
            self.control_plane.max().item())

        rcp = self._process_control_plane(rcp, n_to_keep=32)
        x = self.forward(
            rcp, self.deformations if use_learned_deformations else torch.zeros_like(self.deformations))
        return x

    def forward(
            self,
            cp: torch.Tensor = None,
            deformations: torch.Tensor = None):
        cp = cp if cp is not None else self.control_signal  # / self.control_plane.sum()
        deformations = deformations if deformations is not None else self.deformations
        x = self.network.forward(cp, deformations)
        return x

    def compression_ratio(self, n_samples: int):
        # thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        return (self.deformations.numel() + count_parameters(self.network) + 256) / n_samples


def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()
    forward = mask
    backward = x
    y = backward + (forward - backward).detach()
    return y.sum()


def overfit_model():
    n_samples = 2 ** 16
    resonance_window_size = 2048
    step_size = 1024
    n_frames = n_samples // step_size

    # KLUDGE: control_plane_dim and n_resonances
    # must have the same value
    control_plane_dim = 32
    n_resonances = 32
    expressivity = 4

    target = get_one_audio_segment(n_samples)
    model = OverfitResonanceStack(
        n_layers=1,
        n_samples=n_samples,
        resonance_window_size=resonance_window_size,
        control_plane_dim=control_plane_dim,
        n_resonances=n_resonances,
        expressivity=expressivity,
        base_resonance=0.5,
        n_frames=n_frames
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-2)
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
            # x = stft(target, 2048, 256, pad=True)
            # y = stft(recon, 2048, 256, pad=True)
            loss = torch.abs(x - y).sum()

            loss.backward()
            optimizer.step()
            print(iteration, loss.item(), model.compression_ratio(n_samples))

            with torch.no_grad():
                rand(max_norm(model.random(use_learned_deformations=False)))

            iteration += 1

    train()


if __name__ == '__main__':
    overfit_model()
