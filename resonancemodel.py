from typing import Tuple, Callable, Any, Union

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
from modules.phase import mag_phase_decomposition
from modules.transfer import freq_domain_transfer_function_to_resonance, fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from spiking import SpikingModel
from util import device, encode_audio, make_initializer
from base64 import b64encode
from sklearn.decomposition import DictionaryLearning

MaterializeResonances = Callable[..., torch.Tensor]

init_weights = make_initializer(0.02)

'''
import numpy as np
import matplotlib.pyplot as plt

# 1. Define signal parameters
A = 1.0        # Amplitude
alpha = 0.5    # Damping factor
f0 = 5         # Frequency in Hz
phi = np.pi/4  # Phase in radians

# 2. Define sampling parameters
Fs = 100       # Sampling frequency (Hz)
N = 1024       # Number of samples
T = N / Fs     # Time duration

# 3. Create frequency domain signal
# Generate frequency axis
omega0 = 2 * np.pi * f0
omega = 2 * np.pi * np.fft.fftfreq(N, 1/Fs)

# Compute the frequency domain function X(omega)
X_omega_pos = (A/2) * np.exp(1j*phi) / (alpha + 1j * (omega - omega0))
X_omega_neg = (A/2) * np.exp(-1j*phi) / (alpha + 1j * (omega + omega0))
X_omega = X_omega_pos + X_omega_neg

# 4. Take the inverse FFT to get the time domain signal
x_t = np.fft.ifft(X_omega) * N

# Optional: Add a windowing function to reduce artifacts
window = np.hamming(N)
x_t_windowed = np.fft.ifft(X_omega * window) * N

# Plot the results
t = np.arange(N) / Fs

plt.figure(figsize=(12, 8))

# Plot time-domain signal
plt.subplot(2, 1, 1)
plt.plot(t, np.real(x_t), label='Time-domain signal')
plt.title('Damped Sinusoid in Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

# Plot frequency-domain magnitude
plt.subplot(2, 1, 2)
# The `fftshift` function centers the zero-frequency component
plt.plot(np.fft.fftshift(omega) / (2 * np.pi), np.fft.fftshift(np.abs(X_omega)))
plt.title('Magnitude Spectrum in Frequency Domain')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.xlim([-20, 20])
plt.tight_layout()
plt.show()

'''


def decaying_noise(
        n_items: int,
        n_samples: int,
        low_exp: int,
        high_exp: int,
        device: torch.device,
        include_noise: bool = True):
    t = torch.linspace(1, 0, n_samples, device=device)
    pos = torch.zeros(n_items, device=device).uniform_(low_exp, high_exp)

    if include_noise:
        noise = torch.zeros(n_items, n_samples, device=device).uniform_(-1, 1)
        return (t[None, :] ** pos[:, None]) * noise
    else:
        return (t[None, :] ** pos[:, None])


def materialize_non_windowed_fft_resonance(n_samples: int, amplitudes: torch.Tensor, damping: torch.Tensor):
    phi = np.pi / 4
    X_omega_pos = (amplitudes / 2) * np.exp(1j * phi) / (damping + 1j)
    X_omega_neg = (amplitudes / 2) * np.exp(-1j * phi) / (damping + 1j)
    X_omega = X_omega_pos + X_omega_neg

    # 4. Take the inverse FFT to get the time domain signal
    x_t = np.fft.ifft(X_omega)
    return x_t


class SampleLookupBlock(nn.Module):
    def __init__(
            self,
            n_items: int,
            n_samples: int,
            flatten_kernel_size: Union[int, None] = None,
            initial: Union[torch.Tensor, None] = None,
            randomize_phases: bool = False,
            windowed: bool = False):
        super().__init__()

        self.n_samples = n_samples
        self.decays = nn.Parameter(torch.zeros(n_items).uniform_(0, 1))
        self.latent = nn.Parameter(torch.zeros(n_items, n_items).uniform_(-0.01, 0.01))
        self.network = SampleLookup(n_items, n_samples, None, initial, randomize_phases, windowed)

    def forward(self):
        t = torch.linspace(1, 0, self.n_samples, device=self.decays.device)[None, :]

        exps = 4 + (torch.sigmoid(self.decays) * 90)[:, None]
        decays = t ** exps

        x = self.network(self.latent) * decays
        return x.view(1, 1, -1, 2, self.n_samples)


class SampleLookup(Lookup):

    def __init__(
            self,
            n_items: int,
            n_samples: int,
            flatten_kernel_size: Union[int, None] = None,
            randomize_phases: bool = True,
            windowed: bool = False):

        def sample_lookup_init(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x).uniform_(-0.01, 0.01)

        super().__init__(n_items, n_samples, initialize=sample_lookup_init, selection_type='identity')
        self.randomize_phases = randomize_phases
        self.flatten_kernel_size = flatten_kernel_size
        self.windowed = windowed

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have audio-rate samples at a relatively uniform
        amplitude throughout
        """
        if self.flatten_kernel_size:
            x = flatten_envelope(
                items,
                kernel_size=self.flatten_kernel_size,
                step_size=self.flatten_kernel_size // 2)
        else:
            x = items

        if self.randomize_phases:
            spec = torch.fft.rfft(x, dim=-1)
            # randomize phases
            mags = torch.abs(spec)
            phases = torch.angle(spec)
            phases = torch.zeros_like(phases).uniform_(-np.pi, np.pi)
            # imag = torch.cumsum(phases, dim=1)
            imag = phases
            # imag = (imag + np.pi) % (2 * np.pi) - np.pi
            spec = mags * torch.exp(1j * imag)
            x = torch.fft.irfft(spec, dim=-1)

        if self.windowed:
            x = x * torch.hamming_window(x.shape[-1], device=x.device)

        # TODO: Unit norm
        x = unit_norm(x)
        return x


def materialize_attack_envelopes(
        low_res: torch.Tensor,
        window_size: int,
        is_fft: bool = False) -> torch.Tensor:
    if is_fft:
        low_res = torch.view_as_complex(low_res)
        low_res = torch.fft.irfft(low_res)

    impulse = interpolate_last_axis(low_res, desired_size=window_size) ** 2

    impulse = impulse * torch.zeros_like(impulse).uniform_(-1, 1)
    return impulse


def execute_layer(
        control_signal: torch.Tensor,
        attack_envelopes: torch.Tensor,
        mix: torch.Tensor,
        routing: torch.Tensor,
        res: torch.Tensor,
        deformations: torch.Tensor,
        gains: torch.Tensor,
        window_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, n_events, control_plane_dim, frames = control_signal.shape
    batch, n_events, expressivity, def_frames = deformations.shape
    # cpd, nr = routing.shape

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

    # TODO: Try this with a gamma distribution, or a learnable distribution
    # impulse = torch.hamming_window(128, device=routed.device)

    # impulse = interpolate_last_axis(attack_envelopes ** 2, desired_size=window_size)
    # impulse = impulse * torch.zeros_like(impulse).uniform_(-1, 1)

    impulse = materialize_attack_envelopes(attack_envelopes, window_size)
    impulse = ensure_last_axis_length(impulse, n_samples)

    # print(impulse.shape, routed.shape)
    impulse = impulse.view(1, 1, control_plane_dim, 1, n_samples)
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

    # print('BEFORE', x.shape)

    mixes = mix.view(1, 1, n_resonances, 1, 1, 2)
    mixes = torch.softmax(mixes, dim=-1)
    stacked = torch.stack([routed, x.reshape(*routed.shape)], dim=-1)
    x = mixes * stacked
    x = torch.sum(x, dim=-1)

    x = x.view(1, 1, n_resonances, -1)
    # print('AFTER', x.shape)

    summed = torch.tanh(x * torch.abs(gains.view(1, 1, n_resonances, 1)))
    # summed = x

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
        params_per_band = n_coeffs * 4
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
            phase_dither=init_resonance()
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
            start = self.resonances['amp'][:, self.n_coeffs * 2:self.n_coeffs * 3, :]
            phase_dither = self.resonances['phase_dither'][:, -self.n_coeffs:, :]

            mag = mag.permute(0, 2, 1)
            phase = phase.permute(0, 2, 1)
            start = start.permute(0, 2, 1)
            phase_dither = phase_dither.permute(0, 2, 1)

            # print(mag.shape, phase.shape, start.shape, phase_dither.shape)

            band = freq_domain_transfer_function_to_resonance(
                window_size=self.window_size,
                coeffs=self.base_resonance + ((torch.clamp(mag, 0, 1) * self.resonance_span) * 0.9999),
                n_frames=self.frames_per_band[i],
                apply_decay=True,
                start_phase=torch.tanh(phase) * np.pi,
                start_mags=start ** 2,
                phase_dither=torch.tanh(phase_dither).reshape(-1, 1, self.n_coeffs),
                log_space_scan=True
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
                .uniform_(0.5, 1.5))

        self.tension = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(4, 9))

        self.initial_displacement = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-1, 2))

        self.amplitudes = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity, 1) \
                .uniform_(-1, 1))

    def _materialize_resonances(self, device: torch.device):
        time = torch.linspace(0, 10, self.n_samples, device=device) \
            .view(1, 1, 1, self.n_samples)

        x = damped_harmonic_oscillator(
            time=time,
            mass=torch.sigmoid(self.mass[..., None]),
            damping=torch.sigmoid(self.damping[..., None]) * 30,
            tension=10 ** self.tension[..., None],
            initial_displacement=self.initial_displacement[..., None],
            initial_velocity=0
        )

        x = x.view(self.n_oscillators, self.n_resonances, self.expressivity, self.n_samples)
        x = x * self.amplitudes ** 2
        x = torch.sum(x, dim=0)

        ramp = torch.ones(self.n_samples, device=device)
        ramp[:10] = torch.linspace(0, 1, 10, device=device)
        return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples) * ramp[None, None, None, None, :]

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
        self.n_coeffs = resonance_coeffs

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
            phase_dither=init_resonance()
        ))

    def _materialize_fft_resonances(
            self,
            window_size: int,
            base_resonance: float,
            n_samples: int,
            amp: torch.Tensor,
            phase: torch.Tensor,
            decay: torch.Tensor,
            phase_dither: torch.Tensor) -> torch.Tensor:
        res_span = 1 - base_resonance
        res_factor = 0.99

        n_resonances, n_coeffs, expressivity = amp.shape

        step_size = window_size // 2
        n_resonance_frames = n_samples // step_size

        amp = amp.permute(0, 2, 1)
        phase = phase.permute(0, 2, 1)
        decay = decay.permute(0, 2, 1)
        phase_dither = phase_dither.permute(0, 2, 1)

        # materialize resonances
        # print(amp.shape, decay.shape, phase_dither.shape)
        res = freq_domain_transfer_function_to_resonance(
            window_size,
            base_resonance + ((torch.sigmoid(decay) * res_span) * res_factor),
            n_resonance_frames,
            apply_decay=True,
            start_phase=torch.tanh(phase) * np.pi,
            start_mags=amp ** 2,
            phase_dither=torch.tanh(phase_dither).reshape(-1, 1, self.n_coeffs),
            log_space_scan=False,
            apply_window=False,
            overrlap_add=True)

        res = res.view(1, 1, n_resonances, expressivity, n_samples)
        return res

    def forward(self, *args) -> torch.Tensor:
        return self._materialize_fft_resonances(
            self.resonance_window_size,
            self.base_resonance,
            self.n_samples,
            self.resonances['amp'],
            self.resonances['phase'],
            self.resonances['decay'],
            self.resonances['phase_dither'])


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

        self.attack_envelopes = nn.Parameter(
            # decaying_noise(self.control_plane_dim, 256, 4, 20, device=device, include_noise=False)
            torch.zeros(self.control_plane_dim, 256).uniform_(-1, 1)
        )

        self.router = nn.Parameter(
            torch.zeros((self.control_plane_dim, self.n_resonances)).uniform_(-1, 1))

        # self.resonance = SampleLookupBlock(
        #     n_resonances * expressivity, n_samples, 64, randomize_phases=True, windowed=True)

        self.resonance = DampedHarmonicOscillatorBlock(
            n_samples, 16, n_resonances, expressivity
        )

        self.mix = nn.Parameter(torch.zeros(self.n_resonances, 2).uniform_(-1, 1))

        # self.resonance = LatentResonanceBlock(
        #     n_samples, n_resonances, expressivity, latent_dim=16)

        # self.resonance = FFTResonanceBlock(
        #     n_samples, resonance_window_size, n_resonances, expressivity, base_resonance)

        # self.resonance = MultibandFFTResonanceBlock(
        #     n_resonances,
        #     n_samples,
        #     expressivity,
        #     smallest_band_size=16384,
        #     base_resonance=0.01,
        #     window_size=512)

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

    def get_attack_envelopes(self):
        return materialize_attack_envelopes(self.attack_envelopes, self.resonance_window_size)

    def get_materialized_resonance(self):
        return self.resonance.forward()

    def get_gains(self):
        return self.gains

    def get_router(self):
        return self.router

    def forward(
            self,
            control_signal: torch.Tensor,
            deformations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.resonance.forward()
        # print(res.shape)
        output, fwd = execute_layer(
            control_signal,
            self.attack_envelopes,
            self.mix,
            self.router,
            res,
            deformations,
            self.gains,
            self.resonance_window_size,
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

    def get_materialized_resonance(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_materialized_resonance()

    def get_gains(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_gains()

    def get_router(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_router()

    def get_attack_envelopes(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_attack_envelopes()

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

    @property
    def flattened_deformations(self):
        return self.deformations.view(self.expressivity, self.n_frames)

    def _get_mapping(self, n_components: int) -> np.ndarray:
        cs = self.control_signal.data.cpu().numpy() \
            .reshape(self.control_plane_dim, self.n_frames).T
        pca = DictionaryLearning(n_components=n_components)
        pca.fit(cs)
        # this will be of shape (n_components, control_plane_dim)
        return pca.components_

    def get_hand_tracking_mapping(self) -> np.ndarray:
        mapping = self._get_mapping(n_components=21 * 3)
        print('PCA Weight Shape', mapping.shape)
        return mapping

    def get_materialized_resonance(self, layer: int) -> torch.Tensor:
        return self.network.get_materialized_resonance(layer)

    def get_gains(self, layer: int) -> torch.Tensor:
        return self.network.get_gains(layer)

    def get_router(self, layer: int) -> torch.Tensor:
        return self.network.get_router(layer)

    def get_attack_envelopes(self, layer: int) -> torch.Tensor:
        return self.network.get_attack_envelopes(layer) ** 2

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

    def random(self, use_learned_deformations: bool = False):
        # print(self.control_plane.min().item(), self.control_plane.max().item())
        rcp = torch \
            .zeros_like(self.control_plane) \
            .uniform_(
            self.control_plane.min().item(),
            self.control_plane.max().item())

        rcp = self._process_control_plane(rcp, n_to_keep=16)
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


def encode_array(arr: Union[np.ndarray, torch.Tensor], serializer: NumpySerializer) -> str:
    if isinstance(arr, torch.Tensor):
        arr = arr.data.cpu().numpy()

    return b64encode(serializer.to_bytes(arr)).decode()


def generate_param_dict(key: str, logger: Logger, model: OverfitResonanceStack) -> [dict, MetaData]:
    serializer = NumpySerializer()

    hand = model.get_hand_tracking_mapping().T
    assert hand.shape == (model.control_plane_dim, 21 * 3)

    router = model.get_router(0)
    assert router.shape == (model.control_plane_dim, model.n_resonances)

    gains = model.get_gains(0).view(-1)
    assert gains.shape == (model.n_resonances,)

    resonances = model.get_materialized_resonance(0).reshape(-1, model.n_samples)
    assert resonances.shape == (model.n_resonances * model.expressivity, model.n_samples)

    attacks = model.get_attack_envelopes(0)

    params = dict(
        gains=encode_array(gains, serializer),
        router=encode_array(router, serializer),
        resonances=encode_array(resonances, serializer),
        hand=encode_array(hand, serializer),
        attacks=encode_array(attacks, serializer),
    )
    _, meta = logger.log_json(key, params)
    print('WEIGHTS URI', meta.public_uri.geturl())
    return params, meta


def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()
    forward = mask
    backward = x
    y = backward + (forward - backward).detach()
    return y.sum()


def overfit_model():
    n_samples = 2 ** 17
    resonance_window_size = 2048
    step_size = 1024
    n_frames = n_samples // step_size

    # KLUDGE: control_plane_dim and n_resonances
    # must have the same value
    control_plane_dim = 16
    n_resonances = 16
    expressivity = 2

    target = get_one_audio_segment(n_samples)
    model = OverfitResonanceStack(
        n_layers=1,
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

    remote_collection = conjure.S3Collection('resonancemodel', is_public=True, cors_enabled=True)
    remote_logger = conjure.Logger(remote_collection)

    t, r, rand, res = conjure.loggers(
        ['target', 'recon', 'random', 'resonance'],
        'audio/wav',
        encode_audio,
        collection)

    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.data.cpu().numpy()

    c, deformations, routing, attack = conjure.loggers(
        ['control', 'deformations', 'routing', 'attack'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    serve_conjure(
        [t, r, c, rand, res, deformations, routing, attack],
        port=9999,
        n_workers=1,
        web_components_version='0.0.89')

    t(max_norm(target))

    def train():
        iteration = 0

        while True:
            optimizer.zero_grad()
            recon = model.forward()
            r(max_norm(recon))
            c(model.control_signal[0, 0])


            # x = stft(target, 2048, 256, pad=True)
            # y = stft(recon, 2048, 256, pad=True)

            x = flattened_multiband_spectrogram(target, {'xs': (64, 16)})
            y = flattened_multiband_spectrogram(recon, {'xs': (64, 16)})
            loss = torch.abs(x - y).sum()

            loss.backward()
            optimizer.step()
            print(iteration, loss.item(), model.compression_ratio(n_samples))

            deformations(model.flattened_deformations)
            routing(torch.abs(model.get_router(0)))
            attack(max_norm(model.get_attack_envelopes(0)))

            with torch.no_grad():
                rand(max_norm(model.random(use_learned_deformations=True)))
                rz = model.get_materialized_resonance(0).view(-1, n_samples)
                res(max_norm(rz[np.random.randint(0, n_resonances * expressivity - 1)]))

            iteration += 1

            if iteration > 0 and iteration % 10000 == 0:
                print('Serializing')
                generate_param_dict('resonancemodelparams', remote_logger, model)
                input('Continue?')

    train()


if __name__ == '__main__':
    overfit_model()
