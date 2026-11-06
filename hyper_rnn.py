# Listening notes: every valid --rnn / --output-mode combination
#
# --rnn tanh --output-mode complex                                     notes: gritty, poor recon.  Medium buzz.
# --rnn tanh --output-mode mag-phase                                   notes: non-viable
# --rnn tanh --output-mode time                                        notes:
# --rnn tanh --output-mode noise                                       notes: non-start, BUT, no buzz in synthetic signal
# --rnn tanh --output-mode shift                                       notes: gtritty, buzz in synthetic
# --rnn tanh --output-mode noise-shift                                 notes: gritty, poor recon.  Lots of synthetic buzz
# --rnn tanh --output-mode time-noise --noise-envelope smooth          notes:
# --rnn tanh --output-mode time-noise --noise-envelope frame           notes:
# --rnn complex --output-mode complex --complex-nonlinearity none      notes: poor recon, lots of buzz
# --rnn complex --output-mode complex --complex-nonlinearity modtanh   notes:

"""
Same idea as `parallel.py`, but in jax, and with a much simpler
"instrument": an analysis network hears a segment of audio and factors it
into

    1. the weights of a small, single-layer tanh RNN (via a hypernetwork,
       from a single, time-pooled latent vector), and
    2. a sparse control signal that drives that RNN.

The generated RNN is run with `jax.lax.scan`. Each RNN step's output is
turned into a short, windowed block of samples by one of several output
modes (see `OutputMode`), and blocks are overlap-added to produce the final
audio, so the RNN runs at `n_samples // step_size` steps, rather than once
per audio sample.
"""
import argparse
import itertools
from typing import List, Optional, Tuple, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, loggers, serve_conjure
from conjure.logger import encode_audio

from data.audioiter import AudioIterator


# the generated recurrent weights are rescaled so that their (approximate)
# spectral radius lands in (0, MAX_SPECTRAL_RADIUS), see `HyperRNN.__call__`
MAX_SPECTRAL_RADIUS = 1.2

# the summed log-mel loss is roughly 50-60x larger than the summed multiband
# loss, so the sparsity penalty needs to be scaled to match
DEFAULT_SPARSITY_LOSS_COEFF = {
    'mel': 1.0,
    'multiband': 0.01,
}

# likewise for the decay penalty; these put it at roughly 2-3% of the
# reconstruction loss at initialization.  Like the sparsity penalty, an
# all-zero control signal minimizes it, so it's kept modest.
DEFAULT_DECAY_PENALTY_COEFF = {
    'mel': 2.0,
    'multiband': 0.05,
}

# --self-supervised: weights of the decay curve (dB, summed over steps) and
# onset profile (earth mover's distance, in steps) losses.  At init, with
# default sizes, these put the decay curve loss at ~20-30%, and the onset
# loss at ~3%, of the synthetic examples' reconstruction loss.
DEFAULT_DECAY_CURVE_COEFF = {
    'mel': 1.0,
    'multiband': 0.05,
}
DEFAULT_ONSET_COEFF = {
    'mel': 50.0,
    'multiband': 2.0,
}


# ---------------------------------------------------------------------------
# signal processing helpers and losses (`stft` and the multiband
# spectrogram are jax ports of modules/stft.py and modules/decompose.py, so
# the 'multiband' loss matches `parallel.py`'s)

def hann_window(size: int) -> jnp.ndarray:
    """Periodic hann window, matching `torch.hann_window`"""
    return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(size) / size)


def stft(x: jnp.ndarray, ws: int = 512, step: int = 256, pad: bool = False) -> jnp.ndarray:
    """
    x: (n_samples,)
    returns: (n_frames, ws // 2 + 1)
    """
    n_frames = x.shape[-1] // step

    if pad:
        x = jnp.pad(x, (0, ws))

    total_frames = 1 + (x.shape[-1] - ws) // step
    frame_idx = jnp.arange(ws)[None, :] + step * jnp.arange(total_frames)[:, None]
    frames = x[frame_idx] * hann_window(ws)[None, :]
    spec = jnp.abs(jnp.fft.rfft(frames, axis=-1, norm='ortho'))
    return spec[:n_frames]


def fft_frequency_decompose(x: jnp.ndarray, min_size: int) -> List[jnp.ndarray]:
    coeffs = jnp.fft.rfft(x, norm='ortho')

    output = []
    current_size = min_size

    while current_size <= x.shape[-1]:
        sl = coeffs[:current_size // 2 + 1]
        if current_size > min_size:
            mask = jnp.zeros(sl.shape[0]).at[current_size // 4:].set(1)
            sl = sl * mask

        output.append(jnp.fft.irfft(sl, n=current_size, norm='ortho'))
        current_size *= 2

    return output


def flattened_multiband_spectrogram(
        x: jnp.ndarray,
        ws: int = 64,
        step: int = 16,
        smallest_band_size: int = 512) -> jnp.ndarray:

    bands = fft_frequency_decompose(x, smallest_band_size)
    return jnp.concatenate([stft(b, ws, step, pad=True).reshape(-1) for b in bands])


def build_mel_filterbank(samplerate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(samplerate / 2.0), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / samplerate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def make_log_mel_spectrogram(
        samplerate: int,
        n_fft: int = 2048,
        hop_length: int = 256,
        n_mels: int = 128,
        epsilon: float = 1e-5):

    filterbank = jnp.asarray(build_mel_filterbank(samplerate, n_fft, n_mels))

    def log_mel_spectrogram(x: jnp.ndarray) -> jnp.ndarray:
        """
        x: (n_samples,)
        returns: (n_frames, n_mels)
        """
        power = stft(x, ws=n_fft, step=hop_length, pad=True) ** 2
        return jnp.log(power @ filterbank.T + epsilon)

    return log_mel_spectrogram


def make_loss_func(name: str, samplerate: int):
    """
    'mel' compares log-mel spectrograms; 'multiband' compares the
    multi-band/multi-samplerate spectrograms `parallel.py` uses.
    """
    features = {
        'mel': lambda: make_log_mel_spectrogram(samplerate),
        'multiband': lambda: flattened_multiband_spectrogram,
    }[name]()

    def loss_func(target: jnp.ndarray, recon: jnp.ndarray) -> jnp.ndarray:
        """
        target: (batch, n_samples)
        recon: (batch, n_samples)
        """
        a = jax.vmap(features)(target)
        b = jax.vmap(features)(recon)
        return jnp.abs(a - b).sum()

    return loss_func


def sparsify(x: jnp.ndarray, n_to_keep: int) -> jnp.ndarray:
    """
    Keep only the `n_to_keep` largest elements of `x`, zeroing the rest.
    Gradients flow only through the elements that were kept.
    """
    flat = x.reshape(-1)
    values, indices = jax.lax.top_k(flat, n_to_keep)
    return jnp.zeros_like(flat).at[indices].set(values).reshape(x.shape)


def l0_norm(x: jnp.ndarray) -> jnp.ndarray:
    """
    Count of non-zero elements in the forward pass; acts as the sum of `x`
    (an L1-like penalty, since `x` is non-negative) in the backward pass.
    """
    mask = (x > 0).astype(x.dtype)
    return (x + jax.lax.stop_gradient(mask - x)).sum()


def max_norm(x: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return x / (np.abs(x).max() + epsilon)


# ---------------------------------------------------------------------------
# analysis network (an equinox port of modules/anticausal.py)

class AntiCausalBlock(eqx.Module):
    conv: eqx.nn.Conv1d
    gate: eqx.nn.Conv1d
    tanh_weight: jnp.ndarray
    sigmoid_weight: jnp.ndarray

    def __init__(self, channels: int, kernel_size: int, dilation: int, key):
        k1, k2 = jax.random.split(key)
        # pad only on the right, so each frame can see the future, but not
        # the past
        padding = ((0, (kernel_size * dilation) // 2),)
        self.conv = eqx.nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding, key=k1)
        self.gate = eqx.nn.Conv1d(
            channels, channels, kernel_size, dilation=dilation, padding=padding, key=k2)
        self.tanh_weight = jnp.array(0.5)
        self.sigmoid_weight = jnp.array(0.5)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        a = jnp.tanh(self.conv(x) * self.tanh_weight)
        b = jax.nn.sigmoid(self.gate(x) * self.sigmoid_weight)
        return (a * b) + (x * (1 / 2 ** 0.5))


class Analysis(eqx.Module):
    proj: eqx.nn.Conv1d
    blocks: list
    ff: eqx.nn.Conv1d
    frame_size: int = eqx.field(static=True)
    window_size: int = eqx.field(static=True)

    def __init__(self, frame_size: int, channels: int, dilations: Tuple[int, ...], key):
        self.frame_size = frame_size
        self.window_size = frame_size * 4
        n_coeffs = self.window_size // 2 + 1

        keys = jax.random.split(key, len(dilations) + 2)
        self.proj = eqx.nn.Conv1d(n_coeffs, channels, 1, key=keys[0])
        self.blocks = [
            AntiCausalBlock(channels, kernel_size=2, dilation=d, key=k)
            for d, k in zip(dilations, keys[1:-1])
        ]
        self.ff = eqx.nn.Conv1d(channels, channels, 1, key=keys[-1])

    def __call__(self, audio: jnp.ndarray) -> jnp.ndarray:
        """
        audio: (n_samples,)
        returns: (channels, n_frames)
        """
        spec = stft(audio, ws=self.window_size, step=self.frame_size, pad=True).T
        x = self.proj(spec)

        output = jnp.zeros_like(x)
        for block in self.blocks:
            x = block(x)
            output = output + x
        return self.ff(output / len(self.blocks))


class ControlSignalCreator(eqx.Module):
    conv: eqx.nn.Conv1d
    n_to_keep: int = eqx.field(static=True)
    do_sparsify: bool = eqx.field(static=True)
    subtract_mean: bool = eqx.field(static=True)

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            n_to_keep: int,
            do_sparsify: bool,
            key,
            subtract_mean: bool = True):

        self.conv = eqx.nn.Conv1d(
            in_channels, out_channels, kernel_size=8, padding=((0, 7),), key=key)
        self.n_to_keep = n_to_keep
        self.do_sparsify = do_sparsify
        self.subtract_mean = subtract_mean

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.subtract_mean:
            x = jnp.abs(self.conv(x))
            x = jax.nn.relu(x - jnp.mean(x))
        else:
            x = jax.nn.relu(self.conv(x))

        if self.do_sparsify:
            x = sparsify(x, self.n_to_keep)
        return x


class ParameterGenerator(eqx.Module):
    linear: eqx.nn.Linear
    scale: jnp.ndarray

    def __init__(self, in_features: int, out_features: int, key):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.scale = jnp.array(1.0)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.linear(x) * self.scale


# ---------------------------------------------------------------------------
# the generated instrument: a single-layer tanh RNN

# Output modes: how each RNN step's output vector becomes a block of
# `2 * step_size` audio samples.  Every mode's blocks are windowed and
# overlap-added with a hop of `step_size` (see `execute_rnn`), so a new mode
# only needs to define its output size, its window, and how to turn one
# step's output into one (un-windowed) block.  A mode that needs the whole
# sequence of outputs at once (e.g., to accumulate something over time) can
# override `blocks` instead.

class OutputMode:
    # if True, `block` receives a fresh block of uniform [-1, 1] noise for
    # every step; otherwise it receives None
    needs_noise: bool = False

    def output_size(self, step_size: int) -> int:
        raise NotImplementedError()

    def window(self, block_size: int) -> jnp.ndarray:
        # hann windows sum to one at 50% overlap
        return hann_window(block_size)

    def block(self, out: jnp.ndarray, noise: jnp.ndarray, block_size: int) -> jnp.ndarray:
        raise NotImplementedError()

    def blocks(self, outs: jnp.ndarray, noise: jnp.ndarray, block_size: int) -> jnp.ndarray:
        """
        outs: (n_steps, output_size), every step's output
        noise: (n_steps, block_size), or None
        returns: (n_steps, block_size), un-windowed
        """
        return jax.vmap(self.block, in_axes=(0, None if noise is None else 0, None))(
            outs, noise, block_size)


class TimeDomainOutput(OutputMode):
    """Each step outputs a block of samples directly"""

    def output_size(self, step_size: int) -> int:
        return step_size * 2

    def block(self, out, noise, block_size):
        return out


class ComplexOutput(OutputMode):
    """
    Each step outputs the real and imaginary parts of a complex spectrum
    frame, and audio is materialized via a classic inverse STFT: the inverse
    rfft of each frame is multiplied by a sqrt-hann synthesis window, and
    frames are overlap-added.  The squared sqrt-hann window sums to one at
    50% overlap, so no window normalization is needed.
    """

    def output_size(self, step_size: int) -> int:
        return (step_size + 1) * 2

    def window(self, block_size):
        return jnp.sqrt(hann_window(block_size))

    def block(self, out, noise, block_size):
        n_coeffs = out.shape[-1] // 2
        spec = out[:n_coeffs] + 1j * out[n_coeffs:]
        return jnp.fft.irfft(spec, n=block_size, norm='ortho')


class NoiseOutput(OutputMode):
    """
    Each step outputs a single scalar, which scales a block of uniform
    [-1, 1] noise, so output is inherently noisy, but its amplitude envelope
    is controlled by the RNN.

    The noise is zero-mean on purpose: with [0, 1] noise, half the scalar
    would pass straight through as a deterministic DC/envelope signal (75% of
    the output energy).  Since the noise is symmetric, the scalar's sign has
    no effect; only its magnitude does.
    """
    needs_noise = True

    def output_size(self, step_size: int) -> int:
        return 1

    def block(self, out, noise, block_size):
        return out[0] * noise


def fft_shift(x: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
    """
    Circularly shift `x` (along its last axis) forward by `shift` samples,
    which may be fractional, via a linear phase ramp in the frequency
    domain.  An integer shift matches `jnp.roll`; fractional shifts are
    band-limited interpolations, so the result is differentiable with
    respect to `shift`.
    """
    n = x.shape[-1]
    spec = jnp.fft.rfft(x)
    k = jnp.arange(spec.shape[-1])
    spec = spec * jnp.exp(-2j * jnp.pi * k * shift / n)
    return jnp.fft.irfft(spec, n=n)


class ShiftOutput(OutputMode):
    """
    Each step outputs a block of samples, plus a scalar that circularly
    shifts ("rolls") that block forward or backward before windowing.  The
    scalar is a fraction of the block size, so the shift is periodic in it:
    0.5 and -0.5 are the same (half-block) shift.
    """

    def output_size(self, step_size: int) -> int:
        return step_size * 2 + 1

    def block(self, out, noise, block_size):
        samples, shift = out[:block_size], out[block_size]
        return fft_shift(samples, shift * block_size)


class NoiseShiftOutput(OutputMode):
    """
    The sum of 'shift' and 'noise': each step outputs a block of samples and
    a scalar that circularly shifts it (see `ShiftOutput`), plus a scalar
    that scales a block of uniform [-1, 1] noise (see `NoiseOutput`), giving
    a deterministic and a noisy component, each controlled by the RNN.
    """
    needs_noise = True

    def __init__(self):
        self.shift = ShiftOutput()
        self.noise = NoiseOutput()

    def output_size(self, step_size: int) -> int:
        return self.shift.output_size(step_size) + self.noise.output_size(step_size)

    def block(self, out, noise, block_size):
        n_shift = self.shift.output_size(block_size // 2)
        return self.shift.block(out[:n_shift], None, block_size) \
            + self.noise.block(out[n_shift:], noise, block_size)


class TimeNoiseOutput(OutputMode):
    """
    The sum of 'time' and 'noise': each step outputs a block of samples (a
    deterministic component), plus a scalar that scales a block of uniform
    [-1, 1] noise (a stochastic component).  Noisy material (breath, bow
    noise, reverb tails) is hard for a deterministic RNN to produce without
    repeating similar blocks from step to step, which is heard as a buzz at
    the step rate; this gives that material somewhere else to go.

    envelope:
        'frame'   the noise scalar is used as-is, one per step, so the noise
                  level can change every step
        'smooth'  the noise scalars' magnitudes (their sign is meaningless,
                  since the noise is symmetric) are low-passed along time, with
                  a gaussian roll-off at `cutoff_hz`, so the noise level can't
                  flutter at the step rate, which would itself be heard as
                  roughness
    """
    needs_noise = True

    def __init__(self, envelope: str = 'smooth', cutoff_hz: float = 20., samplerate: int = 22050):
        self.envelope = envelope
        self.cutoff_hz = cutoff_hz
        self.samplerate = samplerate

    def output_size(self, step_size: int) -> int:
        return step_size * 2 + 1

    def noise_envelope(self, scalars: jnp.ndarray, step_size: int) -> jnp.ndarray:
        """
        scalars: (n_steps,), one noise scalar per step
        returns: (n_steps,)
        """
        if self.envelope == 'frame':
            return scalars

        n_steps = scalars.shape[0]
        step_rate = self.samplerate / step_size
        # zero-padded, so the end of the sequence doesn't wrap around into
        # its beginning
        size = n_steps * 2
        spec = jnp.fft.rfft(jnp.abs(scalars), n=size)
        freqs = jnp.fft.rfftfreq(size, d=1 / step_rate)
        # -3dB at the cutoff
        sigma = self.cutoff_hz / jnp.sqrt(jnp.log(2.))
        gain = jnp.exp(-0.5 * (freqs / sigma) ** 2)
        return jnp.fft.irfft(spec * gain, n=size)[:n_steps]

    def blocks(self, outs, noise, block_size):
        samples, scalars = outs[:, :block_size], outs[:, block_size]
        envelope = self.noise_envelope(scalars, block_size // 2)
        return samples + envelope[:, None] * noise


class MagnitudePhaseOutput(OutputMode):
    """
    Each step outputs a spectral magnitude per STFT bin and, per bin, a
    fractional deviation from that bin's expected phase advance.  Phase is
    the cumulative sum of the advances over time, and audio is materialized
    via the same inverse STFT as `ComplexOutput`.

    At 50% overlap (hop = step_size, window = 2 * step_size), a sinusoid at
    bin k's center frequency advances by pi * k radians per frame, so each
    frame's advance is pi * k * (1 + deviation).  A deviation of 0 continues
    every bin's sinusoid coherently from frame to frame; a sinusoid that
    lies between bins k and k + 1, at k + f, needs a deviation of f / k.
    Note that the same deviation moves high bins much further than low ones
    (it's relative to pi * k), and bin 0 (DC) has no expected advance, so
    its deviation has no effect.

    Magnitudes are the absolute value of the head's output.  Every bin's
    phase starts at 0.
    """

    def output_size(self, step_size: int) -> int:
        return (step_size + 1) * 2

    def window(self, block_size):
        return jnp.sqrt(hann_window(block_size))

    def blocks(self, outs, noise, block_size):
        n_coeffs = outs.shape[-1] // 2
        magnitude = jnp.abs(outs[:, :n_coeffs])
        deviation = outs[:, n_coeffs:]

        expected_advance = jnp.pi * jnp.arange(n_coeffs)
        phase = jnp.cumsum(expected_advance * (1 + deviation), axis=0)

        spec = magnitude * jnp.exp(1j * phase)
        return jnp.fft.irfft(spec, n=block_size, norm='ortho', axis=-1)


OUTPUT_MODES = {
    'complex': ComplexOutput(),
    'mag-phase': MagnitudePhaseOutput(),
    'time': TimeDomainOutput(),
    'noise': NoiseOutput(),
    'shift': ShiftOutput(),
    'noise-shift': NoiseShiftOutput(),
    # configured from the command line in `main` (--noise-envelope,
    # --noise-cutoff-hz, --samplerate)
    'time-noise': TimeNoiseOutput(),
}


class RNNWeights(eqx.Module):
    w_in: jnp.ndarray   # (control_channels, hidden)
    w_hh: jnp.ndarray   # (hidden, hidden)
    bias: jnp.ndarray   # (hidden,)
    w_out: jnp.ndarray  # (hidden, OUTPUT_MODES[output_mode].output_size(step_size))
    output_mode: str = eqx.field(static=True)
    step_size: int = eqx.field(static=True)


def execute_rnn(weights: Union[RNNWeights, 'ComplexRNNWeights'], control: jnp.ndarray, key=None) -> jnp.ndarray:
    """
    control: (control_channels, n_steps)
    key: required only for output modes that need noise
    returns: (n_steps * step_size,)

    Each step emits a windowed block of `2 * step_size` samples (see
    `OutputMode`), and blocks are overlap-added with a hop of `step_size`, so
    the output has no discontinuities at block boundaries.

    Complex RNN weights are executed by `execute_complex_rnn`.
    """
    if isinstance(weights, ComplexRNNWeights):
        return execute_complex_rnn(weights, control)

    mode = OUTPUT_MODES[weights.output_mode]
    hidden_dim = weights.w_hh.shape[0]
    n_steps = control.shape[-1]
    step_size = weights.step_size
    block_size = step_size * 2
    window = mode.window(block_size)

    if mode.needs_noise:
        if key is None:
            raise ValueError(f"output mode '{weights.output_mode}' requires a key")
        noise = jax.random.uniform(key, (n_steps, block_size), minval=-1, maxval=1)
    else:
        noise = None

    def step(hidden, control_t):
        hidden = jnp.tanh(control_t @ weights.w_in + hidden @ weights.w_hh + weights.bias)
        return hidden, hidden @ weights.w_out

    # the scan collects every step's raw output, and the output mode turns the
    # whole sequence into blocks, so a mode can accumulate over time
    _, outs = jax.lax.scan(step, jnp.zeros(hidden_dim), control.T)
    blocks = mode.blocks(outs, noise, block_size) * window
    return overlap_add(blocks, step_size)


def overlap_add(blocks: jnp.ndarray, step_size: int) -> jnp.ndarray:
    """
    blocks: (n_steps, 2 * step_size), already windowed
    returns: (n_steps * step_size,)

    Output step t is the first half of block t plus the second half of
    block t - 1.
    """
    first = blocks[:, :step_size]
    second = jnp.pad(blocks[:-1, step_size:], ((1, 0), (0, 0)))
    return (first + second).reshape(-1)


# ---------------------------------------------------------------------------
# the generated instrument, alternative: a diagonal, complex-valued, linear
# RNN in the STFT domain

# The complex RNN's per-unit decay (magnitude of its recurrent weight, per
# frame) is exp(-exp(nu - COMPLEX_DECAY_OFFSET)), where nu is generated by the
# hypernetwork.  It's always strictly less than one, so the linear recurrence
# can't blow up, and at nu = 0 it's ~0.982 per frame (a decay to 1% over
# ~250 frames, ~1.5s at the default step size).
COMPLEX_DECAY_OFFSET = 4.0


class ComplexRNNWeights(eqx.Module):
    w_in: jnp.ndarray       # complex, (control_channels, hidden)
    recurrent: jnp.ndarray  # complex, (hidden,): each unit's (diagonal) recurrent weight
    w_out: jnp.ndarray      # complex, (hidden, step_size + 1): each unit's contribution to each STFT bin
    step_size: int = eqx.field(static=True)
    nonlinearity: str = eqx.field(static=True)


def execute_complex_rnn(weights: ComplexRNNWeights, control: jnp.ndarray) -> jnp.ndarray:
    """
    control: (control_channels, n_steps), real
    returns: (n_steps * step_size,)

    At each step, every hidden unit's complex state is multiplied by its own
    complex recurrent weight (a decay and a phase advance, i.e., a
    convolution) and driven by a complex projection of the control signal:

        hidden = recurrent * hidden + control_t @ w_in

    then mapped to a complex STFT frame (hidden @ w_out), which becomes audio
    via the same inverse STFT as `ComplexOutput`.  Each unit is effectively a
    resonator: its recurrent weight's magnitude sets how fast it decays, and
    its angle how far its phase advances per frame.  A sinusoid at frequency
    f advances 2 * pi * f * step_size / samplerate per frame, so the angle
    (with w_out choosing the bins nearest f) can place a unit at any
    frequency.

    There's no bias (a constant drive would sound with no control input),
    and, by default, no nonlinearity, so the recurrence is entirely linear.
    'modtanh' squashes each unit's magnitude with tanh, keeping its phase
    (complex tanh itself has poles, so isn't safe to use here).
    """
    mode = ComplexOutput()
    step_size = weights.step_size
    block_size = step_size * 2
    window = mode.window(block_size)
    hidden_dim = weights.recurrent.shape[0]

    def step(hidden, control_t):
        hidden = weights.recurrent * hidden + control_t @ weights.w_in
        if weights.nonlinearity == 'modtanh':
            magnitude = jnp.sqrt(hidden.real ** 2 + hidden.imag ** 2 + 1e-12)
            hidden = hidden * (jnp.tanh(magnitude) / magnitude)
        spec = hidden @ weights.w_out
        block = jnp.fft.irfft(spec, n=block_size, norm='ortho')
        return hidden, block * window

    _, blocks = jax.lax.scan(step, jnp.zeros(hidden_dim, dtype=jnp.complex64), control.T)
    return overlap_add(blocks, step_size)


class HyperRNN(eqx.Module):
    analysis: Analysis
    latents: ControlSignalCreator
    latents_attn: ControlSignalCreator
    control: ControlSignalCreator
    to_w_in: ParameterGenerator
    to_w_hh: ParameterGenerator
    to_radius: Optional[ParameterGenerator]  # 'tanh' RNN only
    to_bias: Optional[ParameterGenerator]    # 'tanh' RNN only
    to_w_out: ParameterGenerator

    hidden_dim: int = eqx.field(static=True)
    control_channels: int = eqx.field(static=True)
    step_size: int = eqx.field(static=True)
    output_mode: str = eqx.field(static=True)
    rnn: str = eqx.field(static=True)
    complex_nonlinearity: str = eqx.field(static=True)

    def __init__(
            self,
            step_size: int,
            channels: int,
            control_channels: int,
            hidden_dim: int,
            n_to_keep: int,
            key,
            hard_sparsify: bool = False,
            output_mode: str = 'complex',
            rnn: str = 'tanh',
            complex_nonlinearity: str = 'none'):

        if rnn == 'complex' and output_mode != 'complex':
            raise ValueError("the complex RNN always outputs via an inverse STFT ('complex' output mode)")

        self.hidden_dim = hidden_dim
        self.control_channels = control_channels
        self.step_size = step_size
        self.output_mode = output_mode
        self.rnn = rnn
        self.complex_nonlinearity = complex_nonlinearity

        keys = jax.random.split(key, 9)

        self.analysis = Analysis(
            step_size, channels, dilations=(1, 2, 4, 8, 16, 32, 1), key=keys[0])

        self.latents = ControlSignalCreator(
            channels, channels, n_to_keep, do_sparsify=False, key=keys[1])
        self.latents_attn = ControlSignalCreator(
            channels, channels, n_to_keep, do_sparsify=False, key=keys[2])
        # When sparsity is encouraged via a loss, rather than enforced via
        # top-k, the control signal can't be mean-subtracted: an L1-style
        # penalty on relu(x - mean(x)) pushes *inactive* elements up (to raise
        # the mean), which flattens the distribution and makes the control
        # signal denser, rather than sparser.
        self.control = ControlSignalCreator(
            channels,
            control_channels,
            n_to_keep,
            do_sparsify=hard_sparsify,
            key=keys[3],
            subtract_mean=hard_sparsify)

        if rnn == 'complex':
            # complex weights are generated as (real, imaginary) pairs; the
            # recurrent weights are diagonal, generated as a (decay, angle)
            # pair per unit, see `rnn_weights`
            self.to_w_in = ParameterGenerator(channels, control_channels * hidden_dim * 2, keys[4])
            self.to_w_hh = ParameterGenerator(channels, hidden_dim * 2, keys[5])
            self.to_radius = None
            self.to_bias = None
        else:
            self.to_w_in = ParameterGenerator(channels, control_channels * hidden_dim, keys[4])
            self.to_w_hh = ParameterGenerator(channels, hidden_dim * hidden_dim, keys[5])
            self.to_radius = ParameterGenerator(channels, 1, keys[6])
            self.to_bias = ParameterGenerator(channels, hidden_dim, keys[7])
        # for the complex RNN, output_size is (step_size + 1) * 2: the real
        # and imaginary parts of each unit's contribution to each STFT bin
        self.to_w_out = ParameterGenerator(channels, hidden_dim * self.output_size, keys[8])

    @property
    def output_size(self) -> int:
        """Size of each RNN step's output, see `RNNWeights.w_out`"""
        return OUTPUT_MODES[self.output_mode].output_size(self.step_size)

    def rnn_weights(self, latent: jnp.ndarray) -> Union[RNNWeights, ComplexRNNWeights]:
        h = self.hidden_dim

        if self.rnn == 'complex':
            w_in = self.to_w_in(latent).reshape(2, self.control_channels, h)
            w_out = self.to_w_out(latent).reshape(h, 2, self.output_size // 2)
            nu, angle = self.to_w_hh(latent).reshape(2, h)
            # decay strictly inside the unit circle, so the linear recurrence
            # is stable; angle (phase advance per frame) in (-pi, pi)
            decay = jnp.exp(-jnp.exp(nu - COMPLEX_DECAY_OFFSET))
            return ComplexRNNWeights(
                w_in=w_in[0] + 1j * w_in[1],
                recurrent=decay * jnp.exp(1j * jnp.pi * jnp.tanh(angle)),
                w_out=w_out[:, 0] + 1j * w_out[:, 1],
                step_size=self.step_size,
                nonlinearity=self.complex_nonlinearity,
            )

        # For a random (h, h) matrix, frobenius_norm / sqrt(h) is a decent
        # estimate of its spectral radius.  Normalize by that estimate and
        # let the hypernetwork choose the radius directly, within a bounded
        # range, to keep the generated recurrence from exploding.
        w_hh = self.to_w_hh(latent).reshape(h, h)
        estimated_radius = jnp.linalg.norm(w_hh) / jnp.sqrt(h)
        radius = jax.nn.sigmoid(self.to_radius(latent)[0]) * MAX_SPECTRAL_RADIUS
        w_hh = w_hh * (radius / (estimated_radius + 1e-8))

        return RNNWeights(
            w_in=self.to_w_in(latent).reshape(self.control_channels, h),
            w_hh=w_hh,
            bias=self.to_bias(latent),
            w_out=self.to_w_out(latent).reshape(h, self.output_size),
            output_mode=self.output_mode,
            step_size=self.step_size,
        )

    def factor(self, audio: jnp.ndarray) -> Tuple[Union[RNNWeights, ComplexRNNWeights], jnp.ndarray]:
        """
        audio: (n_samples,)
        returns: the generated RNN's weights and the sparse control signal
            (control_channels, n_steps) that drives it
        """
        x = self.analysis(audio)

        # a single latent vector, describing the "instrument", pooled over
        # the whole segment
        latent = jnp.mean(self.latents(x) * self.latents_attn(x), axis=-1)

        return self.rnn_weights(latent), self.control(x)

    def __call__(self, audio: jnp.ndarray, key=None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        audio: (n_samples,)
        key: required only for output modes that need noise
        returns: sparse control signal (control_channels, n_steps) and
            reconstructed audio (n_samples,)
        """
        weights, control = self.factor(audio)
        return control, execute_rnn(weights, control, key)

    def random_forward(self, audio: jnp.ndarray, key, n_events: int) -> jnp.ndarray:
        """
        Play the instrument inferred from `audio` with a random, sparse
        control signal of `n_events` non-zero elements, instead of the
        inferred one, to hear how well the instrument and its performance
        were factored apart.

        audio: (n_samples,)
        returns: (n_samples,)
        """
        weights, control = self.factor(audio)
        control_key, render_key = jax.random.split(key)
        return execute_rnn(weights, random_control(control_key, control, n_events), render_key)


def random_control(key, control: jnp.ndarray, n_events: int) -> jnp.ndarray:
    """
    A random, sparse control signal with the same shape as `control`, and
    `n_events` non-zero elements.  Amplitudes are on the same scale as
    `control`'s (squared, so most events are quieter than the loudest one).
    """
    scale = jax.lax.stop_gradient(jnp.max(control))
    x = (jax.random.uniform(key, control.shape) ** 2) * scale
    return sparsify(x, n_events)


def decay_penalty(control: jnp.ndarray, relative_epsilon: float = 0.01) -> jnp.ndarray:
    """
    Discourage the control signal from shaping decays itself (gradual,
    step-by-step decreases along a channel), which should be left to the
    instrument's dynamics.

    Gradients reach the control signal over a one-step path, but reach the
    generated recurrent weights only through backpropagation over many
    steps, so without this, the control signal tends to take over shaping
    each note's decay.

    Only decreases are penalized, so onsets and swells are free, and the
    penalty is concave (log-shaped), so a single large drop (a release)
    costs far less than the same drop spread over many small steps (a
    decay).  Sustained, flat control (a bowed string, a wind instrument) and
    impulses are cheap.  epsilon is relative to the control signal's
    maximum, so the penalty can't be avoided by rescaling the control signal.

    control: (control_channels, n_steps)
    """
    epsilon = jax.lax.stop_gradient(jnp.max(control)) * relative_epsilon + 1e-8
    drops = jax.nn.relu(control[:, :-1] - control[:, 1:])
    return jnp.log1p(drops / epsilon).sum()


def energy_violation(control: jnp.ndarray, audio: jnp.ndarray) -> jnp.ndarray:
    """
    "Conservation of energy": at every step, the total energy output so far
    should be no greater than the total energy input via the control signal
    so far.  Only violations are penalized; energy may be stored (and
    released later, or never), so output energy can lag input energy
    arbitrarily.

    control: (control_channels, n_steps)
    audio: (n_steps * step_size,)
    """
    n_steps = control.shape[-1]
    input_energy = jnp.cumsum(jnp.sum(control ** 2, axis=0))
    output_energy = jnp.cumsum(jnp.sum(audio.reshape(n_steps, -1) ** 2, axis=-1))
    return jax.nn.relu(output_energy - input_energy).sum()


# ---------------------------------------------------------------------------
# self-supervised synthetic examples: known instruments (RNN weights) and
# known control signals render known audio, which is then factored by the
# model like any real example.  For these, the estimated instrument and
# control signal can also be compared with the known ones.
#
# Comparing raw weights would be ill-posed: permuting hidden units (or any
# change of basis), flipping a unit's sign, permuting control channels, or
# trading scale between the control signal and `w_in` all leave the audio
# unchanged.  So instead, instruments are compared by their decay curves, and
# control signals by their (channel-summed, normalized) onset profiles, which
# are invariant to all of these.

def random_instrument(
        key,
        control_channels: int,
        hidden_dim: int,
        output_mode: str,
        step_size: int,
        samplerate: int,
        min_t60: float,
        max_t60: float,
        input_scale: float = 0.3) -> RNNWeights:
    """
    A known instrument whose decay times are chosen, not left to chance.

    For small activations, tanh is nearly linear, so the recurrence is
    roughly `h <- W h + input`, and how long the instrument rings is set by
    the eigenvalues of W.  A random matrix's eigenvalues fill a disc roughly
    uniformly, so most of its modes die quickly, however its spectral radius
    is scaled.  Instead, W = Q B Q^T, where B is block-diagonal with 2x2
    rotate-and-shrink blocks r * [[cos t, -sin t], [sin t, cos t]] (each a
    mode decaying by r and rotating by t per step), and Q is a random
    orthogonal matrix that hides the block structure.  Each mode's 60dB decay
    time (T60) is log-uniform in [min_t60, max_t60] seconds, so
    r = 10 ** (-3 / (T60 * steps_per_second)).  W is normal, so there's no
    transient growth before the decay.

    There's no bias (a bias moves the resting point to where tanh is
    flatter, which shortens every decay), and `w_in` is scaled so that
    control impulses keep the hidden state in tanh's near-linear range.
    """
    k_t60, k_angle, k_q, k_in, k_out, k_extra = jax.random.split(key, 6)
    steps_per_second = samplerate / step_size
    n_blocks = hidden_dim // 2

    def decays(k, n):
        t60 = jnp.exp(jax.random.uniform(k, (n,), minval=np.log(min_t60), maxval=np.log(max_t60)))
        return 10 ** (-3 / (t60 * steps_per_second))

    r = decays(k_t60, n_blocks)
    angle = jax.random.uniform(k_angle, (n_blocks,), minval=0, maxval=jnp.pi)
    i = jnp.arange(n_blocks) * 2
    b = jnp.zeros((hidden_dim, hidden_dim))
    b = b.at[i, i].set(r * jnp.cos(angle)).at[i, i + 1].set(-r * jnp.sin(angle))
    b = b.at[i + 1, i].set(r * jnp.sin(angle)).at[i + 1, i + 1].set(r * jnp.cos(angle))
    if hidden_dim % 2:
        # one leftover, non-oscillating mode
        b = b.at[-1, -1].set(decays(k_extra, 1)[0])

    q, upper = jnp.linalg.qr(jax.random.normal(k_q, (hidden_dim, hidden_dim)))
    q = q * jnp.sign(jnp.diag(upper))

    output_size = OUTPUT_MODES[output_mode].output_size(step_size)
    return RNNWeights(
        w_in=jax.random.normal(k_in, (control_channels, hidden_dim)) * input_scale,
        w_hh=q @ b @ q.T,
        bias=jnp.zeros(hidden_dim),
        w_out=jax.random.normal(k_out, (hidden_dim, output_size)) / jnp.sqrt(hidden_dim),
        output_mode=output_mode,
        step_size=step_size,
    )


def synthetic_control(key, control_channels: int, n_steps: int, n_events: int) -> jnp.ndarray:
    """
    A known control signal of `n_events` short impulses, at random channels
    and steps, with amplitudes in [0.1, 1], so that anything long in the
    rendered audio has to come from the instrument.  Impulses land in the
    first 3/4 of the segment, leaving room for their tails.
    """
    k_channel, k_step, k_amp = jax.random.split(key, 3)
    channels = jax.random.randint(k_channel, (n_events,), 0, control_channels)
    steps = jax.random.randint(k_step, (n_events,), 0, (n_steps * 3) // 4)
    amplitudes = jax.random.uniform(k_amp, (n_events,), minval=0.1, maxval=1.)
    return jnp.zeros((control_channels, n_steps)).at[channels, steps].add(amplitudes)


def decay_curve(weights: RNNWeights, n_steps: int, key, probe_amplitude: float, floor_db: float = -60.):
    """
    An instrument's energy decay curve, in dB relative to its total energy:
    the instrument is struck once, on every control channel at once (so the
    curve doesn't depend on how control channels are ordered), at step 0,
    and left to ring.  The curve is the backward-integrated (Schroeder)
    energy of each step's output, so it falls monotonically, from 0dB, with
    a floor at `floor_db`.  The probe is small, to keep the instrument in its
    near-linear range, where the curve's shape doesn't depend on how scale
    is split between the control signal and `w_in`.

    returns: (n_steps,)
    """
    control_channels = weights.w_in.shape[0]
    probe = jnp.zeros((control_channels, n_steps)).at[:, 0].set(probe_amplitude)
    audio = execute_rnn(weights, probe, key)
    energy = jnp.sum(audio.reshape(n_steps, -1) ** 2, axis=-1)
    remaining = jnp.cumsum(energy[::-1])[::-1]
    return 10 * jnp.log10(remaining / (remaining[0] + 1e-12) + 10 ** (floor_db / 10))


def decay_curve_loss(known: jnp.ndarray, estimated: jnp.ndarray) -> jnp.ndarray:
    """
    Summed absolute difference between two decay curves, in dB per step, so
    an estimated instrument that dies too quickly (leaving the control signal
    to fake the tail) is penalized, however good its reconstruction was.
    """
    return jnp.abs(known - estimated).sum()


def onset_profile_loss(known: jnp.ndarray, estimated: jnp.ndarray) -> jnp.ndarray:
    """
    Compare two control signals by *when* they strike, and how hard
    relative to their other strikes: each is summed across channels and
    normalized to sum to one, and their cumulative profiles are compared,
    i.e., the 1-D earth mover's distance, in steps.  Unlike comparing the
    profiles directly, this still gives a useful gradient when known and
    estimated strikes don't overlap in time.  A control signal that sustains
    energy where the known one only struck is penalized, too.

    known, estimated: (control_channels, n_steps)
    """
    def cumulative(control):
        profile = jnp.abs(control).sum(axis=0)
        return jnp.cumsum(profile / (profile.sum() + 1e-12))

    return jnp.abs(cumulative(known) - cumulative(estimated)).sum()


def t60_from_decay_curve(curve: np.ndarray, steps_per_second: float) -> float:
    """
    Rough T60, in seconds, extrapolated from the curve's first fall to -20dB
    (i.e., 3x a T20-style estimate), for monitoring only.
    """
    below = np.nonzero(curve <= -20.)[0]
    if len(below) == 0:
        return float('inf')
    return 3 * below[0] / steps_per_second


# ---------------------------------------------------------------------------
# training

def batch_stream(mode: str, batch_size: int, n_samples: int, samplerate: int):
    if mode == 'overfit':
        target = next(iter(AudioIterator(
            batch_size=1,
            n_samples=n_samples,
            samplerate=samplerate,
            normalize=True,
            overfit=True,
            as_torch=False)))
        target = jnp.asarray(target.reshape(1, n_samples))
        while True:
            yield target
    else:
        stream = AudioIterator(
            batch_size=batch_size,
            n_samples=n_samples,
            samplerate=samplerate,
            normalize=True,
            as_torch=False)
        for batch in stream:
            yield jnp.asarray(batch.reshape(-1, n_samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'overfit'], required=True)
    parser.add_argument('--batch-size', type=int, default=4, help='ignored in overfit mode')
    parser.add_argument('--n-samples', type=int, default=2 ** 17)
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--step-size', type=int, default=128,
                        help='audio samples per RNN step; also the analysis frame size')
    parser.add_argument('--channels', type=int, default=128, help='analysis network channels')
    parser.add_argument('--control-channels', type=int, default=32)
    parser.add_argument('--hidden-dim', type=int, default=128, help='generated RNN hidden size')
    parser.add_argument('--rnn', choices=['tanh', 'complex'], default='tanh',
                        help="'tanh' is a single-layer tanh RNN with a full recurrent matrix; "
                             "'complex' is a linear RNN with complex-valued weights and a diagonal "
                             "recurrence (each unit a decaying, phase-advancing resonator, so every "
                             "recurrent step is a convolution), whose units map to complex STFT "
                             "frames, materialized via an inverse STFT. 'complex' requires "
                             "--output-mode complex (the default)")
    parser.add_argument('--complex-nonlinearity', choices=['none', 'modtanh'], default='none',
                        help="'complex' RNN only: 'none' keeps the recurrence linear; 'modtanh' "
                             "squashes each unit's magnitude with tanh, keeping its phase")
    parser.add_argument('--output-mode', choices=list(OUTPUT_MODES), default='complex',
                        help="how each RNN step's output becomes audio: 'complex' outputs a "
                             "complex spectrum frame, materialized via an inverse STFT; 'time' "
                             "outputs a block of samples directly; 'noise' outputs a scalar "
                             "that scales a block of uniform [-1, 1] noise; 'shift' outputs a "
                             "block of samples and a scalar that circularly shifts it; "
                             "'noise-shift' is the sum of 'shift' and 'noise'; 'mag-phase' outputs "
                             "per-bin magnitudes and per-bin fractional deviations from each bin's "
                             "expected phase advance, accumulated over time into phase, then "
                             "materialized via an inverse STFT; 'time-noise' is the sum of 'time' "
                             "and 'noise' (see --noise-envelope)")
    parser.add_argument('--noise-envelope', choices=['smooth', 'frame'], default='smooth',
                        help="'time-noise' only: 'smooth' low-passes the noise level along time "
                             "(see --noise-cutoff-hz), so it can't flutter at the step rate; "
                             "'frame' lets it change every step")
    parser.add_argument('--noise-cutoff-hz', type=float, default=20.,
                        help="'time-noise' with --noise-envelope smooth only: the noise level's "
                             "low-pass cutoff (-3dB)")
    parser.add_argument('--loss', choices=['mel', 'multiband'], default='mel',
                        help="'mel' is a log-mel spectrogram L1 loss; 'multiband' is the "
                             "multi-band/multi-samplerate spectrogram loss from parallel.py")
    parser.add_argument('--sparsity', choices=['loss', 'topk'], default='loss',
                        help="'loss' encourages a sparse control signal with an additional L0 "
                             "loss term; 'topk' keeps exactly --n-to-keep non-zero elements")
    parser.add_argument('--n-to-keep', type=int, default=128,
                        help="number of non-zero elements in the control signal, 'topk' only")
    parser.add_argument('--sparsity-loss-coeff', type=float, default=None,
                        help="'loss' only; defaults depend on --loss, since the two losses "
                             "have very different magnitudes. Values that are too large can "
                             "drive every control element to zero, from which training can't "
                             "recover")
    parser.add_argument('--n-random-events', type=int, default=16,
                        help='non-zero elements in the random control signal used for the '
                             "'random' logger and the conservation of energy loss")
    parser.add_argument('--energy-loss', action=argparse.BooleanOptionalAction, default=True,
                        help='penalize output energy exceeding the input (control signal) '
                             'energy so far, at every step, for both the reconstruction and '
                             'a render with a random control signal')
    parser.add_argument('--energy-loss-coeff', type=float, default=1.0)
    parser.add_argument('--decay-penalty', action=argparse.BooleanOptionalAction, default=True,
                        help='penalize gradual, step-by-step decreases along each control '
                             'channel, so decays are left to the instrument, while impulses, '
                             'swells and sustained (e.g. bowed) control stay cheap')
    parser.add_argument('--decay-penalty-coeff', type=float, default=None,
                        help='defaults depend on --loss, like --sparsity-loss-coeff')
    parser.add_argument('--self-supervised', action=argparse.BooleanOptionalAction, default=False,
                        help='also train on synthetic examples: known instruments (with chosen, often '
                             'long, decay times) struck by known, sparse impulses. These get the '
                             'reconstruction loss, plus a decay curve loss (estimated vs. known '
                             "instrument) and an onset profile loss (estimated vs. known control). "
                             "Real examples are unaffected. '--rnn tanh' only")
    parser.add_argument('--synthetic-batch-size', type=int, default=2)
    parser.add_argument('--synthetic-min-t60', type=float, default=0.2,
                        help="synthetic instruments' shortest mode decay time (to -60dB), in seconds")
    parser.add_argument('--synthetic-max-t60', type=float, default=4.0,
                        help="synthetic instruments' longest mode decay time (to -60dB), in seconds")
    parser.add_argument('--synthetic-events', type=int, default=8,
                        help='impulses per synthetic control signal')
    parser.add_argument('--synthetic-coeff', type=float, default=1.0,
                        help='weight of all synthetic-example losses together')
    parser.add_argument('--decay-curve-coeff', type=float, default=None,
                        help='weight of the decay curve loss; defaults depend on --loss')
    parser.add_argument('--onset-coeff', type=float, default=None,
                        help='weight of the onset profile loss; defaults depend on --loss')
    parser.add_argument('--probe-amplitude', type=float, default=0.05,
                        help='strength of the impulse used to measure decay curves; small, to keep '
                             'instruments in their near-linear range')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collection', type=str, default='hyper_rnn')
    parser.add_argument('--port', type=int, default=9998)
    args = parser.parse_args()

    if args.n_samples % args.step_size != 0:
        raise ValueError('--n-samples must be divisible by --step-size')
    if args.rnn == 'complex' and args.output_mode != 'complex':
        parser.error('--rnn complex requires --output-mode complex')
    if args.self_supervised and args.rnn != 'tanh':
        parser.error("--self-supervised only supports --rnn tanh")
    if args.self_supervised and not 0 < args.synthetic_min_t60 <= args.synthetic_max_t60:
        parser.error('need 0 < --synthetic-min-t60 <= --synthetic-max-t60')

    OUTPUT_MODES['time-noise'] = TimeNoiseOutput(
        envelope=args.noise_envelope, cutoff_hz=args.noise_cutoff_hz, samplerate=args.samplerate)

    collection = LmdbCollection(path=args.collection)

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random'],
        'audio/wav',
        lambda x: encode_audio(x, samplerate=args.samplerate),
        collection)

    control_logger, = loggers(
        ['control'],
        SupportedContentType.Spectrogram.value,
        lambda x: x,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    # the synthetic pair only exists with --self-supervised
    if args.self_supervised:
        synth_audio, synth_recon_audio = loggers(
            ['synth', 'synth-recon'],
            'audio/wav',
            lambda x: encode_audio(x, samplerate=args.samplerate),
            collection)
        synth_loggers = [synth_audio, synth_recon_audio]
    else:
        synth_loggers = []

    serve_conjure(
        [orig_audio, recon_audio, random_audio, control_logger, *synth_loggers],
        port=args.port,
        n_workers=1,
        web_components_version='0.0.101')

    model_key, random_key = jax.random.split(jax.random.key(args.seed))

    model = HyperRNN(
        step_size=args.step_size,
        channels=args.channels,
        control_channels=args.control_channels,
        hidden_dim=args.hidden_dim,
        n_to_keep=args.n_to_keep,
        key=model_key,
        hard_sparsify=args.sparsity == 'topk',
        output_mode=args.output_mode,
        rnn=args.rnn,
        complex_nonlinearity=args.complex_nonlinearity)

    # if we are "hard" sparsifying, then don't bother with sparsity loss
    if args.sparsity == 'topk':
        sparsity_coeff = 0.0
    elif args.sparsity_loss_coeff is not None:
        sparsity_coeff = args.sparsity_loss_coeff
    else:
        sparsity_coeff = DEFAULT_SPARSITY_LOSS_COEFF[args.loss]

    if args.decay_penalty_coeff is not None:
        decay_coeff = args.decay_penalty_coeff
    else:
        decay_coeff = DEFAULT_DECAY_PENALTY_COEFF[args.loss]

    decay_curve_coeff = args.decay_curve_coeff \
        if args.decay_curve_coeff is not None else DEFAULT_DECAY_CURVE_COEFF[args.loss]
    onset_coeff = args.onset_coeff \
        if args.onset_coeff is not None else DEFAULT_ONSET_COEFF[args.loss]

    n_steps = args.n_samples // args.step_size

    # clip gradient values, and skip any update with NaN/Inf gradients, as
    # `parallel.py` does
    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip(0.5), optax.adam(args.lr)),
        max_consecutive_errors=100)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_func = make_loss_func(args.loss, args.samplerate)

    def compute_loss(model: HyperRNN, batch: jnp.ndarray, key):
        batch_size = batch.shape[0]
        recon_key, control_key, rnd_key = jax.random.split(key, 3)

        weights, control = jax.vmap(model.factor)(batch)
        recon = jax.vmap(execute_rnn)(weights, control, jax.random.split(recon_key, batch_size))

        # each instrument is also played with a random, sparse control
        # signal.  There's no target for these, so (if enabled) they're only
        # trained via the conservation of energy loss
        rnd_control = jax.vmap(random_control, in_axes=(0, 0, None))(
            jax.random.split(control_key, batch_size), control, args.n_random_events)
        rnd = jax.vmap(execute_rnn)(weights, rnd_control, jax.random.split(rnd_key, batch_size))

        recon_loss = loss_func(batch, recon)
        loss = recon_loss + (l0_norm(control) * sparsity_coeff)

        if args.energy_loss:
            energy_loss = \
                jax.vmap(energy_violation)(control, recon).sum() \
                + jax.vmap(energy_violation)(rnd_control, rnd).sum()
            loss = loss + (energy_loss * args.energy_loss_coeff)
        else:
            energy_loss = jnp.zeros(())

        if args.decay_penalty:
            decay_loss = jax.vmap(decay_penalty)(control).sum()
            loss = loss + (decay_loss * decay_coeff)
        else:
            decay_loss = jnp.zeros(())

        synth = dict(
            recon_loss=jnp.zeros(()),
            decay_curve_loss=jnp.zeros(()),
            onset_loss=jnp.zeros(()),
            audio=jnp.zeros((1, args.n_samples)),
            recon=jnp.zeros((1, args.n_samples)),
            known_curves=jnp.zeros((1, n_steps)),
            estimated_curves=jnp.zeros((1, n_steps)))

        if args.self_supervised:
            n = args.synthetic_batch_size
            instrument_key, control_key, render_key, s_recon_key, probe_key = jax.random.split(
                jax.random.fold_in(key, 1), 5)

            # --- synthetic examples: known instruments and control -> known audio ---
            known_weights = jax.vmap(
                lambda k: random_instrument(
                    k, args.control_channels, args.hidden_dim, args.output_mode, args.step_size,
                    args.samplerate, args.synthetic_min_t60, args.synthetic_max_t60))(
                jax.random.split(instrument_key, n))
            known_control = jax.vmap(
                lambda k: synthetic_control(k, args.control_channels, n_steps, args.synthetic_events))(
                jax.random.split(control_key, n))
            synthetic = jax.vmap(execute_rnn)(known_weights, known_control, jax.random.split(render_key, n))
            # peak-normalized, like real examples; the losses below don't
            # depend on the instrument's output scale
            synthetic = synthetic / (jnp.abs(synthetic).max(axis=-1, keepdims=True) + 1e-8)
            synthetic = jax.lax.stop_gradient(synthetic)

            # --- factored and reconstructed exactly like real examples ---
            est_weights, est_control = jax.vmap(model.factor)(synthetic)
            synth_recon = jax.vmap(execute_rnn)(est_weights, est_control, jax.random.split(s_recon_key, n))
            synth_recon_loss = loss_func(synthetic, synth_recon)

            # --- and, since the truth is known, compared with it ---
            probe_keys = jax.random.split(probe_key, n)

            def curve(weights, k):
                return decay_curve(weights, n_steps, k, args.probe_amplitude)

            known_curves = jax.lax.stop_gradient(jax.vmap(curve)(known_weights, probe_keys))
            estimated_curves = jax.vmap(curve)(est_weights, probe_keys)
            synth_decay_curve_loss = jax.vmap(decay_curve_loss)(known_curves, estimated_curves).sum()
            synth_onset_loss = jax.vmap(onset_profile_loss)(known_control, est_control).sum()

            loss = loss + args.synthetic_coeff * (
                synth_recon_loss
                + decay_curve_coeff * synth_decay_curve_loss
                + onset_coeff * synth_onset_loss)

            synth.update(
                recon_loss=synth_recon_loss,
                decay_curve_loss=synth_decay_curve_loss,
                onset_loss=synth_onset_loss,
                audio=synthetic,
                recon=synth_recon,
                known_curves=known_curves,
                estimated_curves=estimated_curves)

        return loss, (recon_loss, energy_loss, decay_loss, control, recon, rnd, synth)

    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        (loss, aux), grads = eqx.filter_value_and_grad(
            compute_loss, has_aux=True)(model, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, *aux

    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}')

    stream = batch_stream(args.mode, args.batch_size, args.n_samples, args.samplerate)

    for i, batch in zip(itertools.count(), stream):
        random_key, step_key = jax.random.split(random_key)
        model, opt_state, loss, recon_loss, energy_loss, decay_loss, control, recon, rnd, synth = train_step(
            model, opt_state, batch, step_key)

        orig_audio(max_norm(np.asarray(batch[0])))
        recon_audio(max_norm(np.asarray(recon[0])))
        random_audio(max_norm(np.asarray(rnd[0])))

        c = np.asarray(control[0])
        control_logger(c / (c.max() + 1e-12))

        sparsity = float((control > 0).sum() / control.size)
        line = f'{i} {float(loss)} {float(recon_loss)} {float(energy_loss)} {float(decay_loss)} {sparsity}'

        if args.self_supervised:
            synth_audio(max_norm(np.asarray(synth['audio'][0])))
            synth_recon_audio(max_norm(np.asarray(synth['recon'][0])))
            steps_per_second = args.samplerate / args.step_size
            known_t60 = np.median([
                t60_from_decay_curve(c, steps_per_second) for c in np.asarray(synth['known_curves'])])
            est_t60 = np.median([
                t60_from_decay_curve(c, steps_per_second) for c in np.asarray(synth['estimated_curves'])])
            line += (
                f'  | synth recon {float(synth["recon_loss"]):.1f}'
                f'  decay curve {float(synth["decay_curve_loss"]):.1f}'
                f'  onset {float(synth["onset_loss"]):.2f}'
                f'  T60 known {known_t60:.2f}s est {est_t60:.2f}s')

        print(line)


if __name__ == '__main__':
    main()
