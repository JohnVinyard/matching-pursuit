"""
The "instrument" as an audio-rate, diagonal, linear, time-invariant
state-space model (SSM):

    state[t] = A state[t - 1] + B u[t]        A = diag(lambda_1 ... lambda_M), complex
    y[t]     = Re(C state[t]) + D u[t]

Since A is diagonal, each state dimension is a single complex pole
lambda_i = exp(-alpha_i + j * theta_i): an exponentially decaying sinusoid
(a mode) with frequency theta_i and decay alpha_i.  The impulse response from
control channel c to the output is

    h_c[k] = Re( sum_i C_i B_ic lambda_i ** k )

i.e., a bank of damped harmonic oscillators that *share* their frequencies and
decays (the object's modes) across control channels, while each channel
excites them with its own complex amplitudes, or "residues", C_i B_ic (where
and how the object is struck).

Because the system is linear and time-invariant, it can be run as a
recurrence (one complex multiply-add per mode, per sample: streaming, cheap,
and runnable in the browser as a bank of 2-pole IIR filters), or, as here
during training, as a convolution with its materialized impulse responses, via
the FFT.  There's no frame rate and no overlap-add, so there are no
frame-rate artifacts, and |lambda_i| < 1 by construction, so the instrument is
always stable, and no energy or spectral radius constraints are needed.

The control signal is a sparse, non-negative signal at the frame rate
(`--step-size`); each non-zero element becomes an impulse at the start of its
frame, which is convolved with a short, learned per-channel excitation (a
"strike" or "pluck"), and then with the SSM (plus a direct path, D).

Modes start on a fixed, log-spaced frequency grid and may each move at most
`--freq-spread` semitones, which keeps the (otherwise very non-convex)
problem of finding frequencies by gradient descent well-conditioned: the
bank is overcomplete, so the model mostly selects modes via their residues
and decays, which have well-behaved gradients.

Modes:
    direct:  free parameters (instrument and control signal) fit to a single
             segment; no encoder
    overfit: the hypernetwork (below) trained on a single segment
    train:   the hypernetwork trained on the dataset: an analysis network
             hears a segment and produces an instrument (its poles,
             residues, excitations and direct paths, from a single,
             time-pooled latent) and a sparse control signal
"""
import argparse
import itertools
from dataclasses import dataclass
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, loggers, serve_conjure
from conjure.logger import encode_audio

from hyper_rnn import (
    Analysis, ControlSignalCreator, ParameterGenerator, make_loss_func, sparsify, l0_norm, max_norm,
    decay_penalty, random_control, batch_stream, DEFAULT_SPARSITY_LOSS_COEFF, DEFAULT_DECAY_PENALTY_COEFF)

# ln(1000): a mode's amplitude falls by 60dB over T60 seconds when its
# per-sample decay is LN_1000 / (T60 * samplerate)
LN_1000 = float(np.log(1000.))


@dataclass(frozen=True)
class SSMConfig:
    n_modes: int = 256
    control_channels: int = 32
    excitation_size: int = 128
    step_size: int = 128
    samplerate: int = 22050
    min_hz: float = 40.
    max_hz: float = 9000.
    freq_spread: float = 1.0
    min_t60: float = 0.02
    max_t60: float = 8.0

    def grid(self) -> jnp.ndarray:
        """The modes' resting frequencies, in Hz, log-spaced"""
        return jnp.asarray(np.geomspace(self.min_hz, self.max_hz, self.n_modes), dtype=jnp.float32)

    def excitation_window(self) -> jnp.ndarray:
        """Excitations fade out over their length (the second half of a hann window)"""
        n = self.excitation_size
        return 0.5 + 0.5 * jnp.cos(jnp.pi * jnp.arange(n) / n)


class Instrument(eqx.Module):
    """
    Unconstrained instrument parameters; `poles` and `render` map them into
    their valid ranges.
    """
    freq: jnp.ndarray        # (M,): deviation from the grid, via tanh, in units of freq_spread
    decay: jnp.ndarray       # (M,): T60, via sigmoid, log-interpolated between min_t60 and max_t60
    residue: jnp.ndarray     # (2, Cc, M): real and imaginary parts of each channel's excitation of each mode
    excitation: jnp.ndarray  # (Cc, excitation_size)
    direct: jnp.ndarray      # (Cc,): the D in y = Cx + Du


def poles(inst: Instrument, cfg: SSMConfig) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    returns: per-sample decay alpha (M,), per-sample phase advance theta (M,),
        frequency in Hz (M,) and T60 in seconds (M,), so that each mode's pole
        is exp(-alpha + j * theta)
    """
    hz = cfg.grid() * 2. ** (jnp.tanh(inst.freq) * cfg.freq_spread / 12.)
    hz = jnp.minimum(hz, 0.49 * cfg.samplerate)
    theta = 2 * jnp.pi * hz / cfg.samplerate

    log_min, log_max = np.log(cfg.min_t60), np.log(cfg.max_t60)
    t60 = jnp.exp(log_min + jax.nn.sigmoid(inst.decay) * (log_max - log_min))
    alpha = LN_1000 / (t60 * cfg.samplerate)
    return alpha, theta, hz, t60


@jax.checkpoint
def impulse_responses(
        alpha: jnp.ndarray,
        theta: jnp.ndarray,
        residue: jnp.ndarray,
        time: jnp.ndarray) -> jnp.ndarray:
    """
    The SSM's convolution kernel, h_c[k] = Re(sum_i W_ci lambda_i ** k), with
    each mode scaled to (roughly) unit energy, sqrt(1 - |lambda| ** 2), so that
    long and short modes start on an equal footing.

    The (M, n_samples) intermediates dominate memory, so they're recomputed
    during the backward pass, rather than stored.

    residue: (2, Cc, M)
    time: (n_samples,) sample indices
    returns: (Cc, n_samples)
    """
    norm = jnp.sqrt(-jnp.expm1(-2 * alpha))
    env = jnp.exp(-alpha[:, None] * time[None, :])
    phase = theta[:, None] * time[None, :]
    wr = residue[0] * norm
    wi = residue[1] * norm
    return wr @ (env * jnp.cos(phase)) - wi @ (env * jnp.sin(phase))


def render(inst: Instrument, control: jnp.ndarray, cfg: SSMConfig) -> jnp.ndarray:
    """
    control: (Cc, n_frames), non-negative and sparse
    returns: (n_frames * step_size,)
    """
    cc, n_frames = control.shape
    n_samples = n_frames * cfg.step_size
    fft_size = n_samples * 2

    alpha, theta, _, _ = poles(inst, cfg)
    h = impulse_responses(alpha, theta, inst.residue, jnp.arange(n_samples, dtype=jnp.float32))

    # each control element is an impulse at the start of its frame
    impulses = jnp.zeros((cc, n_frames, cfg.step_size)).at[:, :, 0].set(control).reshape(cc, n_samples)
    excitation = inst.excitation * cfg.excitation_window()[None, :]

    u = jnp.fft.rfft(impulses, n=fft_size)
    e = jnp.fft.rfft(excitation, n=fft_size)
    k = jnp.fft.rfft(h, n=fft_size)
    spec = jnp.sum(u * e * (k + inst.direct[:, None]), axis=0)
    return jnp.fft.irfft(spec, n=fft_size)[:n_samples]


def tour_control(control: jnp.ndarray) -> jnp.ndarray:
    """
    Strike each control channel once, in order, evenly spaced, at the control
    signal's peak level: a "tour" of the instrument.
    """
    cc, n_frames = control.shape
    spacing = n_frames // cc
    scale = jax.lax.stop_gradient(jnp.max(control))
    return jnp.zeros_like(control).at[jnp.arange(cc), jnp.arange(cc) * spacing].set(scale)


# ---------------------------------------------------------------------------
# models; both expose `factor(audio) -> (Instrument, control)`

def control_from_raw(x: jnp.ndarray, hard_sparsify: bool, n_to_keep: int) -> jnp.ndarray:
    x = jax.nn.relu(x)
    return sparsify(x, n_to_keep) if hard_sparsify else x


class DirectSSM(eqx.Module):
    """Free parameters, fit to a single segment; `audio` is ignored"""
    instrument: Instrument
    control_raw: jnp.ndarray  # (Cc, n_frames)
    cfg: SSMConfig = eqx.field(static=True)
    hard_sparsify: bool = eqx.field(static=True)
    n_to_keep: int = eqx.field(static=True)

    def __init__(self, cfg: SSMConfig, n_frames: int, key, hard_sparsify: bool, n_to_keep: int):
        self.cfg = cfg
        self.hard_sparsify = hard_sparsify
        self.n_to_keep = n_to_keep
        k1, k2, k3 = jax.random.split(key, 3)
        cc, m, e = cfg.control_channels, cfg.n_modes, cfg.excitation_size
        self.instrument = Instrument(
            freq=jnp.zeros(m),
            decay=jnp.zeros(m),
            residue=jax.random.normal(k1, (2, cc, m)) * 0.1,
            excitation=jax.random.normal(k2, (cc, e)) / np.sqrt(e),
            direct=jnp.zeros(cc),
        )
        self.control_raw = jax.random.uniform(k3, (cc, n_frames), minval=-0.1, maxval=0.1)

    def factor(self, audio: jnp.ndarray) -> Tuple[Instrument, jnp.ndarray]:
        return self.instrument, control_from_raw(self.control_raw, self.hard_sparsify, self.n_to_keep)


class HyperSSM(eqx.Module):
    analysis: Analysis
    latents: ControlSignalCreator
    latents_attn: ControlSignalCreator
    control: ControlSignalCreator
    to_freq: ParameterGenerator
    to_decay: ParameterGenerator
    to_residue: ParameterGenerator
    to_excitation: ParameterGenerator
    to_direct: ParameterGenerator
    cfg: SSMConfig = eqx.field(static=True)

    def __init__(self, cfg: SSMConfig, channels: int, n_to_keep: int, key, hard_sparsify: bool):
        self.cfg = cfg
        cc, m, e = cfg.control_channels, cfg.n_modes, cfg.excitation_size
        keys = jax.random.split(key, 9)

        self.analysis = Analysis(
            cfg.step_size, channels, dilations=(1, 2, 4, 8, 16, 32, 1), key=keys[0])
        self.latents = ControlSignalCreator(
            channels, channels, n_to_keep, do_sparsify=False, key=keys[1])
        self.latents_attn = ControlSignalCreator(
            channels, channels, n_to_keep, do_sparsify=False, key=keys[2])
        # see `HyperRNN.__init__` regarding mean-subtraction and soft sparsity
        self.control = ControlSignalCreator(
            channels, cc, n_to_keep, do_sparsify=hard_sparsify, key=keys[3], subtract_mean=hard_sparsify)

        self.to_freq = ParameterGenerator(channels, m, keys[4])
        self.to_decay = ParameterGenerator(channels, m, keys[5])
        self.to_residue = ParameterGenerator(channels, 2 * cc * m, keys[6])
        self.to_excitation = ParameterGenerator(channels, cc * e, keys[7])
        self.to_direct = ParameterGenerator(channels, cc, keys[8])

    def factor(self, audio: jnp.ndarray) -> Tuple[Instrument, jnp.ndarray]:
        """
        audio: (n_samples,)
        returns: the instrument, and its sparse control signal (Cc, n_frames)
        """
        cc, m, e = self.cfg.control_channels, self.cfg.n_modes, self.cfg.excitation_size
        x = self.analysis(audio)
        latent = jnp.mean(self.latents(x) * self.latents_attn(x), axis=-1)
        inst = Instrument(
            freq=self.to_freq(latent),
            decay=self.to_decay(latent),
            residue=self.to_residue(latent).reshape(2, cc, m),
            excitation=self.to_excitation(latent).reshape(cc, e),
            direct=self.to_direct(latent),
        )
        return inst, self.control(x)


# ---------------------------------------------------------------------------
# training

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['direct', 'overfit', 'train'], required=True)
    parser.add_argument('--batch-size', type=int, default=2, help="'train' only")
    parser.add_argument('--n-samples', type=int, default=2 ** 17)
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--step-size', type=int, default=128,
                        help='audio samples per control frame; also the analysis frame size')
    parser.add_argument('--channels', type=int, default=128, help='analysis network channels')
    parser.add_argument('--control-channels', type=int, default=32)
    parser.add_argument('--n-modes', type=int, default=256, help='SSM state size: number of complex poles')
    parser.add_argument('--excitation-size', type=int, default=128,
                        help='length, in samples, of each channel\'s learned excitation ("strike")')
    parser.add_argument('--min-hz', type=float, default=40.)
    parser.add_argument('--max-hz', type=float, default=9000.)
    parser.add_argument('--freq-spread', type=float, default=1.0,
                        help='how far, in semitones, each mode may move from its place on the grid')
    parser.add_argument('--min-t60', type=float, default=0.02)
    parser.add_argument('--max-t60', type=float, default=8.0)
    parser.add_argument('--loss', choices=['mel', 'multiband'], default='mel')
    parser.add_argument('--sparsity', choices=['loss', 'topk'], default='loss')
    parser.add_argument('--n-to-keep', type=int, default=256, help="'topk' only")
    parser.add_argument('--sparsity-loss-coeff', type=float, default=None,
                        help="'loss' only; defaults depend on --loss, as in hyper_rnn.py")
    parser.add_argument('--decay-penalty', action=argparse.BooleanOptionalAction, default=True,
                        help="hyper_rnn.py's penalty on control signals that shape decays themselves")
    parser.add_argument('--decay-penalty-coeff', type=float, default=None)
    parser.add_argument('--n-random-events', type=int, default=16)
    parser.add_argument('--log-every', type=int, default=10,
                        help="render the 'random' and 'tour' examples every N steps")
    parser.add_argument('--lr', type=float, default=None, help="defaults to 1e-3 for 'direct', else 1e-4")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collection', type=str, default='jax_ssm')
    parser.add_argument('--port', type=int, default=9997)
    args = parser.parse_args()

    if args.n_samples % args.step_size != 0:
        parser.error('--n-samples must be divisible by --step-size')

    cfg = SSMConfig(
        n_modes=args.n_modes,
        control_channels=args.control_channels,
        excitation_size=args.excitation_size,
        step_size=args.step_size,
        samplerate=args.samplerate,
        min_hz=args.min_hz,
        max_hz=args.max_hz,
        freq_spread=args.freq_spread,
        min_t60=args.min_t60,
        max_t60=args.max_t60)
    n_frames = args.n_samples // args.step_size
    hard_sparsify = args.sparsity == 'topk'

    collection = LmdbCollection(path=args.collection)
    recon_audio, orig_audio, random_audio, tour_audio = loggers(
        ['recon', 'orig', 'random', 'tour'],
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
    serve_conjure(
        [orig_audio, recon_audio, random_audio, tour_audio, control_logger],
        port=args.port,
        n_workers=1,
        web_components_version='0.0.101')

    model_key, random_key = jax.random.split(jax.random.key(args.seed))
    if args.mode == 'direct':
        model = DirectSSM(cfg, n_frames, model_key, hard_sparsify, args.n_to_keep)
    else:
        model = HyperSSM(cfg, args.channels, args.n_to_keep, model_key, hard_sparsify)

    sparsity_coeff = 0.0 if hard_sparsify else (
        args.sparsity_loss_coeff if args.sparsity_loss_coeff is not None
        else DEFAULT_SPARSITY_LOSS_COEFF[args.loss])
    decay_coeff = args.decay_penalty_coeff if args.decay_penalty_coeff is not None \
        else DEFAULT_DECAY_PENALTY_COEFF[args.loss]
    lr = args.lr if args.lr is not None else (1e-3 if args.mode == 'direct' else 1e-4)

    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip(0.5), optax.adam(lr)),
        max_consecutive_errors=100)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss_func = make_loss_func(args.loss, args.samplerate)

    def play(inst, control):
        return render(inst, control, cfg)

    def compute_loss(model, batch):
        insts, control = jax.vmap(model.factor)(batch)
        recon = jax.vmap(play)(insts, control)
        recon_loss = loss_func(batch, recon)
        loss = recon_loss + l0_norm(control) * sparsity_coeff
        if args.decay_penalty:
            loss = loss + jax.vmap(decay_penalty)(control).sum() * decay_coeff
        return loss, (recon_loss, insts, control, recon)

    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, batch)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        return eqx.apply_updates(model, updates), opt_state, loss, *aux

    @eqx.filter_jit
    def examples(inst, control, key):
        rnd = play(inst, random_control(key, control, args.n_random_events))
        tour = play(inst, tour_control(control))
        return rnd, tour

    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}')
    stream_mode = 'train' if args.mode == 'train' else 'overfit'
    stream = batch_stream(stream_mode, args.batch_size, args.n_samples, args.samplerate)

    for i, batch in zip(itertools.count(), stream):
        model, opt_state, loss, recon_loss, insts, control, recon = train_step(model, opt_state, batch)

        orig_audio(max_norm(np.asarray(batch[0])))
        recon_audio(max_norm(np.asarray(recon[0])))
        c = np.asarray(control[0])
        control_logger(c / (c.max() + 1e-12))

        inst = jax.tree_util.tree_map(lambda x: x[0], insts)
        if i % args.log_every == 0:
            random_key, k = jax.random.split(random_key)
            rnd, tour = examples(inst, control[0], k)
            random_audio(max_norm(np.asarray(rnd)))
            tour_audio(max_norm(np.asarray(tour)))

        _, _, hz, t60 = poles(inst, cfg)
        shift = np.abs(12 * np.log2(np.asarray(hz) / np.asarray(cfg.grid())))
        sparsity = float((control > 0).sum() / control.size)
        print(
            f'{i} loss {float(loss):.1f} recon {float(recon_loss):.1f} density {sparsity:.4f}'
            f'  T60 median {float(jnp.median(t60)):.2f}s max {float(jnp.max(t60)):.2f}s'
            f'  mean shift {shift.mean():.2f}st')


if __name__ == '__main__':
    main()
