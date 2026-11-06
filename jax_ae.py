"""
An audio autoencoder that decomposes a segment of audio greedily, one event
at a time, in the spirit of `iterativedecomposition.py`, but in jax.

At each of a fixed number of steps:

    1. The encoder (an anti-causal, dilated conv stack, as in `hyper_rnn.py`)
       hears the current *residual* spectrogram, joined with a positional
       encoding of each frame, and produces one vector per frame.
    2. Per-frame (1x1) heads turn each frame's vector into a *candidate*
       event starting in that frame:
         - a magnitude (log-scaled, bounded),
         - a "fine" offset in [0, 1), the fraction of a hop into the frame
           at which the event starts, and
         - a low-dimensional event vector.
    3. The frame whose candidate has the largest magnitude is selected (as in
       `iterativedecomposition.py`, where the same relu'd attention value
       both selects the frame and scales the event).  That frame index is the
       event's "coarse" position.
    4. The event generator turns the selected event vector into samples.
       By default (`--generator resonance`, see `ResonanceGenerator`):
       enveloped noise, convolved with a segment-length resonance mixed from
       a learned bank of stacked damped harmonic oscillators (as in
       songsplat), a dry/filtered mix, then a mixture of fixed room impulse
       responses and a dry/wet mix.  Or
       (`--generator atoms`) a sparse, linear combination of a learned
       dictionary of raw-sample atoms.  Or (`--generator parallel`, see
       `ParallelGenerator`) `parallel.py`'s whole pipeline: hard-sparse
       control signals driving layers of routed, modified damped harmonic
       oscillators, then a room.  The result is normalized to unit
       (L2) norm, so the event vector describes only the event's character,
       then scaled by the magnitude, circularly FFT-shifted by `fine * hop`
       samples (fine position; the buffer has a hop of zero padding, so
       nothing audible wraps around), and placed at sample `frame * hop`
       (coarse position).
    5. The event's spectrogram is subtracted from the residual.

The selection itself is not differentiable.  The selected frame's heads get
gradients from the loss, so a frame chosen wrongly has its magnitude pushed
down until another frame wins, and the fine offset learns sub-frame timing.

By default (`--energy-loss greedy`), the loss is greedy: each step is
rewarded for the energy it removes from the residual spectrogram it was
given (the residual is not differentiated through, so steps don't trade off
against each other).  It's normalized by
the target's energy, so the summed loss is -(fraction of energy removed).
Alternatively (`--energy-loss final`), there's no per-step energy loss at
all: just the mean absolute error between the target's spectrogram and the
spectrogram of the final reconstruction (all events summed), normalized by
the target's L1 norm, so it's 1 for silence.  The steps then share credit,
and can trade off against each other.

Alongside it (by default; `--information-coeff 0` turns it off), each step is
also rewarded for the *information* it removes, regardless of scale: the
residual spectrogram, normalized to sum to one, is treated as a distribution
over (frame, bin) cells, and its entropy, relative to the maximum (a flat
spectrogram), measures how far it is from structureless, uniform noise.
Each step is rewarded for the increase in entropy it causes, normalized by
the target's entropy deficit, so this summed loss is -(fraction of the
target's structure removed).  Since it ignores scale, late steps, which can
only remove a sliver of the energy, still count fully for flattening what's
left.  Over-subtraction is clipped to zero before measuring, so it digs holes
(structure) rather than counting as added noise.  White noise's spectrogram
is about 1% short of the maximum entropy, real segments 13-23% short.

An auxiliary, self-supervised loss: synthetic targets are built from a few
known events, with random frames and fine offsets, random magnitudes
(log-uniform over the magnitude head's range) and random event vectors
(uniform over the event head's tanh-bounded range).  The same iterative
process runs on these, with the same greedy loss, and additionally, each
step's selected event is matched to a known one (a rough, greedy best
alignment on position, magnitude and event vector).  Then, at each step, the
heads' outputs *at the matched known event's frame* (not the selected frame)
are pulled toward the known fine offset, magnitude and event vector, i.e.,
the candidates are trained where the events really are, which is also what
makes those frames win the selection.
"""
import argparse
import itertools
from typing import Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, loggers, serve_conjure
from conjure.logger import encode_audio

from hyper_rnn import AntiCausalBlock, batch_stream, fft_shift, flattened_multiband_spectrogram, max_norm, stft
from scalar_jax import gaussian_blur


# ---------------------------------------------------------------------------
# helpers

def positional_encoding(n_frames: int, n_freqs: int) -> jnp.ndarray:
    """
    (2 * n_freqs + 1, n_frames): the normalized time itself, followed by
    sines and cosines at octave-spaced frequencies.
    """
    t = jnp.linspace(0, 1, n_frames, endpoint=False)
    freqs = (2.0 ** jnp.arange(n_freqs)) * jnp.pi
    angles = freqs[:, None] * t[None, :]
    return jnp.concatenate([t[None, :], jnp.sin(angles), jnp.cos(angles)], axis=0)


def sparsify_by_magnitude(x: jnp.ndarray, n_to_keep: int) -> jnp.ndarray:
    """
    Keep the `n_to_keep` elements of `x` with the largest magnitudes (signs
    intact), zeroing the rest.
    """
    _, indices = jax.lax.top_k(jnp.abs(x), n_to_keep)
    return jnp.zeros_like(x).at[indices].set(x[indices])


def gamma_noise_dictionary(key, n_atoms: int, atom_size: int) -> jnp.ndarray:
    """
    (n_atoms, atom_size): uniform [-1, 1] white noise, each atom under a
    gamma-distribution-shaped envelope, t ** (k - 1) * exp(-t / theta),
    peak-normalized, with random shape k in [1, 4] and a mean duration
    (k * theta) log-uniform in [32, atom_size / 4] samples.  The last 10% of
    each atom is faded out, so nothing is cut off abruptly.
    """
    k_shape, k_dur, k_noise = jax.random.split(key, 3)
    shape = jax.random.uniform(k_shape, (n_atoms, 1), minval=1., maxval=4.)
    mean_duration = jnp.exp(jax.random.uniform(
        k_dur, (n_atoms, 1), minval=np.log(32.), maxval=np.log(atom_size / 4)))
    theta = mean_duration / shape

    t = jnp.arange(atom_size)[None, :] + 1.
    log_env = (shape - 1) * jnp.log(t) - t / theta
    env = jnp.exp(log_env - log_env.max(axis=-1, keepdims=True))

    fade_size = atom_size // 10
    fade = jnp.concatenate([jnp.ones(atom_size - fade_size), jnp.linspace(1, 0, fade_size)])

    noise = jax.random.uniform(k_noise, (n_atoms, atom_size), minval=-1, maxval=1)
    return noise * env * fade


def greedy_match(cost: jnp.ndarray) -> jnp.ndarray:
    """
    A rough best alignment: repeatedly pair the (predicted, known) events
    with the lowest remaining cost, removing both from consideration.

    cost: (n_predicted, n_known), n_predicted == n_known
    returns: (n_predicted,), the known event matched to each predicted one
    """
    n = cost.shape[0]

    def body(_, state):
        cost, assignment = state
        flat_index = jnp.argmin(cost)
        i, j = flat_index // n, flat_index % n
        assignment = assignment.at[i].set(j)
        cost = cost.at[i, :].set(jnp.inf).at[:, j].set(jnp.inf)
        return cost, assignment

    _, assignment = jax.lax.fori_loop(0, n, body, (cost, jnp.zeros(n, dtype=jnp.int32)))
    return assignment


# ---------------------------------------------------------------------------
# event generators
#
# Each maps an event vector (and a random key, for generators that use fresh
# noise) to `size` samples.  `EventAutoencoder.render` then normalizes the
# result to unit norm, scales it by the event's magnitude and positions it.

def fft_convolve(a: jnp.ndarray, b: jnp.ndarray, size: int) -> jnp.ndarray:
    """
    The (linear, not circular) convolution of `a` and `b`, cropped to its
    first `size` samples.
    """
    n = a.shape[-1] + b.shape[-1]
    return jnp.fft.irfft(jnp.fft.rfft(a, n=n) * jnp.fft.rfft(b, n=n), n=n)[..., :size]


def load_rooms(path: str, samplerate: int, size: int) -> np.ndarray:
    """
    (n_rooms, size): every .wav impulse response in `path`, mono, resampled
    to `samplerate`, cropped (with a 10% fade-out, so a long tail doesn't end
    in a click) or zero-padded to `size` samples, and normalized to unit L2
    norm, so convolving white noise with any room preserves its energy and
    the dry/wet mix is a mix of comparable levels.  (`NeuralReverb` peak-
    normalizes instead.)
    """
    import glob
    import pathlib
    from librosa import load, to_mono

    paths = sorted(glob.glob(str(pathlib.Path(path).joinpath('*.wav'))))
    if not paths:
        raise ValueError(f'no .wav impulse responses found in {path}')

    fade_size = size // 10
    fade = np.concatenate([np.ones(size - fade_size), np.linspace(1, 0, fade_size)])

    rooms = []
    for p in paths:
        audio, _ = load(p, sr=samplerate, mono=False)
        audio = to_mono(audio)
        if len(audio) > size:
            audio = audio[:size] * fade
        else:
            audio = np.pad(audio, (0, size - len(audio)))
        rooms.append(audio / (np.linalg.norm(audio) + 1e-8))

    print(f'loaded {len(rooms)} impulse responses from {path}')
    return np.stack(rooms).astype(np.float32)


def gamma_envelope_targets(events: jnp.ndarray, n_frames: int, directions: jnp.ndarray) -> jnp.ndarray:
    """
    A gamma-distribution-shaped envelope for each event vector, peak 1, over
    `n_frames` control points: t ** (k - 1) * exp(-t / theta), with shape k
    in [1, 4] and mean duration (k * theta) log-uniform-ish in [2, n_frames /
    4] frames, each a smooth (sigmoid) function of the event vector's
    projection onto one of two fixed random directions, so nearby event
    vectors get similar envelopes.

    events: (n, event_dim), uniform in [-1, 1]
    directions: (2, event_dim), unit norm
    returns: (n, n_frames), the *log* envelopes
    """
    # a projection of U(-1, 1) ** event_dim onto a unit direction has std
    # 1 / sqrt(3); scaled to std 1.5, the sigmoids cover most of their range
    z = (events @ directions.T) * np.sqrt(3.) * 1.5
    shape = 1. + 3. * jax.nn.sigmoid(z[:, :1])
    log_min, log_max = np.log(2.), np.log(n_frames / 4)
    mean_duration = jnp.exp(log_min + jax.nn.sigmoid(z[:, 1:]) * (log_max - log_min))
    theta = mean_duration / shape

    t = jnp.arange(n_frames)[None, :] + 1.
    log_env = (shape - 1) * jnp.log(t) - t / theta
    return log_env - log_env.max(axis=-1, keepdims=True)


def fit_gamma_envelope_head(trunk, head: eqx.nn.Linear, event_dim: int, key, n_fit: int = 4096):
    """
    Initialize the envelope head (only its weights; the parameterization,
    softplus(head(trunk(event))), is unchanged and stays free) so that
    different event vectors produce different gamma-shaped envelopes (see
    `gamma_envelope_targets`): a ridge regression, over `n_fit` random event
    vectors, from the (initial) trunk's features to the inverse softplus of
    each event's target envelope.

    Log-gamma curves, (k - 1) log t - t / theta + c, are closed under linear
    combination, so the head's outputs for unseen event vectors (linear
    combinations of the features) are also roughly gamma-shaped.
    """
    k_dir, k_events = jax.random.split(key)
    directions = jax.random.normal(k_dir, (2, event_dim))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)

    n_frames = head.out_features
    events = jax.random.uniform(k_events, (n_fit, event_dim), minval=-1., maxval=1.)
    log_env = gamma_envelope_targets(events, n_frames, directions)

    # the inverse of softplus, log(exp(y) - 1), computed from log(y); for
    # small y it's ~log(y).  Floored at -12 (softplus(-12) ~ 6e-6, ~-100 dB):
    # fast-decaying targets' log tails run to -100s, and unfloored, those
    # (silent) tails dominate the squared error at the expense of the peaks
    y = jnp.exp(log_env)
    targets = jnp.where(log_env < -10., log_env, jnp.log(jnp.expm1(jnp.maximum(y, 1e-12))))
    targets = jnp.maximum(targets, -12.)

    features = jax.vmap(trunk)(events)
    features = jnp.concatenate([features, jnp.ones((n_fit, 1))], axis=-1)
    gram = features.T @ features
    ridge = 1e-3 * jnp.trace(gram) / gram.shape[0]
    solution = jnp.linalg.solve(gram + ridge * jnp.eye(gram.shape[0]), features.T @ targets)

    head = eqx.tree_at(lambda h: h.weight, head, solution[:-1].T)
    return eqx.tree_at(lambda h: h.bias, head, solution[-1])


class AtomGenerator(eqx.Module):
    """
    `--generator atoms`: a sparse, linear combination of a learned
    dictionary of raw-sample atoms (initialized as white noise with gamma-
    distribution envelopes); only the `n_active_atoms` largest weights (by
    magnitude) are kept.
    """
    to_atom_weights: eqx.nn.MLP
    dictionary: jnp.ndarray
    n_active_atoms: int = eqx.field(static=True)

    def __init__(self, event_dim: int, n_atoms: int, atom_size: int, n_active_atoms: int, key):
        k_mlp, k_dict = jax.random.split(key)
        self.to_atom_weights = eqx.nn.MLP(event_dim, n_atoms, width_size=256, depth=2, key=k_mlp)
        self.dictionary = gamma_noise_dictionary(k_dict, n_atoms, atom_size)
        self.n_active_atoms = n_active_atoms

    @property
    def size(self) -> int:
        return self.dictionary.shape[-1]

    def prepare(self):
        """Nothing shared between events to precompute."""
        return None

    def atom_weights(self, event: jnp.ndarray) -> jnp.ndarray:
        """
        event: (event_dim,)
        returns: (n_atoms,), with only `n_active_atoms` non-zero
        """
        return sparsify_by_magnitude(self.to_atom_weights(event), self.n_active_atoms)

    def __call__(self, event: jnp.ndarray, key=None, context=None) -> jnp.ndarray:
        return self.atom_weights(event) @ self.dictionary


# the damped harmonic oscillators' time step, per sample: songsplat's
# `torch.linspace(0, 10, n_samples)` for its 2 ** 15-sample events.  It's
# fixed, rather than tied to the segment length, so the oscillators'
# frequencies and decay rates (and the init ranges below, calibrated to it)
# mean the same thing whatever --n-samples is.  One time unit is ~3277
# samples, ~0.15s at 22050Hz; the Nyquist frequency is omega = pi / DHO_DT,
# ~10294, i.e., tension ~10 ** 8.
DHO_DT = 10. / 2 ** 15


def damped_harmonic_oscillator(
        time: jnp.ndarray,
        mass: jnp.ndarray,
        damping: jnp.ndarray,
        tension: jnp.ndarray,
        initial_displacement: jnp.ndarray) -> jnp.ndarray:
    """
    `modules.transfer.damped_harmonic_oscillator` (with zero initial
    velocity): the closed-form, underdamped displacement of a mass on a
    spring, released from `initial_displacement`.  All arguments broadcast.
    """
    x = damping / (2 * mass)
    omega = jnp.sqrt(jnp.clip(tension - x ** 2, 1e-12, None))
    phi = jnp.arctan2(x * initial_displacement, initial_displacement * omega)
    a = initial_displacement / jnp.cos(phi)
    return a * jnp.exp(-x * time) * jnp.cos(omega * time - phi)


class DampedHarmonicOscillatorBlock(eqx.Module):
    """
    songsplat's `DampedHarmonicOscillatorBlock` (with expressivity 1):
    `n_resonances` resonances, each the sum of `n_oscillators` damped
    harmonic oscillators, each soft-clipped (tanh of the oscillator times a
    learned amplitude).  Optionally, each oscillator's (log10) tension is
    modulated, over time, by another signal per resonance, scaled by
    per-oscillator `influence`.
    """
    mass: jnp.ndarray
    damping: jnp.ndarray
    tension: jnp.ndarray
    initial_displacement: jnp.ndarray
    amplitudes: jnp.ndarray

    def __init__(self, n_oscillators: int, n_resonances: int, key):
        shape = (n_oscillators, n_resonances)
        k_mass, k_damping, k_tension, k_disp, k_amp = jax.random.split(key, 5)
        self.mass = jax.random.uniform(k_mass, shape, minval=-2., maxval=2.)
        # songsplat initializes damping in U(0.5, 1.5); measured on the
        # mixed resonances (whose tails follow their slowest oscillators),
        # that's a T60 of only 0.10-0.16s, short for a resonance as long as
        # the whole segment.  U(-3, 1.5) spreads T60 over ~0.3-1.6s (10th-90th
        # percentile, median ~0.75s); U(-5, 1.5) overshoots, with a quarter of
        # them ringing past a 6s segment
        self.damping = jax.random.uniform(k_damping, shape, minval=-3., maxval=1.5)
        # songsplat: U(4, 9); capped at 8 here, as tensions above ~10 ** 8
        # alias (see DHO_DT)
        self.tension = jax.random.uniform(k_tension, shape, minval=4., maxval=8.)
        self.initial_displacement = jax.random.uniform(k_disp, shape, minval=-1., maxval=2.)
        self.amplitudes = jax.random.uniform(k_amp, shape, minval=-1., maxval=1.)

    def __call__(
            self,
            time: jnp.ndarray,
            modulation: Optional[jnp.ndarray] = None,
            influence: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        time: (n_samples,)
        modulation: optional (n_resonances, n_samples)
        influence: optional (n_oscillators, n_resonances)
        returns: (n_resonances, n_samples)

        (Or, with every parameter sliced to a single resonance, i.e.,
        (n_oscillators,), and modulation (n_samples,): (n_samples,).)
        """
        log_tension = self.tension[..., None]
        if modulation is not None:
            log_tension = log_tension + modulation[None] * influence[..., None]

        x = damped_harmonic_oscillator(
            time=time,
            mass=jax.nn.sigmoid(self.mass[..., None]) * 2,
            damping=jax.nn.sigmoid(self.damping[..., None]) * 30,
            tension=10 ** log_tension,
            initial_displacement=self.initial_displacement[..., None])
        x = jnp.tanh(x * self.amplitudes[..., None])
        return x.sum(axis=0)


class DampedHarmonicOscillatorResonances(eqx.Module):
    """
    songsplat's `DampedHarmonicOscillatorResonances`: a stack of
    `DampedHarmonicOscillatorBlock`s, each block's tension modulated by the
    previous block's output, and the blocks' outputs mixed (softmax over
    blocks, shared by all resonances).  songsplat computes the third block
    with the second block's weights (`self.dho2` twice, leaving `self.dho3`
    unused), which looks unintended; here each block has its own.

    The resonances are learned, but they don't depend on the event: each
    event chooses a mixture of them.  Each resonance is normalized to unit L2
    norm, so the mixture weights, not the resonances' (very different)
    lengths, decide which dominate.
    """
    blocks: list
    influences: list
    mix: jnp.ndarray
    n_samples: int = eqx.field(static=True)

    def __init__(self, n_samples: int, n_resonances: int, n_oscillators: int, n_layers: int, key):
        keys = jax.random.split(key, 2 * n_layers + 1)
        self.n_samples = n_samples
        self.blocks = [
            DampedHarmonicOscillatorBlock(n_oscillators, n_resonances, k) for k in keys[:n_layers]]
        self.influences = [
            jax.random.uniform(k, (n_oscillators, n_resonances), minval=-0.01, maxval=0.011)
            for k in keys[n_layers:2 * n_layers - 1]]
        self.mix = jax.random.uniform(keys[-1], (n_layers,), minval=-1., maxval=1.)

    def __call__(self, chunk_size: int = 16) -> jnp.ndarray:
        """
        returns: (n_resonances, n_samples)

        Each resonance's stack depends only on its own oscillators (block
        k + 1's modulation is block k's output for the same resonance), so
        resonances are computed `chunk_size` at a time, each chunk
        checkpointed: materializing every oscillator of every resonance at
        once, at full length, costs ~4GB in the backward pass.
        """
        time = jnp.arange(self.n_samples) * DHO_DT
        mix = jax.nn.softmax(self.mix)

        @jax.checkpoint
        def one_resonance(params):
            blocks, influences = params
            x = blocks[0](time)
            outputs = [x]
            for block, influence in zip(blocks[1:], influences):
                x = block(time, x, influence)
                outputs.append(x)
            return jnp.stack(outputs, axis=-1) @ mix

        # (n_oscillators, n_resonances) -> (n_resonances, n_oscillators),
        # so `lax.map` slices out one resonance at a time
        params, static = eqx.partition((self.blocks, self.influences), eqx.is_array)
        params = jax.tree_util.tree_map(lambda a: jnp.moveaxis(a, 1, 0), params)
        x = jax.lax.map(
            lambda p: one_resonance(eqx.combine(p, static)), params, batch_size=chunk_size)
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class ResonanceGenerator(eqx.Module):
    """
    `--generator resonance` (the default), after songsplat's
    `EventGenerator`, the resonance model in `iterativedecomposition.py` and
    `modules/reverb.NeuralReverb`.  A shared MLP trunk turns the event vector
    into:

        1. a low-frequency envelope (`envelope_frames` positive control
           points, linearly upsampled to `envelope_size` samples), which
           multiplies fresh uniform noise: the energy injected.  By default,
           it's initialized so that different event vectors start with
           different gamma-shaped envelopes (`--envelope-init`),
        2. a resonance as long as the whole segment: a (dense, linear)
           mixture of a learned bank of stacked damped harmonic oscillator
           resonances (`DampedHarmonicOscillatorResonances`), normalized to
           unit L2 norm, with which the noise burst is convolved,
        3. a dry/filtered mix (softmax over 2) between the noise burst and the
           noise burst convolved with the resonance,
        4. a room choice (softmax over the fixed impulse responses, so a
           convex combination of rooms) and a dry/wet mix (softmax over 2)
           between the signal from 3 and that signal convolved with the room.

    The rooms are fixed (never trained); everything else, including the
    resonance bank, is learned.  Events are `n_samples` long.
    """
    trunk: eqx.nn.MLP
    to_envelope: eqx.nn.Linear
    to_resonance: eqx.nn.Linear
    to_noise_mix: eqx.nn.Linear
    to_room: eqx.nn.Linear
    to_room_mix: eqx.nn.Linear
    resonances: DampedHarmonicOscillatorResonances
    rooms: jnp.ndarray

    n_samples: int = eqx.field(static=True)
    envelope_size: int = eqx.field(static=True)

    def __init__(
            self,
            event_dim: int,
            n_samples: int,
            envelope_frames: int,
            envelope_size: int,
            n_resonances: int,
            n_oscillators: int,
            n_dho_layers: int,
            rooms: np.ndarray,
            key,
            width: int = 256,
            envelope_init: str = 'gamma'):
        """
        rooms: (n_rooms, n_samples)
        envelope_init: 'gamma' initializes the envelope head so that
            different event vectors start with different gamma-shaped
            envelopes (see `fit_gamma_envelope_head`); 'random' leaves equinox's
            default init (near-flat envelopes)
        """
        keys = jax.random.split(key, 8)
        self.n_samples = n_samples
        self.envelope_size = envelope_size

        self.trunk = eqx.nn.MLP(
            event_dim, width, width_size=width, depth=2, final_activation=jax.nn.leaky_relu, key=keys[0])
        self.to_envelope = eqx.nn.Linear(width, envelope_frames, key=keys[1])
        if envelope_init == 'gamma':
            self.to_envelope = fit_gamma_envelope_head(self.trunk, self.to_envelope, event_dim, keys[7])
        self.to_resonance = eqx.nn.Linear(width, n_resonances, key=keys[2])
        self.to_noise_mix = eqx.nn.Linear(width, 2, key=keys[3])
        self.to_room = eqx.nn.Linear(width, rooms.shape[0], key=keys[4])
        self.to_room_mix = eqx.nn.Linear(width, 2, key=keys[5])
        self.resonances = DampedHarmonicOscillatorResonances(
            n_samples, n_resonances, n_oscillators, n_dho_layers, keys[6])
        self.rooms = jnp.asarray(rooms)

    @property
    def size(self) -> int:
        return self.n_samples

    def prepare(self) -> jnp.ndarray:
        """
        The resonance bank, (n_resonances, n_samples), shared by every event;
        computed once per batch, rather than once per event.
        """
        return self.resonances()

    def parameters(self, event: jnp.ndarray) -> dict:
        """
        The event's interpretable parameters (all but the noise itself).
        """
        h = self.trunk(event)
        control = jax.nn.softplus(self.to_envelope(h))
        return dict(
            envelope=jax.image.resize(control, (self.envelope_size,), method='linear'),
            resonance_weights=self.to_resonance(h),
            noise_mix=jax.nn.softmax(self.to_noise_mix(h)),
            room=jax.nn.softmax(self.to_room(h)),
            room_mix=jax.nn.softmax(self.to_room_mix(h)))

    def __call__(self, event: jnp.ndarray, key, context: jnp.ndarray) -> jnp.ndarray:
        """
        context: the resonance bank, from `prepare`
        """
        p = self.parameters(event)

        # 1. the energy injected: enveloped uniform noise
        noise = jax.random.uniform(key, (self.envelope_size,), minval=-1., maxval=1.)
        burst = jnp.pad(noise * p['envelope'], (0, self.n_samples - self.envelope_size))

        # 2, 3. the resonance (unit norm, so filtering white noise keeps its
        # energy, roughly), mixed with the unfiltered burst
        resonance = p['resonance_weights'] @ context
        resonance = resonance / (jnp.linalg.norm(resonance) + 1e-8)
        filtered = fft_convolve(burst, resonance, self.n_samples)
        dry = p['noise_mix'][0] * burst + p['noise_mix'][1] * filtered

        # 4. the room: a convex combination of the fixed impulse responses,
        # renormalized, since the rooms are largely uncorrelated, so a spread-
        # out combination (as at initialization, ~uniform over the rooms)
        # cancels most of their energy, and the wet signal would be far
        # quieter than the dry; then mixed with the dry signal
        room = p['room'] @ jax.lax.stop_gradient(self.rooms)
        room = room / (jnp.linalg.norm(room) + 1e-8)
        wet = fft_convolve(dry, room, self.n_samples)
        return p['room_mix'][0] * dry + p['room_mix'][1] * wet


# `parallel.py`'s time convention: its DHO's time runs over linspace(0, 10,
# 2 ** 17), so, as with DHO_DT, a fixed time step, rather than one tied to
# --n-samples.  The Nyquist frequency is omega = pi / PARALLEL_DT, ~41175,
# i.e., tension ~10 ** 9.2.
PARALLEL_DT = 10. / 2 ** 17


def sparse_control(x: jnp.ndarray, n_to_keep: int) -> jnp.ndarray:
    """
    `parallel.py`'s `ControlSignalCreator`, followed by its hard `sparsify`:
    rectify, subtract the mean, rectify again, then keep only the
    `n_to_keep` largest elements (of the whole plane), zeroing the rest.
    Gradients reach only the kept elements.
    """
    x = jnp.abs(x)
    x = jax.nn.relu(x - x.mean())
    flat = x.reshape(-1)
    _, indices = jax.lax.top_k(flat, n_to_keep)
    return jnp.zeros_like(flat).at[indices].set(flat[indices]).reshape(x.shape)


def leaky_integrator(forces: jnp.ndarray, damping: jnp.ndarray) -> jnp.ndarray:
    """
    `parallel.py`'s `sequential`: out[i] = (forces[i] + out[i - 1]) *
    damping[i], along the last axis, as a parallel (associative) scan, which
    is exact and, unlike `parallel.py`'s cumprod/cumsum trick, stable.
    """
    def combine(left, right):
        a1, b1 = left
        a2, b2 = right
        return a1 * a2, a2 * b1 + b2

    _, out = jax.lax.associative_scan(combine, (damping, damping * forces), axis=-1)
    return out


class ParallelLayerHeads(eqx.Module):
    """
    One layer's `InstrumentHyperNetwork`, from `parallel.py`: everything that
    defines a layer of `n_nodes` routed, modified damped harmonic
    oscillators, from the generator's shared (trunk) features.
    """
    to_mass: eqx.nn.Linear
    to_tension: eqx.nn.Linear
    to_damping: eqx.nn.Linear
    to_gain: eqx.nn.Linear
    to_output_gain: eqx.nn.Linear
    to_filters: eqx.nn.Linear
    to_filters_mix: eqx.nn.Linear
    to_force_router: eqx.nn.Linear
    to_tension_router: eqx.nn.Linear

    n_nodes: int = eqx.field(static=True)
    n_filter_taps: int = eqx.field(static=True)

    def __init__(self, width: int, n_nodes: int, n_filter_taps: int, key):
        keys = jax.random.split(key, 11)
        self.n_nodes = n_nodes
        self.n_filter_taps = n_filter_taps

        def head(k, size, weight_scale=1., bias=None):
            layer = eqx.nn.Linear(width, size, key=k)
            layer = eqx.tree_at(lambda l: l.weight, layer, layer.weight * weight_scale)
            if bias is not None:
                layer = eqx.tree_at(lambda l: l.bias, layer, jnp.asarray(bias, dtype=jnp.float32))
            return layer

        eye = jnp.eye(n_nodes).reshape(-1)
        self.to_mass = head(keys[0], n_nodes)
        # each node starts at its own pitch, spread over the whole tension
        # range (see `parameters`), which the event vector then moves
        self.to_tension = head(
            keys[1], n_nodes, weight_scale=4.,
            bias=jax.random.uniform(keys[9], (n_nodes,), minval=-3., maxval=3.))
        # likewise, each node's own decay, from ~0.07 (a click) to ~0.95 per
        # control frame (rings for a few seconds)
        self.to_damping = head(
            keys[2], n_nodes, weight_scale=4.,
            bias=jax.random.uniform(keys[10], (n_nodes,), minval=-3., maxval=3.))
        self.to_gain = head(keys[3], n_nodes)
        self.to_output_gain = head(keys[4], 1)
        self.to_filters = head(keys[5], n_nodes * n_filter_taps)
        self.to_filters_mix = head(keys[6], n_nodes * 2)
        # as in `parallel.py`'s `Layer`: forces start (nearly) routed to their
        # own nodes, and the layer-to-layer tension (FM) routing starts small
        self.to_force_router = head(keys[7], n_nodes * n_nodes, weight_scale=0.1, bias=eye)
        self.to_tension_router = head(keys[8], n_nodes * n_nodes, weight_scale=0.1, bias=0. * eye)

    def __call__(self, h: jnp.ndarray, base_resonance: float, tension_range: Tuple[float, float]) -> dict:
        n = self.n_nodes
        lo, hi = tension_range
        return dict(
            mass=1e-5 + jnp.abs(self.to_mass(h)),
            log_tension=lo + jax.nn.sigmoid(self.to_tension(h)) * (hi - lo),
            damping=base_resonance + jax.nn.sigmoid(self.to_damping(h)) * (1 - base_resonance) * 0.9998,
            gain=self.to_gain(h),
            output_gain=self.to_output_gain(h)[0],
            filter_taps=self.to_filters(h).reshape(n, self.n_filter_taps),
            filters_mix=self.to_filters_mix(h).reshape(n, 2),
            force_router=self.to_force_router(h).reshape(n, n),
            tension_router=self.to_tension_router(h).reshape(n, n))


class ParallelGenerator(eqx.Module):
    """
    `--generator parallel`: `parallel.py`'s whole generation pipeline
    (`InstrumentAutoencoder`'s decoder), driven by a single event vector.
    A shared MLP trunk turns the event vector into:

        1. sparse control signals, (n_nodes, control_frames) each, for the
           first `control_frames` frames (of `control_rate` samples) of the
           event: forces, and tension and damping modulations, each hard-
           sparsified to its `n_to_keep` largest elements
           (`sparse_control`),
        2. for each of `n_layers` layers of `n_nodes` oscillators, the
           layer's definition (`ParallelLayerHeads`): mass, tension, damping,
           gains, low-pass filters, and force and tension routing,
        3. a room choice and a dry/wet mix, as in `ResonanceGenerator`.

    Each layer (`parallel.py`'s `execute_parallel_layer`): the forces are
    routed to the nodes, divided by mass and accumulated by a leaky
    integrator at the control rate (`leaky_integrator`, with the node's
    damping plus the damping modulation), upsampled to samples, and that
    energy envelope scales a damped harmonic oscillator's carrier (in place
    of its exponential decay).  Its tension is modulated, in log10 space, by
    the tension modulation, and by the previous layer's output, routed (FM).
    Each node is mixed with itself through its own short low-pass filter,
    soft-clipped with a gain, and scaled by the layer's output gain.  The
    last layer's nodes are summed, then the room is applied.  Events are
    `n_samples` long.

    Departures from `parallel.py`: tension is bounded, `10 ** (lo + sigmoid
    * (hi - lo))`, its non-dictionary path, with the raw head initialized
    to spread the nodes over the range; the oscillator's frequency is
    clipped below Nyquist; the room mixture and dry/wet mix are softmaxes
    (as in `ResonanceGenerator`), rather than relu'd and unconstrained.
    """
    trunk: eqx.nn.MLP
    to_forces: eqx.nn.Linear
    to_tension_mod: eqx.nn.Linear
    to_damp_mod: eqx.nn.Linear
    layers: list
    to_room: eqx.nn.Linear
    to_room_mix: eqx.nn.Linear
    rooms: jnp.ndarray

    n_samples: int = eqx.field(static=True)
    n_nodes: int = eqx.field(static=True)
    control_rate: int = eqx.field(static=True)
    control_frames: int = eqx.field(static=True)
    n_forces: int = eqx.field(static=True)
    n_modulations: int = eqx.field(static=True)
    modulation_scale: float = eqx.field(static=True)
    filter_size: int = eqx.field(static=True)
    mass_coeff: float = eqx.field(static=True)
    base_resonance: float = eqx.field(static=True)
    tension_range: Tuple[float, float] = eqx.field(static=True)

    def __init__(
            self,
            event_dim: int,
            n_samples: int,
            rooms: np.ndarray,
            key,
            n_nodes: int = 16,
            n_layers: int = 2,
            control_rate: int = 512,
            control_frames: int = 16,
            n_forces: int = 8,
            n_modulations: int = 4,
            modulation_scale: float = 0.1,
            filter_size: int = 256,
            n_filter_taps: int = 64,
            mass_coeff: float = 0.01,
            base_resonance: float = 0.02,
            tension_range: Tuple[float, float] = (1., 8.),
            width: int = 256):
        """
        rooms: (n_rooms, n_samples)
        control_frames: the control signals span the event's first
            `control_frames * control_rate` samples (by default 8192, like
            `ResonanceGenerator`'s noise burst); the oscillators ring on for
            the rest of the event
        n_forces, n_modulations: how many nonzero elements the force and
            the (each of the) tension/damping modulation planes keep
        modulation_scale: the tension (log10) and damping modulations' scale
        """
        if n_samples % control_rate != 0:
            raise ValueError('n_samples must be divisible by control_rate')
        keys = jax.random.split(key, 6 + n_layers)
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.control_rate = control_rate
        self.control_frames = control_frames
        self.n_forces = n_forces
        self.n_modulations = n_modulations
        self.modulation_scale = modulation_scale
        self.filter_size = filter_size
        self.mass_coeff = mass_coeff
        self.base_resonance = base_resonance
        self.tension_range = tension_range

        control_size = n_nodes * control_frames
        self.trunk = eqx.nn.MLP(
            event_dim, width, width_size=width, depth=2, final_activation=jax.nn.leaky_relu, key=keys[0])
        self.to_forces = eqx.nn.Linear(width, control_size, key=keys[1])
        self.to_tension_mod = eqx.nn.Linear(width, control_size, key=keys[2])
        self.to_damp_mod = eqx.nn.Linear(width, control_size, key=keys[3])
        self.layers = [
            ParallelLayerHeads(width, n_nodes, n_filter_taps, keys[6 + i]) for i in range(n_layers)]
        self.to_room = eqx.nn.Linear(width, rooms.shape[0], key=keys[4])
        self.to_room_mix = eqx.nn.Linear(width, 2, key=keys[5])
        self.rooms = jnp.asarray(rooms)

    @property
    def size(self) -> int:
        return self.n_samples

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.control_rate

    def prepare(self):
        return None

    def parameters(self, event: jnp.ndarray) -> dict:
        """
        The event's control signals, at the control rate and zero past the
        first `control_frames`, (n_nodes, n_frames) each; each layer's
        definition; and the room and dry/wet mix.
        """
        h = self.trunk(event)
        shape = (self.n_nodes, self.control_frames)
        pad = ((0, 0), (0, self.n_frames - self.control_frames))

        def control(head, n_to_keep, scale=1.):
            return jnp.pad(sparse_control(head(h).reshape(shape), n_to_keep) * scale, pad)

        return dict(
            forces=control(self.to_forces, self.n_forces),
            tension_mod=control(self.to_tension_mod, self.n_modulations, self.modulation_scale),
            damp_mod=control(self.to_damp_mod, self.n_modulations, self.modulation_scale),
            layers=[layer(h, self.base_resonance, self.tension_range) for layer in self.layers],
            room=jax.nn.softmax(self.to_room(h)),
            room_mix=jax.nn.softmax(self.to_room_mix(h)))

    def upsample(self, x: jnp.ndarray) -> jnp.ndarray:
        """(n_nodes, n_frames) -> (n_nodes, n_samples), linearly"""
        return jax.image.resize(x, (x.shape[0], self.n_samples), method='linear')

    def layer(self, p: dict, forces, tension_mod, damp_mod, previous) -> jnp.ndarray:
        """
        `parallel.py`'s `execute_parallel_layer`.

        p: the layer's definition, from `ParallelLayerHeads`
        previous: the previous layer's output, (n_nodes, n_samples), or None
        returns: (n_nodes, n_samples)
        """
        forces = p['force_router'].T @ forces
        damping = jnp.clip(p['damping'][:, None] + damp_mod, 0., 1.)
        mass = p['mass'][:, None] * self.mass_coeff

        energy = self.upsample(leaky_integrator(forces / mass, damping))

        # modulations accumulate in log10 space; the base tension's
        # exponent is bounded, so this is the only unbounded exponential
        log_tension = p['log_tension'][:, None] + self.upsample(tension_mod)
        if previous is not None:
            log_tension = log_tension + p['tension_router'].T @ previous

        # the (modified) damped harmonic oscillator, with unit damping and
        # initial displacement: the energy envelope replaces the decay
        x = 1. / (2 * mass)
        omega = jnp.sqrt(jnp.abs(10. ** log_tension - x ** 2))
        omega = jnp.minimum(omega, 0.95 * np.pi / PARALLEL_DT)
        phi = jnp.arctan2(x, omega)
        time = jnp.arange(self.n_samples) * PARALLEL_DT
        z = energy * jnp.cos(omega * time - phi) / jnp.cos(phi)

        # each node's own low-pass filter (`parallel.py`'s `compute_filters`:
        # taps upsampled in the frequency domain, so nothing above their own
        # Nyquist), unit norm, and a dry/filtered mix
        filters = jnp.fft.irfft(jnp.fft.rfft(p['filter_taps'], axis=-1), n=self.filter_size, axis=-1)
        filters = filters / (jnp.linalg.norm(filters, axis=-1, keepdims=True) + 1e-8)
        filtered = fft_convolve(z, filters, self.n_samples)
        z = p['filters_mix'][:, :1] * z + p['filters_mix'][:, 1:] * filtered

        return jnp.tanh(z * p['gain'][:, None]) * p['output_gain']

    def __call__(self, event: jnp.ndarray, key=None, context=None) -> jnp.ndarray:
        p = self.parameters(event)

        out = None
        for layer in p['layers']:
            out = self.layer(layer, p['forces'], p['tension_mod'], p['damp_mod'], out)
        dry = out.sum(axis=0)

        # the room, as in `ResonanceGenerator`
        room = p['room'] @ jax.lax.stop_gradient(self.rooms)
        room = room / (jnp.linalg.norm(room) + 1e-8)
        wet = fft_convolve(dry, room, self.n_samples)
        return p['room_mix'][0] * dry + p['room_mix'][1] * wet


# ---------------------------------------------------------------------------
# model

class Encoder(eqx.Module):
    """
    Anti-causal, dilated, gated conv stack (as in `hyper_rnn.Analysis`), fed
    a spectrogram joined with a positional encoding of each frame.  Being
    anti-causal, each frame's output summarizes that frame and what follows
    it, i.e., what an event starting there would have to account for.
    """
    spec_proj: eqx.nn.Conv1d
    pe_proj: eqx.nn.Conv1d
    join: eqx.nn.Conv1d
    blocks: list
    ff: eqx.nn.Conv1d
    pe: jnp.ndarray

    def __init__(self, n_coeffs: int, n_frames: int, channels: int, pe_freqs: int, dilations, key):
        keys = jax.random.split(key, len(dilations) + 4)
        self.pe = positional_encoding(n_frames, pe_freqs)
        self.spec_proj = eqx.nn.Conv1d(n_coeffs, channels, 1, key=keys[0])
        self.pe_proj = eqx.nn.Conv1d(self.pe.shape[0], channels, 1, key=keys[1])
        self.join = eqx.nn.Conv1d(channels * 2, channels, 1, key=keys[2])
        self.blocks = [
            AntiCausalBlock(channels, kernel_size=2, dilation=d, key=k)
            for d, k in zip(dilations, keys[3:-1])
        ]
        self.ff = eqx.nn.Conv1d(channels, channels, 1, key=keys[-1])

    def __call__(self, spec: jnp.ndarray) -> jnp.ndarray:
        """
        spec: (n_frames, n_coeffs)
        returns: (channels, n_frames)
        """
        x = self.spec_proj(spec.T)
        p = self.pe_proj(jax.lax.stop_gradient(self.pe))
        x = self.join(jnp.concatenate([x, p], axis=0))

        output = jnp.zeros_like(x)
        for block in self.blocks:
            x = block(x)
            output = output + x
        return self.ff(output / len(self.blocks))


class EventAutoencoder(eqx.Module):
    encoder: Encoder
    frame_norm: eqx.nn.LayerNorm
    to_magnitude: eqx.nn.Conv1d
    to_fine: eqx.nn.Conv1d
    to_event: eqx.nn.Conv1d
    generator: eqx.Module

    n_samples: int = eqx.field(static=True)
    window: int = eqx.field(static=True)
    hop: int = eqx.field(static=True)
    spectrogram_exponent: float = eqx.field(static=True)
    min_magnitude: float = eqx.field(static=True)
    max_magnitude: float = eqx.field(static=True)

    def __init__(
            self,
            n_samples: int,
            window: int,
            hop: int,
            spectrogram_exponent: float,
            channels: int,
            pe_freqs: int,
            event_dim: int,
            generator: eqx.Module,
            min_magnitude: float,
            max_magnitude: float,
            key):
        """
        generator: an event generator (`AtomGenerator` or
            `ResonanceGenerator`), mapping an event vector of size
            `event_dim` (and a random key) to `generator.size` samples
        """

        self.n_samples = n_samples
        self.window = window
        self.hop = hop
        self.spectrogram_exponent = spectrogram_exponent
        self.min_magnitude = min_magnitude
        self.max_magnitude = max_magnitude

        keys = jax.random.split(key, 4)  # the LayerNorm needs no key
        self.encoder = Encoder(
            window // 2 + 1,
            n_samples // hop,
            channels,
            pe_freqs,
            dilations=(1, 2, 4, 8, 16, 32, 64, 1),
            key=keys[0])

        # brings each frame's vector to unit scale, so the heads' biases
        # can't swamp it
        self.frame_norm = eqx.nn.LayerNorm(channels)

        # per-frame heads.  The magnitude head also sees the log of the
        # residual's level, which the encoder's input normalization removes;
        # see `analyze`
        self.to_magnitude = eqx.nn.Conv1d(channels + 1, 1, 1, key=keys[1])
        self.to_fine = eqx.nn.Conv1d(channels, 1, 1, key=keys[2])
        self.to_event = eqx.nn.Conv1d(channels, event_dim, 1, key=keys[3])
        self.generator = generator

    @property
    def n_frames(self) -> int:
        return self.n_samples // self.hop

    def spectrogram(self, audio: jnp.ndarray) -> jnp.ndarray:
        """
        audio: (n_samples,)
        returns: (n_frames, n_coeffs); frame f covers samples
            [f * hop, f * hop + window)
        """
        return stft(audio, ws=self.window, step=self.hop, pad=True) ** self.spectrogram_exponent

    def log_magnitude(self, logits: jnp.ndarray) -> jnp.ndarray:
        """
        Loudness spans orders of magnitude, so the magnitude head chooses a
        point in log-magnitude, bounded to the same range the synthetic
        targets' known magnitudes are drawn from (log-uniformly).
        """
        log_min, log_max = np.log(self.min_magnitude), np.log(self.max_magnitude)
        return log_min + jax.nn.sigmoid(logits) * (log_max - log_min)

    def analyze(self, spec: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        One candidate event per frame.

        spec: (n_frames, n_coeffs), the current residual
        returns: magnitude logits (n_frames,), fine offsets in (0, 1)
            (n_frames,) and event vectors in (-1, 1) (n_frames, event_dim)
        """
        # normalize the residual to unit RMS, so the encoder sees unit-scale
        # input at every step, however much energy is left.  (Unnormalized,
        # spectrogram bins average ~0.02-0.05, and the positional encoding,
        # in [-1, 1], drowns them out.)
        level = jnp.sqrt(jnp.mean(spec ** 2)) + 1e-8
        x = self.encoder(spec / level)
        x = jax.vmap(self.frame_norm, in_axes=1, out_axes=1)(x)

        # the magnitude head also sees the log of the level the input
        # normalization removed, without which it couldn't tell a loud
        # residual from a quiet one
        log_level = jnp.broadcast_to(jnp.log(level), (1, x.shape[-1]))
        logits = self.to_magnitude(jnp.concatenate([x, log_level], axis=0))[0]

        fine = jax.nn.sigmoid(self.to_fine(x)[0])

        # bounded to (-1, 1), the same range the synthetic targets' known
        # event vectors are drawn from (see `--known-events`)
        events = jnp.tanh(self.to_event(x)).T
        return logits, fine, events

    def encode(self, spec: jnp.ndarray):
        """
        Select the frame whose candidate is loudest.

        spec: (n_frames, n_coeffs), the current residual
        returns: the selected frame (int), its fine offset in (0, 1), its
            magnitude in (min_magnitude, max_magnitude), its event vector
            (event_dim,), and all the candidates, as returned by `analyze`
        """
        logits, fine, events = self.analyze(spec)
        frame = jnp.argmax(logits)
        magnitude = jnp.exp(self.log_magnitude(logits[frame]))
        return frame, fine[frame], magnitude, events[frame], (logits, fine, events)

    def render(
            self,
            frame: jnp.ndarray,
            fine: jnp.ndarray,
            magnitude: jnp.ndarray,
            event: jnp.ndarray,
            key,
            context=None) -> jnp.ndarray:
        """
        frame: int in [0, n_frames), the coarse position
        fine: scalar in [0, 1), the fine position, a fraction of a hop
        magnitude: positive scalar, the event's L2 norm
        event: (event_dim,)
        key: for the generator's noise, if it uses any
        context: whatever the generator shares between events (see its
            `prepare`), e.g., the resonance bank
        returns: (n_samples,)
        """
        samples = self.generator(event, key, context)
        samples = samples / (jnp.linalg.norm(samples) + 1e-8) * magnitude

        # fine: a circular shift of up to one hop, with a hop of zero padding
        # at the end, so it's only (near-)silence that wraps around
        padded = jnp.pad(samples, (0, self.hop))
        shifted = fft_shift(padded, fine * self.hop)

        # coarse: placed at the frame's first sample, on a canvas long enough
        # that an event starting near the end rings out into the extra room
        # (which is then cropped) rather than being moved or wrapped
        canvas = jnp.zeros(self.n_samples + shifted.shape[0])
        canvas = jax.lax.dynamic_update_slice(canvas, shifted, (frame * self.hop,))
        return canvas[:self.n_samples]


def make_level_norm(loss_scales: str, levels: int):
    """
    Returns `level_norm(spec)`: the L1 norm of a spectrogram, either as-is
    ('single'), or summed over gaussian blurs of it along time and frequency
    with std 1, 2, 4 ... frames/bins ('blur'), as in `scalar_jax.py`, so the
    energy-removal loss also sees coarse scales.
    """
    def level_norm(spec: jnp.ndarray) -> jnp.ndarray:
        total = jnp.abs(spec).sum()
        if loss_scales == 'blur':
            for i in range(levels):
                sigma = 2.0 ** i
                blurred = gaussian_blur(spec, sigma, 0, circular=False)
                blurred = gaussian_blur(blurred, sigma, 1, circular=False)
                total = total + jnp.abs(blurred).sum()
        return total
    return level_norm


def entropy_deficit(spec: jnp.ndarray) -> jnp.ndarray:
    """
    How far a spectrogram is from flat, i.e., from uniform noise, regardless
    of its scale: 1 - (entropy of the spectrogram normalized to sum to one) /
    (the maximum entropy, log of the number of cells).  0 when flat, 1 when
    all energy is in a single cell.  Negative values (over-subtraction) are
    clipped to zero, so they count as holes, not energy.
    """
    a = jnp.maximum(spec, 0.).reshape(-1)
    p = a / (a.sum() + 1e-12)
    entropy = -(p * jnp.log(p + 1e-12)).sum()
    return 1. - entropy / np.log(a.size)


def decompose(model: EventAutoencoder, target_audio: jnp.ndarray, n_steps: int, level_norm, key, context=None):
    """
    Greedily remove `n_steps` events from `target_audio`.

    target_audio: (n_samples,)
    key: split into one key per step, for the generator's noise
    context: from `model.generator.prepare()`
    returns: a dict of
        frames, fines, magnitudes: (n_steps,), the selected events
        events: (n_steps, event_dim), the selected event vectors
        audio: (n_steps, n_samples)
        losses: (n_steps,), the energy-removal loss
        information_losses: (n_steps,), the information-removal loss
        candidates: each step's (logits, fines, events) for every frame,
            (n_steps, n_frames[, event_dim])
        residual: (n_frames, n_coeffs), the final residual (STFT domain --
            see `step`)
    """
    target_spec = model.spectrogram(target_audio)

    # the greedy (energy-removal) loss below is computed in a different
    # representation than the one driving event selection: the flattened
    # multiband spectrogram (see `flattened_multiband_spectrogram`, and its
    # use in `parallel.py`'s own (commented-out) loss_func), rather than
    # model.spectrogram's single-resolution STFT. Event selection/encoding
    # (`model.encode`, in `step` below) and the information (entropy-
    # deficit) loss are untouched -- they still see model.spectrogram, i.e.
    # the same spectrogram passed to analysis as before.
    total = jax.lax.stop_gradient(
        level_norm(flattened_multiband_spectrogram(target_audio))) + 1e-8
    # floored, so a target that's nearly noise already can't blow this up
    total_deficit = jnp.maximum(jax.lax.stop_gradient(entropy_deficit(target_spec)), 1e-2)

    def step(carry, step_key):
        residual, audio_residual = carry
        frame, fine, magnitude, event, candidates = model.encode(residual)
        audio = model.render(frame, fine, magnitude, event, step_key, context)
        event_spec = model.spectrogram(audio)

        # the greedy loss itself: flattened multiband spectrogram, computed
        # from the actual (time-domain) residual/event audio -- an exact
        # subtraction, unlike the STFT residual's magnitude-only
        # approximation (phase is discarded there, so it's only ever
        # threaded through for analysis, not for this loss)
        residual_mb = flattened_multiband_spectrogram(audio_residual)
        event_mb = flattened_multiband_spectrogram(audio)
        loss = (level_norm(residual_mb - event_mb) - level_norm(residual_mb)) / total

        # likewise, negative when the residual moves toward uniform noise --
        # still STFT-domain, unaffected by the above
        information_loss = (
            entropy_deficit(residual - event_spec) - entropy_deficit(residual)) / total_deficit

        residual = jax.lax.stop_gradient(residual - event_spec)
        audio_residual = jax.lax.stop_gradient(audio_residual - audio)
        return (residual, audio_residual), dict(
            frames=frame, fines=fine, magnitudes=magnitude, events=event,
            audio=audio, losses=loss, information_losses=information_loss,
            candidates=candidates)

    # each step is rematerialized in the backward pass, so only the carry
    # (the STFT and audio residuals) is kept per step, rather than every
    # step's encoder activations, audio and spectrograms: far less memory
    # at the defaults, for roughly one extra forward pass
    (residual, audio_residual), out = jax.lax.scan(
        jax.checkpoint(step, prevent_cse=False),
        (target_spec, target_audio), jax.random.split(key, n_steps))
    return dict(**out, residual=residual)


def alignment_loss(
        model: EventAutoencoder,
        decomposition: dict,
        known_frames: jnp.ndarray,
        known_fines: jnp.ndarray,
        known_magnitudes: jnp.ndarray,
        known_events: jnp.ndarray,
        align_magnitude_weight: float,
        align_event_weight: float):
    """
    Match each step's selected event to a known one, by position, (weighted)
    log-magnitude distance and (weighted) relative event vector distance.
    Then, at each step, read the candidates at the *matched known event's
    frame* and measure how far they are from the known event.  Pulling them
    together trains the candidates where the events really are; a frame
    selected wrongly is meanwhile discouraged by the greedy loss.

    decomposition: as returned by `decompose`, with k steps
    known_*: (k,), or (k, event_dim) for the event vectors
    returns: a dict of
        fine_error: mean absolute fine offset error at the known frames, as
            a fraction of a hop
        magnitude_error: mean absolute log-magnitude error at the known
            frames, as a fraction of the magnitude range
        event_error: mean squared event vector error at the known frames,
            relative to the known vectors' mean squared norm
        position_error: mean absolute distance between the *selected*
            events' positions and their matches, a fraction of the segment
            (for monitoring only)
        hit_rate: the fraction of steps that selected the matched known
            event's frame (for monitoring only)
    """
    log_magnitude_range = np.log(model.max_magnitude / model.min_magnitude)
    scale = jax.lax.stop_gradient(jnp.mean(jnp.sum(known_events ** 2, axis=-1))) + 1e-8

    frames = decomposition['frames']
    positions = (frames + decomposition['fines']) / model.n_frames
    known_positions = (known_frames + known_fines) / model.n_frames
    log_magnitudes = jnp.log(decomposition['magnitudes'])
    known_log_magnitudes = jnp.log(known_magnitudes)
    events = decomposition['events']

    position_dist = jnp.abs(positions[:, None] - known_positions[None, :])
    magnitude_dist = jnp.abs(log_magnitudes[:, None] - known_log_magnitudes[None, :]) / log_magnitude_range
    event_dist = jnp.sqrt(jnp.sum((events[:, None, :] - known_events[None, :, :]) ** 2, axis=-1) / scale)
    assignment = greedy_match(jax.lax.stop_gradient(
        position_dist + align_magnitude_weight * magnitude_dist + align_event_weight * event_dist))

    # each step's candidates at its matched known event's frame
    target_frames = known_frames[assignment]
    steps = jnp.arange(frames.shape[0])
    logits, fines, candidate_events = decomposition['candidates']
    at_logits = logits[steps, target_frames]
    at_fines = fines[steps, target_frames]
    at_events = candidate_events[steps, target_frames]

    fine_error = jnp.mean(jnp.abs(at_fines - known_fines[assignment]))
    magnitude_error = jnp.mean(
        jnp.abs(model.log_magnitude(at_logits) - known_log_magnitudes[assignment])) / log_magnitude_range
    event_error = jnp.mean(jnp.sum((at_events - known_events[assignment]) ** 2, axis=-1)) / scale

    return dict(
        fine_error=fine_error,
        magnitude_error=magnitude_error,
        event_error=event_error,
        position_error=jax.lax.stop_gradient(jnp.mean(jnp.abs(positions - known_positions[assignment]))),
        hit_rate=jnp.mean((frames == target_frames).astype(jnp.float32)))


def freeze_decoder(model: EventAutoencoder) -> EventAutoencoder:
    """
    The same model, but with gradients stopped at the decoder's (event
    generator's) weights, so a loss computed with it can only update the
    encoder.
    """
    def stop(tree):
        return jax.tree_util.tree_map(
            lambda x: jax.lax.stop_gradient(x) if eqx.is_array(x) else x, tree)

    return eqx.tree_at(lambda m: m.generator, model, stop(model.generator))


# ---------------------------------------------------------------------------
# training

def make_compute_loss(args):
    """
    Returns `compute_loss(model, batch, key) -> (loss, aux)`, configured by
    the parsed command-line `args`.
    """
    level_norm = make_level_norm(args.loss_scales, args.loss_blur_levels)

    def final_error(model: EventAutoencoder, decomposition: dict, target_specs: jnp.ndarray) -> jnp.ndarray:
        """
        L1 error between each target's spectrogram and its final
        reconstruction's (all events summed), relative to the target's L1
        norm (so 1 for silence), averaged over the batch
        """
        recon_specs = jax.vmap(model.spectrogram)(decomposition['audio'].sum(axis=1))
        return jax.vmap(
            lambda t, r: level_norm(t - r) / (jax.lax.stop_gradient(level_norm(t)) + 1e-8))(
            target_specs, recon_specs).mean()

    def energy_loss(model: EventAutoencoder, decomposition: dict, target_specs: jnp.ndarray) -> jnp.ndarray:
        """
        --energy-loss 'greedy': the summed per-step energy-removal losses,
        -(fraction of energy removed); 'final': `final_error`
        """
        if args.energy_loss == 'greedy':
            return decomposition['losses'].sum(axis=-1).mean()
        return final_error(model, decomposition, target_specs)

    def compute_loss(model: EventAutoencoder, batch: jnp.ndarray, key):
        real_key, key = jax.random.split(key)
        # anything the generator shares between events (the resonance bank),
        # computed once
        context = model.generator.prepare()
        specs = jax.vmap(model.spectrogram)(batch)
        real = jax.vmap(lambda a, kk: decompose(model, a, args.n_steps, level_norm, kk, context))(
            batch, jax.random.split(real_key, batch.shape[0]))
        # the per-step energy removal is always reported, even when it isn't
        # what's trained ('final')
        greedy_loss = real['losses'].sum(axis=-1).mean()
        information_loss = real['information_losses'].sum(axis=-1).mean()
        energy = energy_loss(model, real, specs)
        loss = energy + args.information_coeff * information_loss

        aux = dict(
            greedy=greedy_loss,
            final_error=energy if args.energy_loss == 'final' else jnp.zeros(()),
            information=information_loss,
            recon=real['audio'].sum(axis=1),
            residual=real['residual'],
            frames=real['frames'],
            synthetic_greedy=jnp.zeros(()),
            fine_error=jnp.zeros(()),
            magnitude_error=jnp.zeros(()),
            event_error=jnp.zeros(()),
            position_error=jnp.zeros(()),
            hit_rate=jnp.zeros(()),
            synthetic=jnp.zeros((1, args.n_samples)),
            synthetic_recon=jnp.zeros((1, args.n_samples)))

        if args.synthetic:
            k = args.n_synthetic_events
            n = args.synthetic_batch_size
            frame_key, fine_key, magnitude_key, event_key, render_key, fake_key = jax.random.split(key, 6)
            log_magnitude_range = np.log(args.max_magnitude / args.min_magnitude)

            # known events: random frames and fine offsets, and magnitudes and
            # event vectors drawn either uniformly from their heads' ranges
            # (log-uniformly, for magnitudes), or from those the encoder just
            # selected for the real batch.  The latter lets the event vector
            # error be satisfied trivially, by collapsing every event vector to
            # the same value.
            if args.known_events == 'uniform':
                known_events = jax.random.uniform(
                    event_key, (n, k, args.event_dim), minval=-1., maxval=1.)
                known_magnitudes = args.min_magnitude * jnp.exp(
                    jax.random.uniform(magnitude_key, (n, k)) * log_magnitude_range)
            else:
                pool_events = jax.lax.stop_gradient(real['events']).reshape(-1, args.event_dim)
                pool_magnitudes = jax.lax.stop_gradient(real['magnitudes']).reshape(-1)
                indices = jax.random.randint(event_key, (n, k), 0, pool_events.shape[0])
                known_events = pool_events[indices]
                known_magnitudes = pool_magnitudes[indices]
            known_frames = jax.random.randint(frame_key, (n, k), 0, model.n_frames)
            known_fines = jax.random.uniform(fine_key, (n, k))

            synthetic = jax.vmap(jax.vmap(
                lambda f, fi, m, e, kk: model.render(f, fi, m, e, kk, context)))(
                known_frames, known_fines, known_magnitudes, known_events,
                jax.random.split(render_key, (n, k))).sum(axis=1)
            synthetic = jax.lax.stop_gradient(synthetic)

            # the synthetic targets come from the decoder itself, so by default
            # they only train the encoder: the decomposition still renders
            # events with the decoder, but no gradient reaches its weights
            s_model = model if args.synthetic_decoder_grads else freeze_decoder(model)
            s_context = context if args.synthetic_decoder_grads else jax.lax.stop_gradient(context)
            synthetic_specs = jax.vmap(model.spectrogram)(synthetic)
            fake = jax.vmap(lambda a, kk: decompose(s_model, a, k, level_norm, kk, s_context))(
                synthetic, jax.random.split(fake_key, n))
            synthetic_greedy = fake['losses'].sum(axis=-1).mean()
            synthetic_information = fake['information_losses'].sum(axis=-1).mean()

            alignment = jax.vmap(
                lambda d, kf, kfi, km, ke: alignment_loss(
                    model, d, kf, kfi, km, ke,
                    args.align_magnitude_weight,
                    args.align_event_weight))(
                fake, known_frames, known_fines, known_magnitudes, known_events)
            alignment = {name: value.mean() for name, value in alignment.items()}

            loss = loss + args.synthetic_coeff * (
                energy_loss(s_model, fake, synthetic_specs)
                + args.information_coeff * synthetic_information
                + args.fine_coeff * alignment['fine_error']
                + args.magnitude_coeff * alignment['magnitude_error']
                + args.event_coeff * alignment['event_error'])

            aux.update(
                synthetic_greedy=synthetic_greedy,
                synthetic=synthetic,
                synthetic_recon=fake['audio'].sum(axis=1),
                **alignment)

        return loss, aux

    return compute_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'overfit'], required=True)
    parser.add_argument('--batch-size', type=int, default=2, help='ignored in overfit mode')
    parser.add_argument('--n-samples', type=int, default=2 ** 17)
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--window', type=int, default=2048, help='spectrogram window size')
    parser.add_argument('--hop', type=int, default=256, help='spectrogram hop size')
    parser.add_argument('--spectrogram-exponent', type=float, default=1.,
                        help='1: magnitude spectrogram (default, as in iterativedecomposition.py); 2: power')
    parser.add_argument('--channels', type=int, default=128, help='encoder channels')
    parser.add_argument('--pe-freqs', type=int, default=16, help='positional encoding frequencies')
    parser.add_argument('--event-dim', type=int, default=16)
    parser.add_argument('--generator', choices=['resonance', 'atoms', 'parallel'], default='resonance',
                        help="'resonance': enveloped noise -> damped harmonic oscillator resonance -> "
                             "dry/filtered mix -> room -> dry/wet mix; events are --n-samples long. "
                             "'atoms': a sparse sum of dictionary atoms, --atom-size long. "
                             "'parallel': parallel.py's whole pipeline (sparse control signals driving "
                             "layers of routed, modified damped harmonic oscillators, then a room), "
                             "configured by ParallelGenerator's constructor defaults; --n-samples long")
    parser.add_argument('--n-atoms', type=int, default=2048, help="'atoms' only: dictionary size")
    parser.add_argument('--atom-size', type=int, default=4096, help="'atoms' only: the event's length")
    parser.add_argument('--n-active-atoms', type=int, default=16,
                        help="'atoms' only: non-zero atom weights per event")
    parser.add_argument('--n-resonances', type=int, default=128,
                        help="'resonance' only: resonances in the shared, learned bank each event mixes")
    parser.add_argument('--n-oscillators', type=int, default=2,
                        help="'resonance' only: damped harmonic oscillators summed per resonance, per layer")
    parser.add_argument('--dho-layers', type=int, default=3,
                        help="'resonance' only: stacked oscillator blocks, each modulating the next's "
                             "tension, mixed (as in songsplat)")
    parser.add_argument('--envelope-frames', type=int, default=128,
                        help="'resonance' only: the noise envelope's control points")
    parser.add_argument('--envelope-size', type=int, default=8192,
                        help="'resonance' only: the noise envelope's length in samples, after "
                             "upsampling (as in iterativedecomposition.py)")
    parser.add_argument('--envelope-init', choices=['gamma', 'random'], default='gamma',
                        help="'resonance' only: 'gamma' initializes the envelope head so different event "
                             "vectors start with different gamma-shaped envelopes (shape 1-4, mean "
                             "duration 2 to --envelope-frames/4 frames); 'random': default init, "
                             "near-flat envelopes")
    parser.add_argument('--rooms-path', type=str, default=None,
                        help="'resonance' and 'parallel' only: a directory of .wav impulse responses; defaults to "
                             "IMPULSE_RESPONSE_PATH from .env")
    parser.add_argument('--min-magnitude', type=float, default=0.01,
                        help="smallest event magnitude (L2 norm); the magnitude head's lower bound")
    parser.add_argument('--max-magnitude', type=float, default=50.,
                        help="largest event magnitude (L2 norm); the magnitude head's upper bound. For "
                             "reference, a 4096-sample event with an RMS of 0.2 has a norm of ~13")
    parser.add_argument('--n-steps', type=int, default=32, help='events removed per segment')
    parser.add_argument('--energy-loss', choices=['greedy', 'final'], default='greedy',
                        help="'greedy': each step is rewarded for the energy it removes from the "
                             "residual; 'final': no per-step energy loss, only the L1 error between the "
                             "target's spectrogram and the final reconstruction's.  Either way, it's "
                             "paired with the per-step information loss (--information-coeff)")
    parser.add_argument('--loss-scales', choices=['single', 'blur'], default='single',
                        help="'single' measures energy removal on the spectrogram as-is; 'blur' also "
                             "on time/frequency blurs of it (as in scalar_jax.py)")
    parser.add_argument('--loss-blur-levels', type=int, default=8,
                        help="'blur' only; std up to 2 ** (levels - 1) frames, a quarter of the "
                             "segment by default")
    parser.add_argument('--information-coeff', type=float, default=1.,
                        help='weight of the information-removal loss (each step pushing the residual '
                             'spectrogram toward flat, uniform noise, regardless of scale), alongside '
                             'the energy-removal loss; 0 for energy only')
    parser.add_argument('--synthetic', action=argparse.BooleanOptionalAction, default=True,
                        help='auxiliary, self-supervised loss on synthetic targets built from known events')
    parser.add_argument('--n-synthetic-events', type=int, default=8,
                        help='known events per synthetic target, and decomposition steps run on it')
    parser.add_argument('--synthetic-batch-size', type=int, default=2)
    parser.add_argument('--known-events', choices=['uniform', 'recent'], default='uniform',
                        help="synthetic targets' event vectors and magnitudes: 'uniform' draws them "
                             "from U(-1, 1), the event head's (tanh) range, and log-uniformly from "
                             "[--min-magnitude, --max-magnitude]; 'recent' draws them from the "
                             "encoder's outputs for the current batch, which permits event vector "
                             "collapse")
    parser.add_argument('--synthetic-decoder-grads', action=argparse.BooleanOptionalAction, default=False,
                        help="let the synthetic task's greedy loss update the decoder (event generator) "
                             "too; by default, the synthetic task only updates the encoder")
    parser.add_argument('--synthetic-coeff', type=float, default=1.)
    parser.add_argument('--fine-coeff', type=float, default=1.,
                        help='weight of the fine offset error (at the known frames), measured as a '
                             'fraction of a hop')
    parser.add_argument('--magnitude-coeff', type=float, default=1.,
                        help='weight of the log-magnitude error (at the known frames), measured as a '
                             'fraction of the magnitude range')
    parser.add_argument('--event-coeff', type=float, default=1.)
    parser.add_argument('--align-magnitude-weight', type=float, default=0.1,
                        help='weight of log-magnitude distance (vs. position distance) when matching '
                             'predicted to known events')
    parser.add_argument('--align-event-weight', type=float, default=0.1,
                        help='weight of event vector distance (vs. position distance) when matching '
                             'predicted to known events')
    parser.add_argument('--n-random-events', type=int, default=8,
                        help="events in the 'random-events' render: random event vectors (uniform over "
                             "the event head's range), equal magnitudes, evenly spaced")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collection', type=str, default='jax_ae')
    parser.add_argument('--port', type=int, default=9997)
    args = parser.parse_args()

    if args.n_samples % args.hop != 0:
        raise ValueError('--n-samples must be divisible by --hop')
    if args.generator == 'parallel' and args.n_samples % 512 != 0:
        raise ValueError("--generator parallel needs --n-samples divisible by 512 (its control rate)")
    if args.generator == 'resonance' and args.envelope_size > args.n_samples:
        raise ValueError('--envelope-size must not exceed --n-samples')
    if args.n_active_atoms > args.n_atoms:
        raise ValueError('--n-active-atoms must not exceed --n-atoms')
    if not 0 < args.min_magnitude < args.max_magnitude:
        raise ValueError('need 0 < --min-magnitude < --max-magnitude')

    collection = LmdbCollection(path=args.collection)

    def audio_loggers(names):
        return loggers(
            names,
            'audio/wav',
            lambda x: encode_audio(x, samplerate=args.samplerate),
            collection)

    orig_audio, recon_audio, random_audio = audio_loggers(['orig', 'recon', 'random-events'])

    # the synthetic pair only exists when the self-supervised loss is on
    if args.synthetic:
        synthetic_audio, synthetic_recon_audio = audio_loggers(['synthetic', 'synthetic-recon'])
        synthetic_loggers = [synthetic_audio, synthetic_recon_audio]
    else:
        synthetic_loggers = []

    residual_logger, = loggers(
        ['residual'],
        SupportedContentType.Spectrogram.value,
        lambda x: x,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    serve_conjure(
        [orig_audio, recon_audio, random_audio, *synthetic_loggers, residual_logger],
        port=args.port,
        n_workers=1,
        web_components_version='0.0.101')

    model_key, generator_key, random_key = jax.random.split(jax.random.key(args.seed), 3)

    if args.generator in ('resonance', 'parallel'):
        from config.dotenv import Config
        rooms = load_rooms(args.rooms_path or Config.impulse_response_path(), args.samplerate, args.n_samples)

    if args.generator == 'parallel':
        generator = ParallelGenerator(
            event_dim=args.event_dim,
            n_samples=args.n_samples,
            rooms=rooms,
            key=generator_key)
    elif args.generator == 'resonance':
        generator = ResonanceGenerator(
            event_dim=args.event_dim,
            n_samples=args.n_samples,
            envelope_frames=args.envelope_frames,
            envelope_size=args.envelope_size,
            n_resonances=args.n_resonances,
            n_oscillators=args.n_oscillators,
            n_dho_layers=args.dho_layers,
            rooms=rooms,
            envelope_init=args.envelope_init,
            key=generator_key)
    else:
        generator = AtomGenerator(
            event_dim=args.event_dim,
            n_atoms=args.n_atoms,
            atom_size=args.atom_size,
            n_active_atoms=args.n_active_atoms,
            key=generator_key)

    model = EventAutoencoder(
        n_samples=args.n_samples,
        window=args.window,
        hop=args.hop,
        spectrogram_exponent=args.spectrogram_exponent,
        channels=args.channels,
        pe_freqs=args.pe_freqs,
        event_dim=args.event_dim,
        generator=generator,
        min_magnitude=args.min_magnitude,
        max_magnitude=args.max_magnitude,
        key=model_key)

    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip(0.5), optax.adam(args.lr)),
        max_consecutive_errors=100)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    compute_loss = make_compute_loss(args)

    @eqx.filter_jit
    def render_random(model, key):
        """
        Hear the generator on its own: `--n-random-events` random event
        vectors, drawn uniformly from the event head's (tanh) range, each with
        the same magnitude, starting at evenly spaced frames.
        """
        n = args.n_random_events
        event_key, noise_key = jax.random.split(key)
        events = jax.random.uniform(event_key, (n, args.event_dim), minval=-1., maxval=1.)
        frames = (jnp.arange(n) * model.n_frames) // n
        context = model.generator.prepare()
        audio = jax.vmap(lambda f, fi, m, e, kk: model.render(f, fi, m, e, kk, context))(
            frames, jnp.zeros(n), jnp.ones(n), events, jax.random.split(noise_key, n))
        return audio.sum(axis=0)

    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    def magnitude_error_db(error):
        """mean absolute log-magnitude error, from a fraction of the range, to decibels"""
        return float(error) * np.log(args.max_magnitude / args.min_magnitude) * 20 / np.log(10)

    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}')

    stream = batch_stream(args.mode, args.batch_size, args.n_samples, args.samplerate)

    for i, batch in zip(itertools.count(), stream):
        random_key, step_key = jax.random.split(random_key)
        model, opt_state, loss, aux = train_step(model, opt_state, batch, step_key)

        orig_audio(max_norm(np.asarray(batch[0])))
        recon_audio(max_norm(np.asarray(aux['recon'][0])))
        random_key, render_key = jax.random.split(random_key)
        random_audio(max_norm(np.asarray(render_random(model, render_key))))

        r = np.abs(np.asarray(aux['residual'][0]))
        residual_logger(r / (r.max() + 1e-12))

        # how many distinct frames the real decomposition selected, of
        # --n-steps; low means the events are piling up in the same places
        distinct = len(np.unique(np.asarray(aux['frames'][0])))
        line = (
            f'{i} loss {float(loss):.4f}  removed {-float(aux["greedy"]):.3f}'
            f'  info removed {-float(aux["information"]):.3f}'
            + (f'  final error {float(aux["final_error"]):.3f}' if args.energy_loss == 'final' else '')
            + f'  distinct frames {distinct}/{args.n_steps}')

        if args.synthetic:
            synthetic_audio(max_norm(np.asarray(aux['synthetic'][0])))
            synthetic_recon_audio(max_norm(np.asarray(aux['synthetic_recon'][0])))
            line += (
                f'  | synthetic removed {-float(aux["synthetic_greedy"]):.3f}'
                f'  hit rate {float(aux["hit_rate"]):.2f}'
                f'  position error {float(aux["position_error"]) * args.n_samples:.0f} samples'
                f'  fine error {float(aux["fine_error"]) * args.hop:.0f} samples'
                f'  magnitude error {magnitude_error_db(aux["magnitude_error"]):.1f} dB'
                f'  event error {float(aux["event_error"]):.3f}')

        print(line)


if __name__ == '__main__':
    main()
