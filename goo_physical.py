"""
A hybrid of `goo.py` and `claude_jax_physical.py`, in jax.

A mass-spring network is simulated at a *reduced* rate (sample_rate / block),
as in `goo.py`, and each node is confined to a sphere around its rest
position, as in `claude_jax_physical.py`: when a node leaves its sphere it is
clamped back onto the surface and the outward part of its velocity is
reflected (with restitution < 1). That's the source of nonlinearity.

Output is the sum of two parts:

  - low frequencies: node displacements read through per-node "mics"
    (N, D), then FFT-upsampled to the full sample rate.

  - high frequencies: a pluggable `HFGenerator`, which sees the whole
    low-rate simulation state (`SimOutput`: displacement, velocity, and the
    wall-collision events) and renders full-rate audio from it. See
    `HF_GENERATORS` for what's available; `Stack` sums several.

Run directly to render a few random instruments (full mix + lf/hf stems) to
wav and check that gradients through the whole thing are finite.
"""
import argparse
import math
import os
from typing import Dict, NamedTuple, Optional, Sequence, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from claude_jax_physical import (
    Geometry, build_mel_filterbank, damping_from_unit_interval, log_uniform_interval,
    make_mel_fn, plate_geometry, random_geometry, string_geometry,
)


# ---------------------------------------------------------------------------
# configuration

class GooConfig(NamedTuple):
    n_nodes: int
    dim: int
    n_samples: int        # full rate
    block: int            # simulation runs at sample_rate / block
    sample_rate: int = 22050
    restitution: float = 0.5
    # the ranges below are used both to sample synthetic instruments and to
    # bound the encoder's predictions. Same reasoning as
    # claude_jax_physical.PhysicsConfig; units are per *low-rate* step, so
    # pitches scale down with `block`.
    stiffness_range: Tuple[float, float] = (0.001, 0.15)
    mass_range: Tuple[float, float] = (1.0, 5.0)
    anchor_range: Tuple[float, float] = (0.0, 0.05)
    damping_range: Tuple[float, float] = (0.99, 0.9995)
    radius_range: Tuple[float, float] = (0.02, 2.0)
    force_scale: float = 1.0
    # synthetic instruments only
    sparsity: float = 0.25
    max_impulses: int = 4
    # synthetic impulses land on multiples of this many low-rate steps (set it
    # to the encoder's control rate, so synthetic targets are reachable)
    control_stride: int = 1

    @property
    def n_steps(self) -> int:
        return self.n_samples // self.block

    @property
    def sim_nyquist(self) -> float:
        return self.sample_rate / self.block / 2


# ---------------------------------------------------------------------------
# low-rate physics

class Physics(NamedTuple):
    positions: jnp.ndarray    # (N, D) rest positions
    stiffness: jnp.ndarray    # (N, N) symmetric, zero diagonal
    mass: jnp.ndarray         # (N,)
    anchor_k: jnp.ndarray     # (N,)
    damping: jnp.ndarray      # () in (0, 1)
    radius: jnp.ndarray       # (N,) confining sphere radius per node
    forces: jnp.ndarray       # (T_lo, N, D) external control force
    mics: jnp.ndarray         # (N, D) displacement -> lf audio


class SimOutput(NamedTuple):
    displacement: jnp.ndarray  # (T_lo, N, D)
    velocity: jnp.ndarray      # (T_lo, N, D), after any bounce
    impulse: jnp.ndarray       # (T_lo, N, D) velocity change from wall bounces (zero elsewhere)
    crossing: jnp.ndarray      # (T_lo, N) fraction of the step at which the wall was crossed


def simulate(physics: Physics, fixed_mask: Optional[jnp.ndarray] = None, restitution: float = 0.5) -> SimOutput:
    """`claude_jax_physical.simulate`, but with D-dim forces, and returning
    per-node state and collision events instead of a mixed-down signal."""
    n_nodes, dim = physics.positions.shape
    rest = physics.positions
    resting = rest[:, None, :] - rest[None, :, :]
    mass = physics.mass[:, None]
    radius = physics.radius[:, None]
    if fixed_mask is None:
        fixed_mask = jnp.zeros((n_nodes,), dtype=bool)
    free = (~fixed_mask.astype(bool))[:, None]

    def norm(a):
        # eps inside the sqrt: see claude_jax_physical.simulate (fixed nodes
        # sit at exactly zero displacement and would otherwise NaN the grad)
        return jnp.sqrt(jnp.sum(a ** 2, axis=-1, keepdims=True) + 1e-12)

    def step(carry, force_t):
        x, v = carry
        current = x[:, None, :] - x[None, :, :]
        spring_force = -jnp.einsum('ij,ijd->id', physics.stiffness, current - resting)
        anchor_force = -physics.anchor_k[:, None] * (x - rest)
        acc = (spring_force + anchor_force + force_t) / mass
        v = (v + acc) * physics.damping * free

        prev_dist = norm(x - rest)
        x = x + v

        disp = x - rest
        dist = norm(disp)
        normal = disp / dist
        over = dist > radius

        # where within this step the wall was crossed, assuming the distance
        # from rest changed linearly over the step. prev_dist <= radius (it
        # was clamped last step), so when `over`, the denominator is > 0; the
        # floor only keeps the untaken branch of `where` finite.
        crossing = jnp.clip((radius - prev_dist) / jnp.maximum(dist - prev_dist, 1e-6), 0.0, 1.0)

        x = jnp.where(over, rest + normal * radius, x)
        v_radial = jnp.sum(v * normal, axis=-1, keepdims=True)
        hit = over & (v_radial > 0)
        dv = (1 + restitution) * v_radial * normal
        v = jnp.where(hit, v - dv, v)
        impulse = jnp.where(hit, dv, 0.0)

        return (x, v), (x - rest, v, impulse, crossing[:, 0])

    init = (rest, jnp.zeros_like(rest))
    _, out = jax.lax.scan(step, init, physics.forces)
    return SimOutput(*out)


# ---------------------------------------------------------------------------
# shared dsp helpers

def fft_upsample(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Band-limited upsampling along the last axis by zero-padding the spectrum."""
    n = x.shape[-1]
    spec = jnp.fft.rfft(x, axis=-1)
    n_hi = n * factor
    pad = (n_hi // 2 + 1) - spec.shape[-1]
    spec = jnp.pad(spec, [(0, 0)] * (x.ndim - 1) + [(0, pad)])
    return jnp.fft.irfft(spec, n=n_hi, axis=-1) * factor


def linear_upsample(x: jnp.ndarray, factor: int) -> jnp.ndarray:
    """Linear interpolation along the last axis; never overshoots, so it's
    the right choice for envelopes that must stay non-negative."""
    return jax.image.resize(x, x.shape[:-1] + (x.shape[-1] * factor,), method='linear')


def place_impulses(impulse: jnp.ndarray, crossing: jnp.ndarray, block: int) -> jnp.ndarray:
    """(T_lo, N, D) low-rate collision impulses -> (N, D, T_hi) at full rate,
    each placed at its sub-step crossing time and linearly split between the
    two neighboring samples (so it's differentiable w.r.t. timing)."""
    n_lo, n_nodes, dim = impulse.shape
    n_hi = n_lo * block

    # the wall was crossed between the positions recorded at steps t-1 and t
    t = jnp.arange(n_lo, dtype=jnp.float32)[:, None] - 1.0 + crossing
    pos = jnp.clip(t * block, 0.0, n_hi - 1.0)
    idx = jnp.floor(pos).astype(jnp.int32)
    a = (pos - idx)[..., None]

    node = jnp.broadcast_to(jnp.arange(n_nodes)[None, :, None], impulse.shape)
    axis = jnp.broadcast_to(jnp.arange(dim)[None, None, :], impulse.shape)
    idx = jnp.broadcast_to(idx[..., None], impulse.shape)

    out = jnp.zeros((n_nodes, dim, n_hi + 1))
    out = out.at[node, axis, idx].add(impulse * (1 - a))
    out = out.at[node, axis, idx + 1].add(impulse * a)
    return out[..., :n_hi]


def convolve_and_sum(signals: jnp.ndarray, responses: jnp.ndarray, n_out: int) -> jnp.ndarray:
    """Convolves each (..., T) signal with its matching (..., L) response and
    sums them all. Convolution is linear, so the sum happens in the frequency
    domain, with a single inverse transform."""
    size = signals.shape[-1] + responses.shape[-1]
    n_fft = 1 << int(math.ceil(math.log2(size)))
    spec = jnp.fft.rfft(signals, n=n_fft) * jnp.fft.rfft(responses, n=n_fft)
    spec = jnp.sum(spec.reshape(-1, spec.shape[-1]), axis=0)
    return jnp.fft.irfft(spec, n=n_fft)[:n_out]


def damped_modes(freqs: jnp.ndarray, decays: jnp.ndarray, sample_rate: int, size: int) -> jnp.ndarray:
    """(..., K) frequencies (Hz) and decay times (s) -> (..., K, size)
    impulse responses of damped harmonic oscillators."""
    t = jnp.arange(size) / sample_rate
    return jnp.exp(-t / decays[..., None]) * jnp.sin(2 * jnp.pi * freqs[..., None] * t)


# ---------------------------------------------------------------------------
# pluggable high-frequency generators
#
# Each generator is an eqx.Module, so besides the per-example parameters the
# encoder predicts for it, it can hold its own *global*, learned weights
# (e.g. `EnergyNoise.filters`), trained by the real-audio loss.
#
# The contract:
#   n_params            how many raw, unconstrained values per example the
#                       encoder must produce for it (may be 0)
#   unpack(raw)         (n_params,) raw values -> the generator's params
#                       (any pytree), mapped into sensible ranges
#   sample(key)         random params for synthetic training targets; should
#                       cover (roughly) the same space `unpack` can reach
#   __call__(params, sim, key) -> (T_hi,) full-rate audio, given the low-rate
#                       simulation state. `key` is for any noise it needs.

class HFGenerator(eqx.Module):
    n_nodes: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    block: int = eqx.field(static=True)
    sample_rate: int = eqx.field(static=True)

    @property
    def n_params(self) -> int:
        raise NotImplementedError

    def unpack(self, raw: jnp.ndarray):
        raise NotImplementedError

    def sample(self, key):
        raise NotImplementedError

    def __call__(self, params, sim: SimOutput, key) -> jnp.ndarray:
        raise NotImplementedError


class NoHF(HFGenerator):
    """Low-frequency physics only; a baseline."""

    def __init__(self, n_nodes, dim, block, sample_rate):
        self.n_nodes, self.dim, self.block, self.sample_rate = n_nodes, dim, block, sample_rate

    @property
    def n_params(self) -> int:
        return 0

    def unpack(self, raw):
        return ()

    def sample(self, key):
        return ()

    def __call__(self, params, sim, key):
        return jnp.zeros((sim.displacement.shape[0] * self.block,))


class CollisionModesParams(NamedTuple):
    freqs: jnp.ndarray     # (N, K) Hz
    decays: jnp.ndarray    # (N, K) seconds
    coupling: jnp.ndarray  # (N, D, K) impact direction -> mode excitation
    gain: jnp.ndarray      # ()


class CollisionModes(HFGenerator):
    """
    Wall collisions are hard discontinuities, so they're broadband, and at the
    reduced simulation rate everything above its nyquist is lost (or
    aliased). Each collision's impulse vector is placed on the full-rate
    timeline at its sub-step crossing time and rings a per-node bank of K
    damped harmonic oscillators living above the simulation's nyquist. The
    coupling is per (node, axis), so *where* a node hits its sphere changes
    which modes ring.
    """
    n_modes: int = eqx.field(static=True)
    response_size: int = eqx.field(static=True)
    freq_range: Tuple[float, float] = eqx.field(static=True)
    decay_range: Tuple[float, float] = eqx.field(static=True)
    gain_range: Tuple[float, float] = eqx.field(static=True)

    def __init__(self, n_nodes, dim, block, sample_rate, n_modes: int = 8, response_size: int = 4096,
                 decay_range=(0.003, 0.15), gain_range=(0.01, 10.0)):
        self.n_nodes, self.dim, self.block, self.sample_rate = n_nodes, dim, block, sample_rate
        self.n_modes, self.response_size = n_modes, response_size
        self.freq_range = (0.8 * sample_rate / block / 2, 0.45 * sample_rate)
        self.decay_range = decay_range
        self.gain_range = gain_range

    @property
    def n_params(self) -> int:
        n, d, k = self.n_nodes, self.dim, self.n_modes
        return n * k + n * k + n * d * k + 1

    def _split(self, raw):
        n, d, k = self.n_nodes, self.dim, self.n_modes
        sizes = [n * k, n * k, n * d * k, 1]
        parts = jnp.split(raw, np.cumsum(sizes)[:-1])
        return parts[0].reshape(n, k), parts[1].reshape(n, k), parts[2].reshape(n, d, k), parts[3][0]

    def unpack(self, raw):
        f, dec, c, g = self._split(raw)
        return CollisionModesParams(
            freqs=log_uniform_interval(jax.nn.sigmoid(f), self.freq_range),
            decays=log_uniform_interval(jax.nn.sigmoid(dec), self.decay_range),
            coupling=jnp.tanh(c),
            gain=log_uniform_interval(jax.nn.sigmoid(g), self.gain_range))

    def sample(self, key):
        n, d, k = self.n_nodes, self.dim, self.n_modes
        k1, k2, k3, k4 = jax.random.split(key, 4)
        return CollisionModesParams(
            freqs=log_uniform_interval(jax.random.uniform(k1, (n, k)), self.freq_range),
            decays=log_uniform_interval(jax.random.uniform(k2, (n, k)), self.decay_range),
            coupling=jax.random.uniform(k3, (n, d, k), minval=-1.0, maxval=1.0),
            gain=log_uniform_interval(jax.random.uniform(k4, ()), self.gain_range))

    def __call__(self, params: CollisionModesParams, sim, key):
        excitation = place_impulses(sim.impulse, sim.crossing, self.block)        # (N, D, T)
        modes = damped_modes(params.freqs, params.decays, self.sample_rate, self.response_size)  # (N, K, L)
        responses = jnp.einsum('ndk,nkl->ndl', params.coupling, modes)            # (N, D, L)
        return convolve_and_sum(excitation, responses, excitation.shape[-1]) * params.gain


class EnergyNoiseParams(NamedTuple):
    mixture: jnp.ndarray   # (N, F) per-node weights over the learned filter bank
    gain: jnp.ndarray      # ()


class EnergyNoise(HFGenerator):
    """
    `goo.py`'s high-frequency path: each node's speed (linearly upsampled)
    is the envelope for white noise, which is then filtered by a per-node
    mixture of a *globally learned* filter bank. Continuous excitation,
    rather than `CollisionModes`' discrete one.
    """
    filters: jnp.ndarray   # (F, L), learned
    gain_range: Tuple[float, float] = eqx.field(static=True)

    def __init__(self, n_nodes, dim, block, sample_rate, key, n_filters: int = 32, filter_size: int = 512,
                 gain_range=(0.01, 10.0)):
        self.n_nodes, self.dim, self.block, self.sample_rate = n_nodes, dim, block, sample_rate
        # decaying noise bursts; the network learns their spectra
        env = jnp.exp(-jnp.linspace(0, 8, filter_size))
        filters = jax.random.normal(key, (n_filters, filter_size)) * env
        self.filters = filters / jnp.linalg.norm(filters, axis=-1, keepdims=True)
        self.gain_range = gain_range

    @property
    def n_params(self) -> int:
        return self.n_nodes * self.filters.shape[0] + 1

    def unpack(self, raw):
        mix = raw[:-1].reshape(self.n_nodes, self.filters.shape[0])
        return EnergyNoiseParams(
            mixture=jax.nn.softmax(mix, axis=-1),
            gain=log_uniform_interval(jax.nn.sigmoid(raw[-1]), self.gain_range))

    def sample(self, key):
        k1, k2 = jax.random.split(key)
        mix = jax.random.normal(k1, (self.n_nodes, self.filters.shape[0])) * 2.0
        return EnergyNoiseParams(
            mixture=jax.nn.softmax(mix, axis=-1),
            gain=log_uniform_interval(jax.random.uniform(k2, ()), self.gain_range))

    def __call__(self, params: EnergyNoiseParams, sim, key):
        speed = jnp.sqrt(jnp.sum(sim.velocity ** 2, axis=-1) + 1e-12).T            # (N, T_lo)
        envelope = linear_upsample(speed, self.block)                               # (N, T)
        noise = jax.random.uniform(key, envelope.shape, minval=-1.0, maxval=1.0)
        filters = self.filters / (jnp.linalg.norm(self.filters, axis=-1, keepdims=True) + 1e-8)
        responses = params.mixture @ filters                                         # (N, L)
        return convolve_and_sum(envelope * noise, responses, envelope.shape[-1]) * params.gain


class Stack(HFGenerator):
    """Sum of several generators; their per-example params are concatenated."""
    generators: Tuple[HFGenerator, ...]

    def __init__(self, generators: Sequence[HFGenerator]):
        g = generators[0]
        self.n_nodes, self.dim, self.block, self.sample_rate = g.n_nodes, g.dim, g.block, g.sample_rate
        self.generators = tuple(generators)

    @property
    def n_params(self) -> int:
        return sum(g.n_params for g in self.generators)

    def unpack(self, raw):
        sizes = [g.n_params for g in self.generators]
        parts = jnp.split(raw, np.cumsum(sizes)[:-1])
        return tuple(g.unpack(p) for g, p in zip(self.generators, parts))

    def sample(self, key):
        keys = jax.random.split(key, len(self.generators))
        return tuple(g.sample(k) for g, k in zip(self.generators, keys))

    def __call__(self, params, sim, key):
        keys = jax.random.split(key, len(self.generators))
        return sum(g(p, sim, k) for g, p, k in zip(self.generators, params, keys))


HF_GENERATORS = ('none', 'collision', 'energy')


def make_hf_generator(names: Sequence[str], cfg: GooConfig, key, **kwargs) -> HFGenerator:
    """`names`: one or more of HF_GENERATORS; more than one are `Stack`ed."""
    common = dict(n_nodes=cfg.n_nodes, dim=cfg.dim, block=cfg.block, sample_rate=cfg.sample_rate)
    gens = []
    for name, k in zip(names, jax.random.split(key, len(names))):
        if name == 'none':
            gens.append(NoHF(**common))
        elif name == 'collision':
            gens.append(CollisionModes(**common, n_modes=kwargs.get('n_modes', 8)))
        elif name == 'energy':
            gens.append(EnergyNoise(**common, key=k, n_filters=kwargs.get('n_filters', 32)))
        else:
            raise ValueError(f'unknown hf generator {name!r}; choose from {HF_GENERATORS}')
    return gens[0] if len(gens) == 1 else Stack(gens)


# ---------------------------------------------------------------------------
# rendering

class Render(NamedTuple):
    audio: jnp.ndarray         # (T_hi,)
    lf: jnp.ndarray
    hf: jnp.ndarray
    n_collisions: jnp.ndarray  # (node, step) pairs with a bounce


def render(
        physics: Physics,
        hf: HFGenerator,
        hf_params,
        key,
        block: int,
        fixed_mask: Optional[jnp.ndarray] = None,
        restitution: float = 0.5) -> Render:
    sim = simulate(physics, fixed_mask, restitution)
    lf = fft_upsample(jnp.einsum('tnd,nd->t', sim.displacement, physics.mics), block)
    h = hf(hf_params, sim, key)
    n_collisions = jnp.sum(jnp.any(sim.impulse != 0, axis=-1))
    return Render(lf + h, lf, h, n_collisions)


# ---------------------------------------------------------------------------
# random instruments: synthetic training targets, and for listening

class Geometries(NamedTuple):
    """Every flavor shares one node count (the plate's, which is rounded to
    a grid), so a batch can mix flavors freely."""
    string: Geometry
    plate: Geometry
    n_nodes: int


def make_geometries(n_nodes: int, dim: int) -> Geometries:
    plate, n = plate_geometry(n_nodes, dim)
    string, _ = string_geometry(n, dim)
    return Geometries(string, plate, n)


FLAVORS = ('string', 'plate', 'random')


def sample_geometry(key, cfg: GooConfig, geometries: Geometries, flavors: Sequence[str]) -> Geometry:
    k_flavor, k_geo = jax.random.split(key)
    templates = {'string': geometries.string, 'plate': geometries.plate,
                 'random': random_geometry(k_geo, cfg.n_nodes, cfg.dim, cfg.sparsity)}
    candidates = [templates[f] for f in flavors]
    pick = jax.random.randint(k_flavor, (), 0, len(candidates))
    # (jnp.select's default fill promotes bool to int, so cast back: an int
    # `fixed` mask makes `~fixed` a bitwise not, i.e. -1, in `simulate`)
    return jax.tree.map(
        lambda *xs: jnp.select([pick == i for i in range(len(xs))], xs).astype(xs[0].dtype), *candidates)


def sample_physics(key, cfg: GooConfig, geometry: Geometry) -> Physics:
    ks = jax.random.split(key, 10)
    n, d = cfg.n_nodes, cfg.dim

    stiffness = jax.random.uniform(ks[0], (n, n), minval=cfg.stiffness_range[0], maxval=cfg.stiffness_range[1])
    stiffness = jnp.where(geometry.adjacency, stiffness, 0.0)
    stiffness = (stiffness + stiffness.T) * 0.5

    mass = jax.random.uniform(ks[1], (n,), minval=cfg.mass_range[0], maxval=cfg.mass_range[1])
    anchor_k = jax.random.uniform(ks[2], (n,), minval=cfg.anchor_range[0], maxval=cfg.anchor_range[1])
    damping = damping_from_unit_interval(jax.random.uniform(ks[3], ()), cfg.damping_range)
    # one radius scale per instrument (so some are rattly everywhere and some
    # barely collide), with a little per-node spread
    radius = log_uniform_interval(jax.random.uniform(ks[4], ()), cfg.radius_range) \
        * jnp.exp(jax.random.normal(ks[5], (n,)) * 0.25)
    radius = jnp.clip(radius, *cfg.radius_range)

    n_slots = cfg.n_steps // cfg.control_stride
    slot_t = jax.random.randint(ks[6], (cfg.max_impulses,), 0, n_slots // 2) * cfg.control_stride
    slot_n = jax.random.randint(ks[7], (cfg.max_impulses,), 0, n)
    vec = jax.random.uniform(ks[8], (cfg.max_impulses, d), minval=-1.0, maxval=1.0) * cfg.force_scale
    forces = jnp.zeros((cfg.n_steps, n, d)).at[slot_t, slot_n].add(vec)

    mics = jax.random.uniform(ks[9], (n, d), minval=-1.0, maxval=1.0)

    return stabilize(Physics(geometry.positions, stiffness, mass, anchor_k, damping, radius, forces, mics))


def stabilize(physics: Physics, limit: float = 3.5) -> Physics:
    """
    The integrator (symplectic Euler, unit step) is stable while the
    stiffest mode's omega^2 = lambda_max(M^-1 (L + A)) stays under 4. By
    Gershgorin, lambda_max <= max_i (2 * sum_j k_ij + anchor_i) / m_i, so if
    that bound exceeds `limit`, scale the springs down until it doesn't.
    Matters for dense, predicted graphs (16 nodes * 15 springs * 0.15 can
    exceed it); sparse sampled ones are rarely touched.
    """
    bound = jnp.max((2 * jnp.sum(physics.stiffness, axis=-1) + physics.anchor_k) / physics.mass)
    scale = jnp.minimum(1.0, limit / (bound + 1e-8))
    return physics._replace(stiffness=physics.stiffness * scale)


def max_norm(x: np.ndarray) -> np.ndarray:
    return x / (np.abs(x).max() + 1e-8)


def main():
    import soundfile as sf

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--flavors', nargs='+', choices=FLAVORS, default=['plate'])
    parser.add_argument('--hf', nargs='+', choices=HF_GENERATORS, default=['collision'],
                        help='high-frequency generator(s); several are summed')
    parser.add_argument('--n-nodes', type=int, default=16)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--block', type=int, default=8, help='simulation runs at sample_rate / block')
    parser.add_argument('--n-samples', type=int, default=2 ** 16)
    parser.add_argument('--sample-rate', type=int, default=22050)
    parser.add_argument('--restitution', type=float, default=0.5)
    parser.add_argument('--n-examples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='goo_physical_out')
    parser.add_argument('--grad-check', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    geometries = make_geometries(args.n_nodes, args.dim)
    cfg = GooConfig(
        n_nodes=geometries.n_nodes, dim=args.dim, n_samples=args.n_samples, block=args.block,
        sample_rate=args.sample_rate, restitution=args.restitution)
    key = jax.random.key(args.seed)
    key, hf_key = jax.random.split(key)
    hf = make_hf_generator(args.hf, cfg, hf_key)

    print(f'simulating at {cfg.sample_rate / cfg.block:.0f} Hz ({cfg.n_steps} steps), '
          f'{cfg.n_nodes} nodes, flavors={args.flavors}, hf={args.hf} ({hf.n_params} params/example)')

    @eqx.filter_jit
    def random_instrument(hf, key):
        k_geo, k_phys, k_hf, k_noise = jax.random.split(key, 4)
        geometry = sample_geometry(k_geo, cfg, geometries, args.flavors)
        physics = sample_physics(k_phys, cfg, geometry)
        hf_params = hf.sample(k_hf)
        return physics, hf_params, geometry.fixed, k_noise

    @eqx.filter_jit
    def run(hf, physics, hf_params, fixed, key):
        return render(physics, hf, hf_params, key, cfg.block, fixed, cfg.restitution)

    os.makedirs(args.out_dir, exist_ok=True)
    for i in range(args.n_examples):
        key, k = jax.random.split(key)
        out = run(hf, *random_instrument(hf, k))
        lf, h = np.asarray(out.lf), np.asarray(out.hf)
        rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
        print(f'[{i}] collisions={int(out.n_collisions)} rms lf={rms(lf):.4g} hf={rms(h):.4g}')
        # stems are normalized separately so both are audible; the mix as-is
        for name, x in [('mix', np.asarray(out.audio)), ('lf', lf), ('hf', h)]:
            sf.write(os.path.join(args.out_dir, f'{i:02d}_{name}.wav'), max_norm(x), cfg.sample_rate)
    print(f'wrote wavs to {os.path.abspath(args.out_dir)}')

    if args.grad_check:
        mel_fn = make_mel_fn(512, 128, jnp.array(build_mel_filterbank(cfg.sample_rate, 512, 64)))
        key, k1, k2 = jax.random.split(key, 3)
        target = run(hf, *random_instrument(hf, k1)).audio
        physics, hf_params, fixed, noise_key = random_instrument(hf, k2)

        def loss(trainable):
            hf_, physics_, hf_params_ = trainable
            out = render(physics_, hf_, hf_params_, noise_key, cfg.block, fixed, cfg.restitution)
            return jnp.mean(jnp.abs(mel_fn(out.audio) - mel_fn(target)))

        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))((hf, physics, hf_params))
        print(f'grad check: loss={float(value):.4f}')
        leaves = jax.tree_util.tree_leaves_with_path(eqx.filter(grads, eqx.is_array))
        for path, g in leaves:
            g = np.asarray(g)
            print(f'  {jax.tree_util.keystr(path):40s} finite={bool(np.all(np.isfinite(g)))} '
                  f'|g|={float(np.linalg.norm(g)):.3e}')


if __name__ == '__main__':
    main()
