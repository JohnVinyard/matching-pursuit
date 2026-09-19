"""
Minimal experiment: an "infinite dataset" of audio rendered from random
mass-spring-damper networks (strings, plates, or fully random graphs) driven
by a sparse control signal, plus a network that hears only the rendered
audio, guesses a mass/spring/force configuration, re-renders it through the
same (differentiable, jax.lax.scan-based) physics, and is trained purely on
a mel-spectrogram loss between the two audios.
"""
import argparse
import math
from typing import NamedTuple, Optional, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from conjure import LmdbCollection, loggers, serve_conjure
from conjure.logger import encode_audio


# ---------------------------------------------------------------------------
# physics: an arbitrary-dimension network of masses connected by springs,
# each spring's rest length fixed by the nodes' initial positions, plus a
# small per-node spring anchoring it back to its initial position (keeps the
# system bounded) and a sparse external control force.

class PhysicalParams(NamedTuple):
    positions: jnp.ndarray   # (N, D) rest positions
    stiffness: jnp.ndarray   # (N, N) symmetric, zero diagonal
    mass: jnp.ndarray        # (N,)
    anchor_k: jnp.ndarray    # (N,)
    damping: jnp.ndarray     # scalar in (0, 1)
    forces: jnp.ndarray      # (T, N) external force, applied along axis 0


def simulate(params: PhysicalParams, fixed_mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    n_nodes, dim = params.positions.shape
    resting = params.positions[:, None, :] - params.positions[None, :, :]
    mass = params.mass[:, None]
    if fixed_mask is None:
        fixed_mask = jnp.zeros((n_nodes,), dtype=bool)
    free = (~fixed_mask)[:, None]

    def step(carry, force_t):
        x, v = carry
        current = x[:, None, :] - x[None, :, :]
        d = current - resting
        spring_force = -jnp.einsum('ij,ijd->id', params.stiffness, d)
        anchor_force = -params.anchor_k[:, None] * (x - params.positions)
        external = jnp.zeros((n_nodes, dim)).at[:, 0].add(force_t)
        acc = (spring_force + anchor_force + external) / mass
        v = (v + acc) * params.damping * free
        x = x + v
        # mix all nodes' velocity rather than reading a single fixed
        # "pickup" node: some geometries (e.g. string endpoints) pin
        # particular nodes, whose velocity is always exactly zero, which
        # would otherwise make the recording silent by construction.
        return (x, v), jnp.mean(v[:, 0])

    init = (params.positions, jnp.zeros_like(params.positions))
    _, audio = jax.lax.scan(step, init, params.forces)
    return audio


# ---------------------------------------------------------------------------
# geometry "flavors": string (1d chain), plate (2d grid), random graph.
# positions/adjacency live in `dim`-dimensional space so the code is
# dimension-agnostic even though we default to 3.

class Geometry(NamedTuple):
    positions: jnp.ndarray  # (N, D)
    adjacency: jnp.ndarray  # (N, N) bool
    fixed: jnp.ndarray      # (N,) bool


def string_geometry(n_nodes: int, dim: int) -> Tuple[Geometry, int]:
    positions = np.zeros((n_nodes, dim), dtype=np.float32)
    positions[:, 0] = np.linspace(-1.0, 1.0, n_nodes)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    adjacency[np.arange(n_nodes - 1), np.arange(1, n_nodes)] = True
    adjacency |= adjacency.T
    fixed = np.zeros((n_nodes,), dtype=bool)
    fixed[[0, -1]] = True
    return Geometry(jnp.array(positions), jnp.array(adjacency), jnp.array(fixed)), n_nodes


def plate_geometry(n_nodes: int, dim: int) -> Tuple[Geometry, int]:
    rows = max(1, int(round(n_nodes ** 0.5)))
    cols = int(np.ceil(n_nodes / rows))
    n_nodes = rows * cols
    node_id = np.arange(n_nodes).reshape(rows, cols)

    positions = np.zeros((n_nodes, dim), dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, rows)
    ys = np.linspace(-1.0, 1.0, cols) if dim > 1 else np.zeros(cols)
    grid_r, grid_c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    positions[:, 0] = xs[grid_r].flatten()
    if dim > 1:
        positions[:, 1] = ys[grid_c].flatten()

    adjacency = np.zeros((n_nodes, n_nodes), dtype=bool)
    for r in range(rows):
        for c in range(cols):
            a = node_id[r, c]
            if r + 1 < rows:
                b = node_id[r + 1, c]
                adjacency[a, b] = adjacency[b, a] = True
            if c + 1 < cols:
                b = node_id[r, c + 1]
                adjacency[a, b] = adjacency[b, a] = True

    fixed = np.zeros((n_nodes,), dtype=bool)
    return Geometry(jnp.array(positions), jnp.array(adjacency), jnp.array(fixed)), n_nodes


def random_geometry(key: jnp.ndarray, n_nodes: int, dim: int, sparsity: float) -> Geometry:
    k_pos, k_edge = jax.random.split(key)
    positions = jax.random.uniform(k_pos, (n_nodes, dim), minval=-1.0, maxval=1.0)
    upper = jnp.triu(jax.random.bernoulli(k_edge, p=sparsity, shape=(n_nodes, n_nodes)), k=1)
    adjacency = upper | upper.T
    fixed = jnp.zeros((n_nodes,), dtype=bool)
    return Geometry(positions, adjacency, fixed)


# ---------------------------------------------------------------------------
# sampling random physical parameters ("the true dataset")

class PhysicsConfig(NamedTuple):
    n_nodes: int
    dim: int
    n_samples: int
    sparsity: float
    # NOTE: this simulator is a symplectic-Euler integration with a unit
    # timestep, which is only numerically stable while, roughly, the sum of
    # a node's (stiffness + anchor_k) / mass stays well under 4 (per-node
    # natural frequency omega = sqrt(k/m) radians/sample must stay under 2).
    # These ranges are kept conservative, accounting for nodes with several
    # neighbors, so random configurations don't blow up.
    stiffness_range: Tuple[float, float] = (0.001, 0.15)
    mass_range: Tuple[float, float] = (1.0, 5.0)
    anchor_range: Tuple[float, float] = (0.0, 0.05)
    # decay time is exponentially sensitive to damping as it approaches 1, so
    # this range is sampled log-uniformly over (1 - damping) (see
    # `damping_from_unit_interval`), giving even coverage from percussive
    # (a few ms) to sustained/ringing-through-the-clip (a few seconds).
    damping_range: Tuple[float, float] = (0.95, 0.9999)
    # a handful of discrete impulses per example, rather than a dense random
    # mask, so the system actually gets to ring and decay between hits
    # instead of being continuously re-driven.
    min_impulses: int = 1
    max_impulses: int = 4
    force_scale: float = 1.0


def damping_from_unit_interval(u: jnp.ndarray, damping_range: Tuple[float, float]) -> jnp.ndarray:
    """Maps u in [0, 1] to a damping value, log-uniform over (1 - damping)."""
    lo, hi = damping_range
    log_lo = math.log(1.0 - hi)
    log_hi = math.log(1.0 - lo)
    return 1.0 - jnp.exp(log_lo + u * (log_hi - log_lo))


def sample_true_example(
        key: jnp.ndarray,
        cfg: PhysicsConfig,
        string_geom: Geometry,
        plate_geom: Geometry,
        flavor_names: Tuple[str, ...],
):
    k_flavor, k_geo, k_stiff, k_mass, k_anchor, k_damp, k_fcount, k_fidx, k_fmag = jax.random.split(key, 9)

    # every candidate geometry is computed unconditionally (cheap relative to
    # the physics rollout below), then blended by a per-example random pick,
    # so each item in a batch can land on a different flavor.
    templates = {'string': string_geom, 'plate': plate_geom,
                 'random': random_geometry(k_geo, cfg.n_nodes, cfg.dim, cfg.sparsity)}
    candidates = [templates[name] for name in flavor_names]

    flavor_id = jax.random.randint(k_flavor, (), 0, len(candidates))
    positions, adjacency, fixed = candidates[0]
    for i in range(1, len(candidates)):
        match = flavor_id == i
        positions = jnp.where(match, candidates[i].positions, positions)
        adjacency = jnp.where(match, candidates[i].adjacency, adjacency)
        fixed = jnp.where(match, candidates[i].fixed, fixed)
    geometry = Geometry(positions, adjacency, fixed)

    lo, hi = cfg.stiffness_range
    raw_stiffness = jax.random.uniform(k_stiff, (cfg.n_nodes, cfg.n_nodes), minval=lo, maxval=hi)
    stiffness = jnp.where(geometry.adjacency, raw_stiffness, 0.0)
    stiffness = (stiffness + stiffness.T) * 0.5

    lo, hi = cfg.mass_range
    mass = jax.random.uniform(k_mass, (cfg.n_nodes,), minval=lo, maxval=hi)

    lo, hi = cfg.anchor_range
    anchor_k = jax.random.uniform(k_anchor, (cfg.n_nodes,), minval=lo, maxval=hi)

    damping = damping_from_unit_interval(jax.random.uniform(k_damp, ()), cfg.damping_range)

    # a random `n_impulses` in [min_impulses, max_impulses], placed at random
    # (time, node) locations; `max_impulses` stays a static shape and unused
    # slots are masked out, so this remains vmap/jit friendly.
    n_impulses = jax.random.randint(k_fcount, (), cfg.min_impulses, cfg.max_impulses + 1)
    flat_size = cfg.n_samples * cfg.n_nodes
    flat_idx = jax.random.randint(k_fidx, (cfg.max_impulses,), 0, flat_size)
    active = jnp.arange(cfg.max_impulses) < n_impulses
    magnitudes = jax.random.uniform(
        k_fmag, (cfg.max_impulses,), minval=-cfg.force_scale, maxval=cfg.force_scale) * active
    forces = jnp.zeros((flat_size,)).at[flat_idx].add(magnitudes).reshape(cfg.n_samples, cfg.n_nodes)

    params = PhysicalParams(
        positions=geometry.positions,
        stiffness=stiffness,
        mass=mass,
        anchor_k=anchor_k,
        damping=damping,
        forces=forces,
    )
    return params, geometry.fixed


def make_true_batch(
        key: jnp.ndarray,
        batch_size: int,
        cfg: PhysicsConfig,
        string_geom: Geometry,
        plate_geom: Geometry,
        flavor_names: Tuple[str, ...],
):
    keys = jax.random.split(key, batch_size)

    def one(k):
        params, fixed = sample_true_example(k, cfg, string_geom, plate_geom, flavor_names)
        return simulate(params, fixed_mask=fixed)

    return jax.vmap(one)(keys)


# ---------------------------------------------------------------------------
# differentiable mel spectrogram (pure numpy filterbank, jax STFT)

def build_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(hz_to_mel(0.0), hz_to_mel(sample_rate / 2.0), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb


def make_mel_fn(n_fft: int, hop_length: int, filterbank: jnp.ndarray):
    window = jnp.hanning(n_fft)

    def mel_spectrogram(audio: jnp.ndarray) -> jnp.ndarray:
        n_frames = 1 + (audio.shape[-1] - n_fft) // hop_length
        frame_idx = jnp.arange(n_fft)[None, :] + hop_length * jnp.arange(n_frames)[:, None]
        frames = audio[frame_idx] * window[None, :]
        mag = jnp.abs(jnp.fft.rfft(frames, axis=-1))
        mel = mag @ filterbank.T
        return jnp.log1p(mel).T  # (n_mels, n_frames)

    return mel_spectrogram


# ---------------------------------------------------------------------------
# model: hears a mel spectrogram, guesses a PhysicalParams configuration

class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv1d

    def __init__(self, in_c: int, out_c: int, kernel_size: int, key):
        self.conv = eqx.nn.Conv1d(in_c, out_c, kernel_size, padding='SAME', key=key)

    def __call__(self, x):
        return jax.nn.gelu(self.conv(x))


class PhysicsGuesser(eqx.Module):
    blocks: list
    global_head: eqx.nn.MLP
    force_head: eqx.nn.Conv1d
    n_nodes: int = eqx.field(static=True)
    dim: int = eqx.field(static=True)
    n_samples: int = eqx.field(static=True)
    hop_length: int = eqx.field(static=True)
    stiffness_range: Tuple[float, float] = eqx.field(static=True)
    mass_range: Tuple[float, float] = eqx.field(static=True)
    anchor_range: Tuple[float, float] = eqx.field(static=True)
    damping_range: Tuple[float, float] = eqx.field(static=True)

    def __init__(self, n_mels, n_nodes, dim, n_samples, hop_length, hidden, n_layers, cfg, key):
        keys = jax.random.split(key, n_layers + 2)
        channels = [n_mels] + [hidden] * n_layers
        self.blocks = [ConvBlock(channels[i], channels[i + 1], 5, keys[i]) for i in range(n_layers)]

        n_pairs = n_nodes * (n_nodes - 1) // 2
        global_out = n_nodes * dim + n_pairs + n_nodes + n_nodes + 1
        self.global_head = eqx.nn.MLP(hidden, global_out, width_size=hidden, depth=2, key=keys[-2])
        self.force_head = eqx.nn.Conv1d(hidden, n_nodes, 1, key=keys[-1])

        self.n_nodes, self.dim = n_nodes, dim
        self.n_samples, self.hop_length = n_samples, hop_length
        self.stiffness_range = cfg.stiffness_range
        self.mass_range = cfg.mass_range
        self.anchor_range = cfg.anchor_range
        self.damping_range = cfg.damping_range

    def __call__(self, mel: jnp.ndarray) -> PhysicalParams:
        x = mel
        for block in self.blocks:
            x = block(x)

        g = self.global_head(jnp.mean(x, axis=-1))
        n, d = self.n_nodes, self.dim
        n_pairs = n * (n - 1) // 2

        i = 0
        positions = g[i:i + n * d].reshape(n, d); i += n * d

        lo, hi = self.stiffness_range
        stiff_flat = lo + jax.nn.sigmoid(g[i:i + n_pairs]) * (hi - lo); i += n_pairs

        lo, hi = self.mass_range
        mass = lo + jax.nn.sigmoid(g[i:i + n]) * (hi - lo); i += n

        lo, hi = self.anchor_range
        anchor_k = lo + jax.nn.sigmoid(g[i:i + n]) * (hi - lo); i += n

        damping = damping_from_unit_interval(jax.nn.sigmoid(g[i]), self.damping_range)

        stiffness = jnp.zeros((n, n))
        iu = jnp.triu_indices(n, k=1)
        stiffness = stiffness.at[iu].set(stiff_flat)
        stiffness = stiffness + stiffness.T

        forces_ctrl = jnp.tanh(self.force_head(x))  # (n, n_frames)
        forces = jnp.repeat(forces_ctrl, self.hop_length, axis=-1)
        pad = self.n_samples - forces.shape[-1]
        forces = forces[:, :self.n_samples] if pad <= 0 else jnp.pad(forces, ((0, 0), (0, pad)))

        return PhysicalParams(
            positions=positions,
            stiffness=stiffness,
            mass=mass,
            anchor_k=anchor_k,
            damping=damping,
            forces=forces.T,
        )


# ---------------------------------------------------------------------------
# training

def max_norm(audio: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return audio / (np.abs(audio).max() + epsilon)


def make_loss_fn(model_apply, mel_fn):
    def loss_fn(model, true_audio):
        mel_true = jax.vmap(mel_fn)(true_audio)
        pred_params = jax.vmap(model_apply, in_axes=(None, 0))(model, mel_true)
        pred_audio = jax.vmap(lambda p: simulate(p, fixed_mask=None))(pred_params)
        mel_pred = jax.vmap(mel_fn)(pred_audio)
        loss = jnp.mean(jnp.abs(mel_true - mel_pred))
        return loss, pred_audio

    return loss_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-nodes', type=int, default=12)
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--flavors', nargs='+', choices=['string', 'plate', 'random'],
                         default=['string', 'plate', 'random'],
                         help='geometry flavors to mix; each example in a batch independently '
                              'picks one at random')
    parser.add_argument('--sparsity', type=float, default=0.25, help='edge probability, random flavor only')
    parser.add_argument('--min-impulses', type=int, default=1, help='min discrete control impulses per example')
    parser.add_argument('--max-impulses', type=int, default=4, help='max discrete control impulses per example')
    parser.add_argument('--n-samples', type=int, default=22050, help='audio length in samples')
    parser.add_argument('--sample-rate', type=int, default=22050)
    parser.add_argument('--n-fft', type=int, default=512)
    parser.add_argument('--hop-length', type=int, default=128)
    parser.add_argument('--n-mels', type=int, default=64)
    parser.add_argument('--hidden-channels', type=int, default=32)
    parser.add_argument('--n-conv-layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--n-steps', type=int, default=2000)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--sample-every', type=int, default=1, help='log audio to conjure every N steps')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--conjure-collection', type=str, default='claude_jax_physical')
    parser.add_argument('--conjure-port', type=int, default=9998)
    parser.add_argument('--wipe-conjure-data', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    collection = LmdbCollection(path=args.conjure_collection)
    if args.wipe_conjure_data:
        print('Wiping previous experiment data')
        collection.destroy()
        collection = LmdbCollection(path=args.conjure_collection)

    true_audio_logger, pred_audio_logger = loggers(
        ['true', 'pred'],
        'audio/wav',
        lambda x: encode_audio(x, samplerate=args.sample_rate),
        collection)

    serve_conjure(
        [true_audio_logger, pred_audio_logger],
        port=args.conjure_port,
        n_workers=1,
        web_components_version='0.0.101')

    # plate geometry rounds n_nodes up to fit a rectangular grid; that
    # adjusted count becomes the canonical node count so every flavor
    # (string/plate/random) shares the same shapes and can be mixed freely
    # within a batch.
    plate_geom, n_nodes = plate_geometry(args.n_nodes, args.dim)
    string_geom, _ = string_geometry(n_nodes, args.dim)
    flavor_names = tuple(args.flavors)

    cfg = PhysicsConfig(
        n_nodes=n_nodes, dim=args.dim, n_samples=args.n_samples, sparsity=args.sparsity,
        min_impulses=args.min_impulses, max_impulses=args.max_impulses,
    )

    filterbank = jnp.array(build_mel_filterbank(args.sample_rate, args.n_fft, args.n_mels))
    mel_fn = make_mel_fn(args.n_fft, args.hop_length, filterbank)

    key = jax.random.key(args.seed)
    key, model_key = jax.random.split(key)
    model = PhysicsGuesser(
        n_mels=args.n_mels, n_nodes=n_nodes, dim=args.dim,
        n_samples=args.n_samples, hop_length=args.hop_length,
        hidden=args.hidden_channels, n_layers=args.n_conv_layers,
        cfg=cfg, key=model_key,
    )

    optimizer = optax.adam(args.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_fn = make_loss_fn(lambda m, mel: m(mel), mel_fn)

    @eqx.filter_jit
    def train_step(model, opt_state, key):
        true_audio = make_true_batch(key, args.batch_size, cfg, string_geom, plate_geom, flavor_names)
        (loss, pred_audio), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model, true_audio)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, true_audio, pred_audio

    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}')
    print(f'flavors={flavor_names} n_nodes={n_nodes} dim={args.dim}')

    for step in range(1, args.n_steps + 1):
        key, step_key = jax.random.split(key)
        model, opt_state, loss, true_audio, pred_audio = train_step(model, opt_state, step_key)

        if step % args.log_every == 0 or step == 1:
            print(f'step {step:5d} loss {float(loss):.5f}')

        if step % args.sample_every == 0:
            true_audio_logger(max_norm(np.asarray(true_audio[0])))
            pred_audio_logger(max_norm(np.asarray(pred_audio[0])))


if __name__ == '__main__':
    main()
