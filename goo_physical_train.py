"""
Train an encoder that hears audio and guesses a `goo_physical` instrument
(the spring network's physics, a sparse control signal, and the pluggable
high-frequency generator's parameters), which is then re-rendered through the
same differentiable synth and compared in a (mel / multiband) spectral space.

Two kinds of targets, as in `jax_ae.py` and `claude_jax_physical.py`:

  - real audio (`--mode train` streams it; `--mode overfit` repeats one
    segment)
  - synthetic audio: random instruments drawn from `goo_physical`'s sampler
    (`--flavors`), rendered by the *current* model's synth, so every target is
    exactly reachable. Turned off with `--synthetic-coeff 0`.

The high-frequency generator(s) are chosen with `--hf` (several are summed);
any global weights they own (e.g. `energy`'s filter bank) are trained by the
real-audio loss alongside the encoder.

Rest positions aren't predicted: springs in `goo_physical.simulate` are
linear in displacement and the spheres are centered on the rest positions, so
positions have no effect; only which pairs are connected (and how stiffly)
does. The encoder predicts a dense, symmetric stiffness matrix.
"""
import argparse
import itertools
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from conjure import LmdbCollection, loggers, serve_conjure
from conjure.logger import encode_audio

from claude_jax_physical import build_mel_filterbank, make_mel_fn
from goo_physical import (
    FLAVORS, HF_GENERATORS, GooConfig, HFGenerator, Physics, damping_from_unit_interval,
    log_uniform_interval, make_geometries, make_hf_generator, render, sample_geometry,
    sample_physics, stabilize,
)
from hyper_rnn import AntiCausalBlock, batch_stream, flattened_multiband_spectrogram, max_norm, stft


class Guess(NamedTuple):
    physics: Physics
    hf_params: object
    gain: jnp.ndarray  # () overall output gain


class Encoder(eqx.Module):
    proj: eqx.nn.Conv1d
    blocks: list
    global_head: eqx.nn.MLP
    force_head: eqx.nn.Conv1d
    window: int = eqx.field(static=True)
    hop: int = eqx.field(static=True)

    def __init__(self, window, hop, channels, n_global, n_force, dilations, key):
        keys = jax.random.split(key, len(dilations) + 3)
        self.proj = eqx.nn.Conv1d(window // 2 + 1, channels, 1, key=keys[0])
        self.blocks = [AntiCausalBlock(channels, kernel_size=2, dilation=d, key=k) for d, k in zip(dilations, keys[1:])]
        self.global_head = eqx.nn.MLP(channels, n_global, width_size=channels, depth=2, key=keys[-2])
        self.force_head = eqx.nn.Conv1d(channels, n_force, 1, key=keys[-1])
        self.window, self.hop = window, hop

    def __call__(self, audio):
        spec = stft(audio, self.window, self.hop, pad=True).T  # (freqs, frames)
        x = self.proj(jnp.log(spec + 1e-3))
        for block in self.blocks:
            x = block(x)
        return self.global_head(jnp.max(x, axis=-1)), self.force_head(x)


class GooAutoencoder(eqx.Module):
    encoder: Encoder
    hf: HFGenerator
    cfg: GooConfig = eqx.field(static=True)
    n_impulses: int = eqx.field(static=True)

    def __init__(self, cfg: GooConfig, hf: HFGenerator, window, hop, channels, dilations, n_impulses, key):
        self.cfg, self.hf, self.n_impulses = cfg, hf, n_impulses
        self.encoder = Encoder(
            window, hop, channels, self.n_physics_params + hf.n_params, cfg.n_nodes * cfg.dim, dilations, key)

    @property
    def n_physics_params(self) -> int:
        n, d = self.cfg.n_nodes, self.cfg.dim
        # stiffness pairs, mass, anchor, radius, mics, damping, gain
        return n * (n - 1) // 2 + n + n + n + n * d + 2

    def unpack_physics(self, raw, force_frames) -> Physics:
        cfg = self.cfg
        n, d = cfg.n_nodes, cfg.dim
        n_pairs = n * (n - 1) // 2
        sizes = [n_pairs, n, n, n, n * d, 1, 1]
        stiff, mass, anchor, radius, mics, damping, gain = jnp.split(raw, np.cumsum(sizes)[:-1])

        lo, hi = cfg.stiffness_range
        stiffness = jnp.zeros((n, n)).at[jnp.triu_indices(n, k=1)].set(lo + jax.nn.sigmoid(stiff) * (hi - lo))
        stiffness = stiffness + stiffness.T
        lo, hi = cfg.mass_range
        mass = lo + jax.nn.sigmoid(mass) * (hi - lo)
        lo, hi = cfg.anchor_range
        anchor_k = lo + jax.nn.sigmoid(anchor) * (hi - lo)
        radius = log_uniform_interval(jax.nn.sigmoid(radius), cfg.radius_range)
        damping = damping_from_unit_interval(jax.nn.sigmoid(damping[0]), cfg.damping_range)

        # sparse control: the `n_impulses` strongest (frame, node) force
        # vectors, each applied for one low-rate step at the start of its frame
        n_frames = force_frames.shape[-1]
        f = jnp.tanh(force_frames).T.reshape(n_frames, n, d) * cfg.force_scale
        strength = jnp.linalg.norm(f, axis=-1).reshape(-1)
        threshold = jax.lax.top_k(strength, self.n_impulses)[0][-1]
        f = f * (strength >= threshold).reshape(n_frames, n, 1)
        forces = jnp.zeros((cfg.n_steps, n, d)).at[jnp.arange(n_frames) * cfg.control_stride].set(f)

        physics = Physics(
            positions=jnp.zeros((n, d)), stiffness=stiffness, mass=mass, anchor_k=anchor_k,
            damping=damping, radius=radius, forces=forces, mics=jnp.tanh(mics.reshape(n, d)))
        return stabilize(physics), log_uniform_interval(jax.nn.sigmoid(gain[0]), (0.01, 100.0))

    def guess(self, audio) -> Guess:
        g, force_frames = self.encoder(audio)
        physics, gain = self.unpack_physics(g[:self.n_physics_params], force_frames)
        hf_params = self.hf.unpack(g[self.n_physics_params:])
        return Guess(physics, hf_params, gain)

    def render(self, guess: Guess, key, fixed=None):
        out = render(guess.physics, self.hf, guess.hf_params, key, self.cfg.block, fixed, self.cfg.restitution)
        return out._replace(audio=out.audio * guess.gain, lf=out.lf * guess.gain, hf=out.hf * guess.gain)


def make_spectral_loss(args):
    mel_fn = make_mel_fn(args.window, args.hop, jnp.array(build_mel_filterbank(args.samplerate, args.window, 128)))
    fns = {
        'mel': [mel_fn],
        'multiband': [flattened_multiband_spectrogram],
        'both': [mel_fn, flattened_multiband_spectrogram],
    }[args.loss]

    def loss(a, b):
        # magnitudes are compressed so quiet detail (decays, hf) still counts.
        # the eps matters: x ** c (c < 1) has an infinite derivative at 0, and
        # exact zeros (silence, empty bins) turn every gradient into NaN
        c = args.compression
        return sum(jnp.mean(jnp.abs((fn(a) + 1e-6) ** c - (fn(b) + 1e-6) ** c)) for fn in fns)

    return loss


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'overfit'], required=True)
    parser.add_argument('--batch-size', type=int, default=4, help='real examples per step; ignored in overfit mode')
    parser.add_argument('--n-samples', type=int, default=2 ** 16)
    parser.add_argument('--samplerate', type=int, default=22050)
    # synth
    parser.add_argument('--n-nodes', type=int, default=16, help='rounded up to a grid, see goo_physical.Geometries')
    parser.add_argument('--dim', type=int, default=3)
    parser.add_argument('--block', type=int, default=8, help='physics runs at samplerate / block')
    parser.add_argument('--restitution', type=float, default=0.5)
    parser.add_argument('--hf', nargs='+', choices=HF_GENERATORS, default=['collision'],
                        help='high-frequency generator(s); several are summed')
    parser.add_argument('--n-modes', type=int, default=8, help="'collision': damped oscillators per node")
    parser.add_argument('--n-filters', type=int, default=32, help="'energy': learned filter bank size")
    parser.add_argument('--n-impulses', type=int, default=16,
                        help='predicted control signal keeps this many (frame, node) force vectors')
    # encoder
    parser.add_argument('--window', type=int, default=1024, help='spectrogram window (encoder input and mel loss)')
    parser.add_argument('--hop', type=int, default=256, help='spectrogram hop; also the control rate')
    parser.add_argument('--channels', type=int, default=128)
    parser.add_argument('--dilations', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 1])
    # loss
    parser.add_argument('--loss', choices=['mel', 'multiband', 'both'], default='both')
    parser.add_argument('--compression', type=float, default=0.5, help='spectral magnitudes ** this')
    parser.add_argument('--synthetic-coeff', type=float, default=1.0, help='0 turns off synthetic targets')
    parser.add_argument('--synthetic-batch-size', type=int, default=4)
    parser.add_argument('--flavors', nargs='+', choices=FLAVORS, default=list(FLAVORS),
                        help='geometries synthetic instruments are drawn from')
    parser.add_argument('--max-impulses', type=int, default=4, help='per synthetic instrument')
    # optimization / logging
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collection', type=str, default='goo_physical')
    parser.add_argument('--port', type=int, default=9996)
    args = parser.parse_args(argv)

    if args.hop % args.block != 0:
        raise ValueError('--hop must be divisible by --block')
    if args.n_samples % args.hop != 0:
        raise ValueError('--n-samples must be divisible by --hop')
    return args


def build(args):
    """-> (cfg, model, compute_loss(model, batch, key) -> (loss, aux))"""
    geometries = make_geometries(args.n_nodes, args.dim)
    cfg = GooConfig(
        n_nodes=geometries.n_nodes, dim=args.dim, n_samples=args.n_samples, block=args.block,
        sample_rate=args.samplerate, restitution=args.restitution, max_impulses=args.max_impulses,
        control_stride=args.hop // args.block)

    model_key, hf_key = jax.random.split(jax.random.key(args.seed))
    hf = make_hf_generator(args.hf, cfg, hf_key, n_modes=args.n_modes, n_filters=args.n_filters)
    model = GooAutoencoder(cfg, hf, args.window, args.hop, args.channels, args.dilations, args.n_impulses, model_key)

    spectral_loss = make_spectral_loss(args)

    def compute_loss(model, batch, key):
        real_key, synth_key, s_noise_key, p_noise_key = jax.random.split(key, 4)

        guesses = jax.vmap(model.guess)(batch)
        recon = jax.vmap(model.render)(guesses, jax.random.split(real_key, batch.shape[0]))
        real_loss = jnp.mean(jax.vmap(spectral_loss)(batch, recon.audio))

        aux = dict(real=real_loss, recon=recon, collisions=jnp.mean(recon.n_collisions))

        if args.synthetic_coeff > 0:
            n = args.synthetic_batch_size

            def random_instrument(k):
                k_geo, k_phys, k_hf, k_noise = jax.random.split(k, 4)
                geometry = sample_geometry(k_geo, cfg, geometries, args.flavors)
                truth = Guess(sample_physics(k_phys, cfg, geometry), model.hf.sample(k_hf), jnp.array(1.0))
                audio = model.render(truth, k_noise, geometry.fixed).audio
                # the encoder hears (and the loss compares) max-normalized
                # audio, like the real data; the predicted gain absorbs it
                return audio / (jnp.max(jnp.abs(audio)) + 1e-8)

            synthetic = jax.lax.stop_gradient(jax.vmap(random_instrument)(jax.random.split(synth_key, n)))
            s_guesses = jax.vmap(model.guess)(synthetic)
            s_recon = jax.vmap(model.render)(s_guesses, jax.random.split(p_noise_key, n))
            synthetic_loss = jnp.mean(jax.vmap(spectral_loss)(synthetic, s_recon.audio))
            aux.update(synthetic_loss=synthetic_loss, synthetic=synthetic, synthetic_recon=s_recon.audio)
        else:
            synthetic_loss = jnp.zeros(())

        return real_loss + args.synthetic_coeff * synthetic_loss, aux

    return cfg, model, compute_loss


def main():
    args = parse_args()
    cfg, model, compute_loss = build(args)
    key = jax.random.key(args.seed + 1)

    optimizer = optax.apply_if_finite(
        optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr)),
        max_consecutive_errors=100)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, opt_state, batch, key):
        (loss, aux), grads = eqx.filter_value_and_grad(compute_loss, has_aux=True)(model, batch, key)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss, aux

    collection = LmdbCollection(path=args.collection)
    names = ['orig', 'recon', 'recon-lf', 'recon-hf']
    if args.synthetic_coeff > 0:
        names += ['synthetic', 'synthetic-recon']
    audio_loggers = dict(zip(names, loggers(
        names, 'audio/wav', lambda x: encode_audio(x, samplerate=args.samplerate), collection)))
    serve_conjure(list(audio_loggers.values()), port=args.port, n_workers=1, web_components_version='0.0.101')

    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}')
    print(f'physics at {cfg.sample_rate / cfg.block:.0f} Hz, {cfg.n_nodes} nodes, hf={args.hf} '
          f'({model.n_physics_params} physics + {model.hf.n_params} hf params/example, '
          f'control every {cfg.control_stride} steps)')

    stream = batch_stream(args.mode, args.batch_size, args.n_samples, args.samplerate)
    for i, batch in zip(itertools.count(), stream):
        key, step_key = jax.random.split(key)
        model, opt_state, loss, aux = train_step(model, opt_state, batch, step_key)

        recon = aux['recon']
        audio_loggers['orig'](max_norm(np.asarray(batch[0])))
        audio_loggers['recon'](max_norm(np.asarray(recon.audio[0])))
        audio_loggers['recon-lf'](max_norm(np.asarray(recon.lf[0])))
        audio_loggers['recon-hf'](max_norm(np.asarray(recon.hf[0])))

        line = f'{i} loss {float(loss):.4f}  real {float(aux["real"]):.4f}  collisions {float(aux["collisions"]):.0f}'
        if args.synthetic_coeff > 0:
            audio_loggers['synthetic'](max_norm(np.asarray(aux['synthetic'][0])))
            audio_loggers['synthetic-recon'](max_norm(np.asarray(aux['synthetic_recon'][0])))
            line += f'  synthetic {float(aux["synthetic_loss"]):.4f}'
        print(line)


if __name__ == '__main__':
    main()
