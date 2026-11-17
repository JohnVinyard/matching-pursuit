"""
Does `jax_ae.EventAutoencoder`'s analysis handle scale the way we want?

Two sweeps over one audio segment, comparing each gain's analysis (the
per-frame magnitude logits, fine offsets and event vectors) to the
analysis at gain 1:

1. Global gain: the whole segment is scaled by g.  Ideally the selected
   frame, fine offset and event vector don't change, and the predicted
   magnitude tracks g exactly (so "magnitude tracking error" stays at 0 dB),
   until it hits --min-magnitude / --max-magnitude.

2. Local gain: only a region of the segment (--region) is scaled by g, with
   short linear ramps.  Reported per zone, as the mean relative change of
   the (non-silent) frames' event vectors:
     - after: frames past the region, far enough that their STFT windows
       don't overlap it.  An anti-causal encoder can't see the region from
       there, so any change means the description of a sound depends on
       how loud *other* sounds are (e.g. through a global normalization).
     - inside: frames within the region.  The sound there is the same, just
       louder, but (anti-causally) they also see the ungained audio after
       the region, so some change is legitimate.
     - before: frames before the region, which see it; change expected.
   It also reports where the selected (loudest-candidate) frame lands:
   as g grows, it should move into the region.

Pass the same model flags the model was trained with (see `jax_ae.py`), and
--checkpoint (saved by `jax_ae.py --save-path`).  Without a checkpoint, the
sweep runs on a freshly initialized model, which shows only the invariances
the architecture has by construction.

    python gain_sweep.py --encoder attention --checkpoint ae.eqx --source overfit
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import lfilter

from hyper_rnn import batch_stream
from jax_ae import build_model, build_parser, validate_args


def bursts(n_samples: int, samplerate: int, seed: int) -> np.ndarray:
    """
    A test segment that needs no dataset: 12 exponentially decaying,
    one-pole-filtered noise bursts at random positions, with levels spread
    over 40 dB.
    """
    rng = np.random.default_rng(seed)
    audio = np.zeros(n_samples)
    t = np.arange(n_samples) / samplerate
    for _ in range(12):
        start = rng.integers(0, n_samples - n_samples // 8)
        decay = np.exp(-t[:n_samples - start] / rng.uniform(0.02, 0.3))
        noise = rng.normal(size=n_samples - start)
        a = rng.uniform(0., 0.95)
        noise = lfilter([1 - a], [1, -a], noise)
        audio[start:] += noise * decay * 10 ** rng.uniform(-2, 0)
    return audio / np.abs(audio).max()


def relative_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """per-row ||a - b|| / ||b||"""
    return np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-8)


def main():
    parser = build_parser(training=False)
    parser.description = __doc__
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='a model saved by jax_ae.py --save-path; omit to sweep a freshly initialized model')
    parser.add_argument('--source', choices=['bursts', 'overfit', 'train'], default='bursts',
                        help="'bursts': synthetic noise bursts (no dataset needed); 'overfit': the "
                             "training overfit segment; 'train': one segment from the training stream")
    parser.add_argument('--region', type=float, nargs=2, default=(0.375, 0.625),
                        help='the local sweep\'s region, as fractions of the segment')
    parser.add_argument('--gains-db', type=float, nargs='+',
                        default=[-60., -40., -20., -10., 0., 10., 20.])
    args = parser.parse_args()
    validate_args(args)

    model_key, generator_key, _ = jax.random.split(jax.random.key(args.seed), 3)
    model = build_model(args, model_key, generator_key)
    if args.checkpoint:
        model = eqx.tree_deserialise_leaves(args.checkpoint, model)
    else:
        print('no --checkpoint: sweeping a freshly initialized model\n')

    if args.source == 'bursts':
        audio = jnp.asarray(bursts(args.n_samples, args.samplerate, args.seed), dtype=jnp.float32)
    else:
        audio = next(batch_stream(args.source, 1, args.n_samples, args.samplerate))[0]

    @eqx.filter_jit
    def analyze(model, audio):
        spec = model.spectrogram(audio)
        logits, fine, events = model.analyze(spec)
        return logits, fine, events, model.log_magnitude(logits), jnp.sqrt(jnp.mean(spec ** 2, axis=-1))

    def run(audio):
        return [np.asarray(x) for x in analyze(model, audio)]

    ref_logits, ref_fine, ref_events, ref_logmag, ref_level = run(audio)
    ref_frame = int(np.argmax(ref_logits))
    log_min, log_max = np.log(args.min_magnitude), np.log(args.max_magnitude)
    to_db = 20 / np.log(10)
    # frames worth comparing: within 60 dB of the loudest
    active = ref_level > ref_level.max() * 1e-3

    print(f'reference: selected frame {ref_frame}, magnitude {np.exp(ref_logmag[ref_frame]):.3g}, '
          f'{active.sum()}/{len(active)} frames active\n')

    # ---------------------------------------------------------------- global
    print('1. global gain')
    print(f'{"gain dB":>8} {"frame":>6} {"event Δ@sel":>12} {"event Δ mean":>13} '
          f'{"fine Δ smp":>11} {"mag track dB":>13}')
    for db in args.gains_db:
        g = 10 ** (db / 20)
        logits, fine, events, logmag, _ = run(audio * g)
        frame = int(np.argmax(logits))
        d_sel = relative_change(events[ref_frame], ref_events[ref_frame])
        d_mean = relative_change(events[active], ref_events[active]).mean()
        d_fine = abs(fine[ref_frame] - ref_fine[ref_frame]) * args.hop
        track = (logmag[ref_frame] - ref_logmag[ref_frame] - np.log(g)) * to_db
        bound = ' (at bound)' if min(logmag[ref_frame] - log_min, log_max - logmag[ref_frame]) < 0.01 else ''
        print(f'{db:>8.0f} {frame:>6d} {d_sel:>12.4f} {d_mean:>13.4f} {d_fine:>11.1f} {track:>13.1f}{bound}')

    # ----------------------------------------------------------------- local
    n_frames = model.n_frames
    start, end = (int(r * args.n_samples) for r in args.region)
    ramp = args.hop
    t = np.arange(args.n_samples)
    # 0 outside the region, 1 inside, linear ramps at the edges
    inside_env = np.clip(np.minimum(t - start, end - t) / ramp + 0.5, 0, 1)

    # frame f covers samples [f * hop, f * hop + window)
    f_start, f_end = start // args.hop, end // args.hop
    margin = args.window // args.hop + 1
    frames = np.arange(n_frames)
    zones = {
        'before': (frames < f_start - margin) & active,
        'inside': (frames >= f_start) & (frames < f_end - margin) & active,
        'after': (frames >= f_end + 1) & active,
    }

    print(f'\n2. local gain on samples [{start}, {end}) = frames [{f_start}, {f_end}); '
          + ', '.join(f'{k}: {v.sum()} active frames' for k, v in zones.items()))
    print(f'{"gain dB":>8} {"frame":>6} {"in region":>10} '
          + ' '.join(f'{"Δ " + k:>10}' for k in zones))
    for db in args.gains_db:
        g = 10 ** (db / 20)
        env = jnp.asarray(1 + (g - 1) * inside_env, dtype=audio.dtype)
        logits, fine, events, logmag, _ = run(audio * env)
        frame = int(np.argmax(logits))
        in_region = 'yes' if f_start <= frame < f_end else 'no'
        deltas = [
            relative_change(events[m], ref_events[m]).mean() if m.any() else float('nan')
            for m in zones.values()]
        print(f'{db:>8.0f} {frame:>6d} {in_region:>10} ' + ' '.join(f'{d:>10.4f}' for d in deltas))


if __name__ == '__main__':
    main()
