"""
Multiband matching pursuit and dictionary learning, in jax.

Follows the pattern of `modules/multibanddict.py`: audio is split with
`fft_frequency_decompose` into bands of increasing size (and samplerate),
each band has its own dictionary, every band is sparse-coded with matching
pursuit, and then each dictionary is updated from the atoms it used for the
current batch (as `modules/matchingpursuit.dictionary_learning_step` does).

The difference: all atoms have the same *duration*, rather than the same
number of samples, so atom length doubles with each band. To keep the cost
of the search roughly constant across bands, each step uses a randomized,
frequency-domain convolution:

    1. The residual is kept in the frequency domain (an rfft of the band's
       size), so removing an atom is an elementwise, phase-shifted
       subtraction, and no FFT of the residual is needed per step.
    2. `M_k` of the band's passband bins are chosen (see --coeff-sampling),
       and multiplied by the conjugate spectra of every atom. Only these
       `n_atoms x M_k` complex products are computed.
    3. The products are folded (summed modulo `K`, see --n-positions) and
       inverse transformed with a size-`K` FFT. Folding the spectrum is
       exactly subsampling the correlation in time, so this gives the
       (approximate) correlation of every atom at every `N / K`-th position.
    4. The best atom on that coarse grid wins, and only that atom's exact,
       full-resolution correlation is computed, to choose its position and
       amplitude.

Per step, the search costs `n_atoms * (M_k + K log K)`, no matter the band
size. How `M_k` changes from band to band is set by --coeff-schedule.

By default, the lowest band uses every passband bin and `K` equals its size,
so it's exact (up to choosing atoms by envelope; see --selection).
"""
import argparse
import itertools
import os
import time
from functools import partial
from typing import List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, loggers, serve_conjure
from conjure.logger import encode_audio

from hyper_rnn import batch_stream, fft_frequency_decompose, max_norm


# ---------------------------------------------------------------------------
# how the number of complex coefficients, `M_k`, grows with band index `k`
# (k = 0 is the lowest band).  The result is always clipped to the number of
# bins in the band's passband, where the search is exact.

COEFF_SCHEDULES = {
    # M_k = base: constant cost per band
    'constant': lambda base, k, growth: base,
    # M_k = base * sqrt(k + 1)
    'sublinear': lambda base, k, growth: base * np.sqrt(k + 1),
    # M_k = base * (k + 1)
    'linear': lambda base, k, growth: base * (k + 1),
    # M_k = base * growth ** k; growth=2 keeps the same fraction of bins in
    # every band (i.e., exact everywhere, if the lowest band is exact)
    'exponential': lambda base, k, growth: base * growth ** k,
}


class Band(NamedTuple):
    """Static description of a single band (hashable, so usable as a static jit arg)"""

    index: int
    # band length, in samples
    size: int
    # atom length, in samples
    atom_size: int
    # first rfft bin in the band's passband (the passband runs to Nyquist)
    lo_bin: int
    # number of rfft bins in the passband
    n_bins: int
    # number of complex coefficients used for the randomized search
    n_coeffs: int
    # number of positions on the coarse search grid
    n_positions: int

    @property
    def stride(self) -> int:
        return self.size // self.n_positions

    @property
    def n_valid(self) -> int:
        """Number of positions at which an atom fits entirely within the band"""
        return self.size - self.atom_size + 1

    @property
    def all_bins(self) -> bool:
        return self.n_coeffs == self.n_bins


def make_bands(
        n_samples: int,
        min_size: int,
        atom_size: int,
        n_positions: int,
        schedule: str,
        coeff_base: int,
        coeff_growth: float) -> List[Band]:

    bands = []
    size = min_size
    k = 0
    while size <= n_samples:
        # matches the masks in `fft_frequency_decompose`: the lowest band
        # keeps everything, higher bands keep the upper half of their spectrum
        lo_bin = 0 if k == 0 else size // 4
        n_bins = size // 2 + 1 - lo_bin
        n_coeffs = int(round(COEFF_SCHEDULES[schedule](coeff_base, k, coeff_growth)))
        n_coeffs = int(np.clip(n_coeffs, 1, n_bins))

        bands.append(Band(
            index=k,
            size=size,
            atom_size=atom_size * 2 ** k,
            lo_bin=lo_bin,
            n_bins=n_bins,
            n_coeffs=n_coeffs,
            n_positions=min(n_positions, size)))

        size *= 2
        k += 1

    return bands


# ---------------------------------------------------------------------------
# signal helpers

def unit_norm(x: jnp.ndarray, axis: int = -1, epsilon: float = 1e-8) -> jnp.ndarray:
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + epsilon)


def band_limit(atoms: jnp.ndarray, band: Band) -> jnp.ndarray:
    """
    Project atoms onto the band's passband.  Atoms run at the band's
    samplerate, so the passband is the same fraction of their spectrum.
    """
    if band.index == 0:
        return atoms
    spec = jnp.fft.rfft(atoms, axis=-1)
    spec = spec.at[..., :band.atom_size // 4].set(0)
    return jnp.fft.irfft(spec, n=band.atom_size, axis=-1)


def fft_resample(x: jnp.ndarray, desired_size: int, is_lowest_band: bool) -> jnp.ndarray:
    """A jax port of `modules.decompose.fft_resample`"""
    coeffs = jnp.fft.rfft(x, norm='ortho')
    n_coeffs = coeffs.shape[-1]
    new_coeffs = jnp.zeros(x.shape[:-1] + (desired_size // 2 + 1,), dtype=coeffs.dtype)
    if is_lowest_band:
        new_coeffs = new_coeffs.at[..., :n_coeffs].set(coeffs)
    else:
        new_coeffs = new_coeffs.at[..., n_coeffs // 2:n_coeffs].set(coeffs[..., n_coeffs // 2:])
    return jnp.fft.irfft(new_coeffs, n=desired_size, norm='ortho')


def fft_frequency_recompose(bands: List[jnp.ndarray], desired_size: int) -> jnp.ndarray:
    """A jax port of `modules.decompose.fft_frequency_recompose`; `bands` is ordered lowest first"""
    return sum(fft_resample(b, desired_size, i == 0) for i, b in enumerate(bands))


def init_dictionary(key, n_atoms: int, band: Band) -> jnp.ndarray:
    atoms = jax.random.uniform(key, (n_atoms, band.atom_size), minval=-1, maxval=1)
    return unit_norm(band_limit(atoms, band))


# ---------------------------------------------------------------------------
# randomized, frequency-domain matching pursuit

def choose_bins(key, residual_spec: jnp.ndarray, band: Band, sampling: str) -> jnp.ndarray:
    """Choose `band.n_coeffs` (absolute) rfft bins from the band's passband"""
    if band.all_bins:
        return jnp.arange(band.lo_bin, band.lo_bin + band.n_bins)

    if sampling == 'uniform':
        # with replacement, so this costs O(n_coeffs), rather than O(n_bins);
        # duplicates are just counted twice
        idx = jax.random.randint(key, (band.n_coeffs,), 0, band.n_bins)
    else:
        log_mag = jnp.log(jnp.abs(residual_spec[band.lo_bin:]) + 1e-12)
        if sampling == 'magnitude':
            # gumbel top-k: sampling without replacement, in proportion to
            # the residual's current magnitude spectrum
            log_mag = log_mag + jax.random.gumbel(key, log_mag.shape)
        _, idx = jax.lax.top_k(log_mag, band.n_coeffs)

    return idx + band.lo_bin


def coarse_feature_map(
        residual_spec: jnp.ndarray,
        atom_specs: jnp.ndarray,
        bins: jnp.ndarray,
        band: Band,
        selection: str) -> jnp.ndarray:
    """
    Approximate correlation of every atom with the residual, at every
    `band.stride`-th position, using only the complex coefficients in `bins`.

    residual_spec: (size // 2 + 1,)
    atom_specs:    (n_atoms, size // 2 + 1)
    returns:       (n_atoms, n_positions), with invalid positions set to -1

    The correlation is c[t] = (1/N) Re sum_f w_f C_f exp(2 pi i f t / N),
    with C = R * conj(D), and w_f = 2, except at DC and Nyquist, where it's
    1.  At t = stride * tau, the exponential only depends on f mod K, so
    summing coefficients modulo K and taking a size-K inverse FFT gives the
    correlation at every coarse position, exactly (when all bins are used).
    Without taking the real part, this is the analytic signal, whose
    magnitude is the correlation's envelope.
    """
    n_atoms = atom_specs.shape[0]
    k = band.n_positions

    w = jnp.where((bins == 0) | (bins == band.size // 2), 1., 2.)
    products = (residual_spec[bins] * w)[None, :] * jnp.conj(atom_specs[:, bins])

    folded = jnp.zeros((n_atoms, k), dtype=products.dtype).at[:, bins % k].add(products)

    # scale is (K / N), which doesn't affect the argmax
    fm = jnp.fft.ifft(folded, axis=-1)

    if selection == 'envelope':
        fm = jnp.abs(fm)
    else:
        fm = jnp.abs(fm.real)

    valid = jnp.arange(k) * band.stride < band.n_valid
    return jnp.where(valid[None, :], fm, -1.)


def shift_phasor(band: Band, position: jnp.ndarray) -> jnp.ndarray:
    """exp(-2 pi i f p / N), for every rfft bin f, computed with integer phase to keep float32 precise"""
    freqs = jnp.arange(band.size // 2 + 1, dtype=jnp.int32)
    phase = (freqs * position.astype(jnp.int32)) % band.size
    return jnp.exp(-2j * jnp.pi * phase.astype(jnp.float32) / band.size)


def encode_one(
        x: jnp.ndarray,
        atom_specs: jnp.ndarray,
        key,
        band: Band,
        n_steps: int,
        sampling: str,
        selection: str):
    """
    Matching pursuit for a single band of a single example.

    x:          (size,)
    atom_specs: (n_atoms, size // 2 + 1), rffts of the zero-padded, unit-norm atoms
    returns:    the residual (size,), and events (atom index, position, amplitude), each (n_steps,)
    """
    valid = jnp.arange(band.size) < band.n_valid

    def step(residual_spec, step_key):
        bins = choose_bins(step_key, residual_spec, band, sampling)
        fm = coarse_feature_map(residual_spec, atom_specs, bins, band, selection)
        atom_index = jnp.argmax(jnp.max(fm, axis=-1))

        # exact, full-resolution correlation, for the chosen atom only
        atom_spec = atom_specs[atom_index]
        corr = jnp.fft.irfft(residual_spec * jnp.conj(atom_spec), n=band.size)
        position = jnp.argmax(jnp.where(valid, jnp.abs(corr), -1.))
        amp = corr[position]

        residual_spec = residual_spec - amp * atom_spec * shift_phasor(band, position)
        return residual_spec, (atom_index, position, amp)

    residual_spec, events = jax.lax.scan(step, jnp.fft.rfft(x), jax.random.split(key, n_steps))
    return jnp.fft.irfft(residual_spec, n=band.size), events


@partial(jax.jit, static_argnames=('band', 'n_steps', 'sampling', 'selection'))
def encode_band(atoms: jnp.ndarray, x: jnp.ndarray, key, band: Band, n_steps: int, sampling: str, selection: str):
    """
    atoms: (n_atoms, atom_size)
    x:     (batch, size)
    """
    # zero-pads each atom to the band's size.  Atom positions are restricted
    # so that atoms fit within the band, which means circular correlation
    # never wraps around, and no further padding is needed
    atom_specs = jnp.fft.rfft(atoms, n=band.size, axis=-1)
    keys = jax.random.split(key, x.shape[0])
    return jax.vmap(encode_one, in_axes=(0, None, 0, None, None, None, None))(
        x, atom_specs, keys, band, n_steps, sampling, selection)


@partial(jax.jit, static_argnames=('band',))
def update_dictionary(
        atoms: jnp.ndarray,
        residual: jnp.ndarray,
        events: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
        band: Band,
        lr: float):
    """
    A vectorized `dictionary_learning_step`: for every event, add its atom
    back to the residual at its position, and gather that segment.  Each
    used atom becomes the (amplitude-weighted) sum of its segments.  Unused
    atoms are left alone.

    residual: (batch, size)
    events:   (atom index, position, amplitude), each (batch, n_steps)
    """
    n_atoms = atoms.shape[0]
    atom_index, position, amp = (e.reshape(-1) for e in events)
    batch_index = jnp.repeat(jnp.arange(residual.shape[0]), events[0].shape[1])

    segments = jax.vmap(
        lambda b, p: jax.lax.dynamic_slice(residual[b], (p,), (band.atom_size,)))(batch_index, position)
    segments = segments + amp[:, None] * atoms[atom_index]

    new_atoms = jax.ops.segment_sum(amp[:, None] * segments, atom_index, num_segments=n_atoms)
    counts = jax.ops.segment_sum(jnp.ones_like(amp), atom_index, num_segments=n_atoms)

    new_atoms = unit_norm(band_limit(new_atoms, band))
    mixed = unit_norm((1 - lr) * atoms + lr * new_atoms)
    return jnp.where(counts[:, None] > 0, mixed, atoms), counts


# ---------------------------------------------------------------------------
# monitoring

def snr_db(signal: np.ndarray, residual: np.ndarray) -> float:
    return float(10 * np.log10((signal ** 2).sum() / ((residual ** 2).sum() + 1e-12) + 1e-12))


def event_map(events: List[Tuple[np.ndarray, np.ndarray, np.ndarray]], bands: List[Band], n_frames: int) -> np.ndarray:
    """(n_bands, n_frames) map of event magnitude over (unit) time, for the first example in the batch"""
    out = np.zeros((len(bands), n_frames), dtype=np.float32)
    for k, ((_, position, amp), band) in enumerate(zip(events, bands)):
        frames = (np.asarray(position[0]) / band.size * n_frames).astype(int)
        np.add.at(out[k], frames, np.abs(np.asarray(amp[0])))
    return out / (out.max() + 1e-12)


def atom_audition(
        dictionaries: List[jnp.ndarray],
        counts: List[jnp.ndarray],
        bands: List[Band],
        n_samples: int,
        samplerate: int) -> np.ndarray:
    """The most-used atom of every band, lowest first, resampled to the native samplerate, separated by silence"""
    gap = np.zeros(samplerate // 10, dtype=np.float32)
    chunks = []
    for d, c, band in zip(dictionaries, counts, bands):
        atom = d[int(jnp.argmax(c))]
        native = fft_resample(atom, band.atom_size * (n_samples // band.size), band.index == 0)
        chunks.extend([max_norm(np.asarray(native)), gap])
    return np.concatenate(chunks)


def print_bands(bands: List[Band], n_atoms: int, n_samples: int, samplerate: int):
    atom_seconds = bands[0].atom_size / bands[0].size * n_samples / samplerate
    print(f'{len(bands)} bands, {n_atoms} atoms each, every atom {atom_seconds * 1000:.1f}ms long')
    print(f'{"band":>6} {"atom":>6} {"pass bins":>10} {"coeffs":>7} {"% bins":>7} '
          f'{"positions":>10} {"stride":>7} {"search cost":>12} {"exact cost":>12}')
    for b in bands:
        # rough multiply-adds per step: atom products + folded inverse FFTs
        cost = n_atoms * (b.n_coeffs + b.n_positions * np.log2(b.n_positions))
        exact = n_atoms * (b.n_bins + b.size * np.log2(b.size))
        print(f'{b.size:>6} {b.atom_size:>6} {b.n_bins:>10} {b.n_coeffs:>7} '
              f'{100 * b.n_coeffs / b.n_bins:>6.1f}% {b.n_positions:>10} {b.stride:>7} '
              f'{cost:>12.3g} {exact:>12.3g}')


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'overfit'], default='train')
    parser.add_argument('--batch-size', type=int, default=4, help='ignored in overfit mode')
    parser.add_argument('--n-samples', type=int, default=2 ** 15)
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--min-band-size', type=int, default=512,
                        help='size of the lowest band, in samples; bands double up to --n-samples')
    parser.add_argument('--atom-size', type=int, default=64,
                        help='atom length in the lowest band, in samples.  It doubles with each band, '
                             'so every atom spans the same duration')
    parser.add_argument('--n-atoms', type=int, default=256, help='atoms per band')
    parser.add_argument('--n-steps', type=int, default=32, help='matching pursuit steps per band')
    parser.add_argument('--coeff-schedule', choices=list(COEFF_SCHEDULES), default='constant',
                        help='how the number of complex coefficients used by the search grows with '
                             "band index k: 'constant' (base), 'sublinear' (base * sqrt(k + 1)), "
                             "'linear' (base * (k + 1)) or 'exponential' (base * growth ** k).  "
                             "Always clipped to the band's passband bins")
    parser.add_argument('--coeff-base', type=int, default=None,
                        help="coefficients in the lowest band; defaults to all of the lowest band's bins")
    parser.add_argument('--coeff-growth', type=float, default=2.,
                        help="'exponential' only: growth per band")
    parser.add_argument('--coeff-sampling', choices=['uniform', 'magnitude', 'topk'], default='uniform',
                        help="how bins are chosen at each step: 'uniform' at random; 'magnitude' at random, "
                             "in proportion to the residual's magnitude spectrum; 'topk' the largest bins "
                             "of the residual (deterministic)")
    parser.add_argument('--n-positions', type=int, default=None,
                        help='positions on the coarse search grid (a power of two); defaults to '
                             '--min-band-size, so the lowest band searches every position')
    parser.add_argument('--selection', choices=['envelope', 'real'], default='envelope',
                        help="statistic used to pick an atom from the coarse grid: 'envelope' is the "
                             "magnitude of the analytic correlation, which is robust to the grid's stride "
                             "aliasing oscillating correlations; 'real' is the usual |correlation|")
    parser.add_argument('--dict-lr', type=float, default=1.0,
                        help='1.0 replaces used atoms, as `dictionary_learning_step` does; smaller values '
                             'blend new atoms into old ones')
    parser.add_argument('--dict-path', type=str, default=None,
                        help='.npz file to load dictionaries from (if it exists) and periodically save to')
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--report-every', type=int, default=25,
                        help='print per-band SNR, timing and atom usage every this many steps')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--collection', type=str, default='jax_multiband_mp')
    parser.add_argument('--port', type=int, default=9995)
    args = parser.parse_args()

    def is_pow2(n):
        return n > 0 and (n & (n - 1)) == 0

    n_positions = args.n_positions or args.min_band_size
    for name, value in [('--n-samples', args.n_samples), ('--min-band-size', args.min_band_size),
                        ('--atom-size', args.atom_size), ('--n-positions', n_positions)]:
        if not is_pow2(value):
            parser.error(f'{name} must be a power of two')
    if args.min_band_size > args.n_samples:
        parser.error('--min-band-size must be <= --n-samples')
    if args.atom_size >= args.min_band_size:
        parser.error('--atom-size must be smaller than --min-band-size')

    coeff_base = args.coeff_base or (args.min_band_size // 2 + 1)
    bands = make_bands(
        args.n_samples, args.min_band_size, args.atom_size, n_positions,
        args.coeff_schedule, coeff_base, args.coeff_growth)
    print_bands(bands, args.n_atoms, args.n_samples, args.samplerate)

    collection = LmdbCollection(path=args.collection)

    orig_audio, recon_audio, residual_audio, atoms_audio = loggers(
        ['orig', 'recon', 'residual', 'atoms'],
        'audio/wav',
        lambda x: encode_audio(x, samplerate=args.samplerate),
        collection)

    events_logger, = loggers(
        ['events'],
        SupportedContentType.Spectrogram.value,
        lambda x: x,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    serve_conjure(
        [orig_audio, recon_audio, residual_audio, atoms_audio, events_logger],
        port=args.port,
        n_workers=1,
        web_components_version='0.0.101')

    key = jax.random.key(args.seed)
    key, *dict_keys = jax.random.split(key, len(bands) + 1)
    dictionaries = [init_dictionary(k, args.n_atoms, band) for k, band in zip(dict_keys, bands)]

    if args.dict_path and os.path.exists(args.dict_path):
        stored = np.load(args.dict_path)
        dictionaries = [jnp.asarray(stored[f'band_{band.size}']) for band in bands]
        print(f'loaded dictionaries from {args.dict_path}')

    decompose = jax.jit(jax.vmap(lambda x: fft_frequency_decompose(x, args.min_band_size)))
    recompose = jax.jit(lambda bs: fft_frequency_recompose(bs, args.n_samples))

    print(f'jax backend: {jax.default_backend()}, devices: {jax.devices()}')

    stream = batch_stream(args.mode, args.batch_size, args.n_samples, args.samplerate)
    band_ms = np.zeros(len(bands))

    for i, batch in zip(itertools.count(), stream):
        band_signals = decompose(batch)

        recon_bands, residual_bands, all_events, all_counts = [], [], [], []
        snrs, used = [], []

        for k, (band, x) in enumerate(zip(bands, band_signals)):
            key, step_key = jax.random.split(key)

            start = time.perf_counter()
            residual, events = encode_band(
                dictionaries[k], x, step_key, band, args.n_steps, args.coeff_sampling, args.selection)
            residual.block_until_ready()
            elapsed = (time.perf_counter() - start) * 1000
            # skip the first step, which includes compilation
            band_ms[k] = elapsed if i == 1 else (0.9 * band_ms[k] + 0.1 * elapsed)

            dictionaries[k], counts = update_dictionary(dictionaries[k], residual, events, band, args.dict_lr)

            recon_bands.append(x - residual)
            residual_bands.append(residual)
            all_events.append(events)
            all_counts.append(counts)
            snrs.append(snr_db(np.asarray(x), np.asarray(residual)))
            used.append(int((counts > 0).sum()))

        recon = recompose(recon_bands)
        residual = recompose(residual_bands)

        orig_audio(max_norm(np.asarray(batch[0])))
        recon_audio(max_norm(np.asarray(recon[0])))
        residual_audio(max_norm(np.asarray(residual[0])))
        events_logger(event_map(all_events, bands, n_frames=256))
        atoms_audio(atom_audition(dictionaries, all_counts, bands, args.n_samples, args.samplerate))

        total_snr = snr_db(np.asarray(batch), np.asarray(batch - recon))
        print(f'{i} snr {total_snr:.2f}dB  search {band_ms.sum():.1f}ms')

        if i > 0 and i % args.report_every == 0:
            print(f'    {"band":>6} {"snr dB":>7} {"ms":>7} {"used":>5}')
            for band, s, ms, u in zip(bands, snrs, band_ms, used):
                print(f'    {band.size:>6} {s:>7.2f} {ms:>7.2f} {u:>5}')

        if args.dict_path and i > 0 and i % args.save_every == 0:
            np.savez(args.dict_path, **{f'band_{b.size}': np.asarray(d) for b, d in zip(bands, dictionaries)})


if __name__ == '__main__':
    main()
