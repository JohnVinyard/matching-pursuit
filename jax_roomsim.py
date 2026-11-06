"""
A voxel room simulator: the room is a grid of voxels, each with a real-valued
(zero-phase) transfer function describing how much of each frequency survives
a trip through that volume of space.  A control signal injects energy (here, a
burst of noise) into particular voxels, and we listen at a "mic" voxel.

Propagation is a block waveguide mesh rather than an averaging kernel: an
averaging (first-order-in-time) update is the heat equation, which diffuses
energy (distance ~ sqrt(t)) and has no speed of sound at all.  A waveguide
mesh carries directional wave variables between face-neighbours, so a
wavefront moves at a fixed speed.

  - each voxel is a scattering junction with 2N ports (+/- each axis)
  - the junction pressure is p = (1/N) * sum(incoming) + control
  - the wave leaving on port d is p - incoming[d], filtered by the voxel's
    transfer function, and arrives at the neighbour D samples later

Speed of sound and voxel size
-----------------------------

A rectilinear mesh moves 1/sqrt(N) voxels per link-step (at low frequencies),
so with a link delay of D samples,

    dx = c * D * sqrt(N) / samplerate

Each simulation step advances one hop of H samples, and every link has to
delay by at least H samples for the block update to be causal.  A real-valued
transfer function with L/2 + 1 bins becomes a linear-phase FIR with group delay
L/2, so

    D = H + L / 2

and the transfer function's frequency resolution, samplerate / L, is roughly
c / dx: sound only spends ~dx / c seconds in a voxel, so a voxel can't have
spectral detail much finer than ~c / dx Hz.  Choosing voxel size and choosing
transfer-function resolution are the same choice.

No COLA windowing is needed while transfer functions are fixed: the system is
linear and time-invariant, and the block filtering here is exact linear
convolution (overlap-add into a buffer of waves still in flight).

Caveat: wave behaviour (modes, diffraction, interference) is only accurate
well below samplerate / D.  Every link is z^-D times a smooth response, so the
mesh's structure repeats in frequency with period samplerate / D, which may
sound like a metallic comb.

    python jax_roomsim.py
    python jax_roomsim.py --dims 2 --wall-reflection 0     # (nearly) anechoic
    python jax_roomsim.py --hop 32 --filter-length 64      # coarser voxels
    python jax_roomsim.py --air-absorption-scale 20        # exaggerate the air
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, loggers, serve_conjure
from conjure.logger import encode_audio


def speed_of_sound(temperature_c: float) -> float:
    return 331.3 * np.sqrt(1 + temperature_c / 273.15)


def air_absorption_db_per_m(
        freqs: np.ndarray,
        temperature_c: float = 20.,
        humidity: float = 50.,
        pressure_kpa: float = 101.325) -> np.ndarray:
    """
    ISO 9613-1 atmospheric absorption, in dB per metre.
    """
    t = temperature_c + 273.15
    t0 = 293.15
    t01 = 273.16
    pr = 101.325
    pa = pressure_kpa

    # molar concentration of water vapour, percent
    c = -6.8346 * (t01 / t) ** 1.261 + 4.6151
    h = humidity * (10 ** c) * (pr / pa)

    fr_o = (pa / pr) * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = (pa / pr) * (t / t0) ** -0.5 * (9 + 280 * h * np.exp(-4.170 * ((t / t0) ** (-1 / 3) - 1)))

    f2 = freqs ** 2
    return 8.686 * f2 * (
        1.84e-11 * (pr / pa) * (t / t0) ** 0.5
        + (t / t0) ** -2.5 * (
            0.01275 * np.exp(-2239.1 / t) / (fr_o + f2 / fr_o)
            + 0.1068 * np.exp(-3352.0 / t) / (fr_n + f2 / fr_n)))


def hann_window(n: int) -> jnp.ndarray:
    return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n) / n)


def linear_phase_fir(transfer: jnp.ndarray) -> jnp.ndarray:
    """
    Real-valued transfer functions (..., L/2 + 1) -> causal linear-phase FIRs
    (..., L + 1) with group delay L/2.  The zero-phase impulse response is
    rolled to be centered at L/2; its middle sample would land on both ends,
    so it's split in half between them.  The result's frequency response is
    exactly `transfer` at the bin frequencies (times the L/2 delay).
    """
    n = 2 * (transfer.shape[-1] - 1)
    h = jnp.roll(jnp.fft.irfft(transfer, n=n, axis=-1), n // 2, axis=-1)
    h = jnp.concatenate([h, h[..., :1]], axis=-1)
    return h.at[..., 0].multiply(0.5).at[..., -1].multiply(0.5)


def gather_incoming(heads: jnp.ndarray, wall_reflection: float) -> jnp.ndarray:
    """
    `heads` is (*grid, 2N, H): the waves leaving each voxel on each port,
    arriving now.  Port 2a faces +axis a, port 2a + 1 faces -axis a.  Returns
    the waves arriving *at* each voxel on each port, (*grid, 2N, H).  A wave
    that leaves toward a wall comes back into the same port, scaled by
    `wall_reflection`.
    """
    n_dims = heads.ndim - 2
    incoming = []
    for a in range(n_dims):
        plus = heads[..., 2 * a, :]
        minus = heads[..., 2 * a + 1, :]
        size = plus.shape[a]

        # arriving on the +a port: travelling -a, from the neighbour at v + 1
        from_plus = jnp.concatenate([
            jax.lax.slice_in_dim(minus, 1, size, axis=a),
            wall_reflection * jax.lax.slice_in_dim(plus, size - 1, size, axis=a)], axis=a)

        # arriving on the -a port: travelling +a, from the neighbour at v - 1
        from_minus = jnp.concatenate([
            wall_reflection * jax.lax.slice_in_dim(minus, 0, 1, axis=a),
            jax.lax.slice_in_dim(plus, 0, size - 1, axis=a)], axis=a)

        incoming += [from_plus, from_minus]
    return jnp.stack(incoming, axis=-2)


def simulate(
        transfer: jnp.ndarray,
        source_signals: jnp.ndarray,
        source_idx: jnp.ndarray,
        mic_idx: jnp.ndarray,
        hop: int,
        wall_reflection: float = 0.9) -> jnp.ndarray:
    """
    transfer        (*grid, L/2 + 1), real-valued, in [0, 1]
    source_signals  (n_sources, n_samples), injected into the junction pressure
    source_idx      (n_sources, N) integer voxel coordinates
    mic_idx         (n_mics, N) integer voxel coordinates

    Returns the junction pressure at each mic, (n_mics, n_samples).
    `n_samples` must be a multiple of `hop`.
    """
    grid = transfer.shape[:-1]
    n_dims = len(grid)
    n_sources, n_samples = source_signals.shape
    n_hops = n_samples // hop

    fir = linear_phase_fir(transfer)
    fir_len = fir.shape[-1]
    # length of hop * fir, linear convolution
    conv_len = hop + fir_len - 1
    fir_spec = jnp.fft.rfft(fir, n=conv_len, axis=-1)[..., None, :]

    src = tuple(source_idx.T)
    mic = tuple(mic_idx.T)

    def step(pending, u_block):
        heads = pending[..., :hop]
        pending = jnp.concatenate([pending[..., hop:], jnp.zeros_like(heads)], axis=-1)

        incoming = gather_incoming(heads, wall_reflection)

        u = jnp.zeros(grid + (hop,)).at[src].add(u_block)
        p = incoming.sum(axis=-2) / n_dims + u
        outgoing = p[..., None, :] - incoming

        # linear convolution with each voxel's FIR; the extra hop of delay
        # comes from these samples only being read starting next step
        filtered = jnp.fft.irfft(jnp.fft.rfft(outgoing, n=conv_len, axis=-1) * fir_spec, n=conv_len, axis=-1)
        pending = pending + filtered

        return pending, p[mic]

    pending = jnp.zeros(grid + (2 * n_dims, conv_len))
    blocks = source_signals.reshape(n_sources, n_hops, hop).transpose(1, 0, 2)
    _, recorded = jax.lax.scan(step, pending, blocks)
    # (n_hops, n_mics, hop) -> (n_mics, n_samples)
    return recorded.transpose(1, 0, 2).reshape(mic_idx.shape[0], n_samples)


def to_voxel(position_m, dx: float, grid) -> np.ndarray:
    idx = np.floor(np.array(position_m) / dx).astype(np.int32)
    return np.clip(idx, 0, np.array(grid) - 1)


def stft_magnitude(x: np.ndarray, window: int = 512, hop: int = 128) -> np.ndarray:
    n_frames = (len(x) - window) // hop + 1
    idx = np.arange(n_frames)[:, None] * hop + np.arange(window)[None, :]
    return np.abs(np.fft.rfft(x[idx] * np.hanning(window), axis=-1))


def measured_arrival(source: np.ndarray, recorded: np.ndarray) -> int:
    """
    Lag (in samples) of the peak cross-correlation between the source burst and
    the recording, i.e. where the bulk of the direct sound lands.
    """
    n = len(source) + len(recorded)
    xc = np.fft.irfft(np.fft.rfft(recorded, n=n) * np.conj(np.fft.rfft(source, n=n)), n=n)
    return int(np.argmax(np.abs(xc[:len(recorded)])))


def max_norm(audio: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    return audio / (np.abs(audio).max() + epsilon)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--hop', type=int, default=8, help='samples per simulation step')
    parser.add_argument(
        '--filter-length', type=int, default=16,
        help='transfer functions have filter-length / 2 + 1 bins; link delay is hop + filter-length / 2')
    parser.add_argument('--dims', type=int, choices=[2, 3], default=3)
    parser.add_argument('--room', type=float, nargs='+', default=[8., 6., 4.], help='room size in metres, per axis')
    parser.add_argument('--source', type=float, nargs='+', default=[2., 3., 2.], help='metres')
    parser.add_argument('--mic', type=float, nargs='+', default=[6., 3., 2.], help='metres')
    parser.add_argument('--seconds', type=float, default=3.)
    parser.add_argument('--burst-ms', type=float, default=30.)
    parser.add_argument('--wall-reflection', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=20., help='degrees C')
    parser.add_argument('--humidity', type=float, default=50., help='percent relative humidity')
    parser.add_argument('--air-absorption-scale', type=float, default=1., help='exaggerate the air absorption')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=9999)
    parser.add_argument('--conjure-collection', type=str, default='jax_roomsim')
    parser.add_argument('--wipe-conjure-data', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--no-serve', action='store_true', help='just simulate and print diagnostics')
    args = parser.parse_args()

    n = args.dims
    room, source_m, mic_m = (np.array(v[:n]) for v in (args.room, args.source, args.mic))
    if args.filter_length % 2:
        parser.error('--filter-length must be even')

    sr = args.samplerate
    c = speed_of_sound(args.temperature)
    link_delay = args.hop + args.filter_length // 2
    dx = c * link_delay * np.sqrt(n) / sr
    grid = tuple(int(g) for g in np.maximum(1, np.round(room / dx)))

    n_bins = args.filter_length // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    # sound travels c * D / sr = dx / sqrt(N) metres per link-step
    alpha = air_absorption_db_per_m(freqs, args.temperature, args.humidity) * args.air_absorption_scale
    air = 10 ** (-alpha * (dx / np.sqrt(n)) / 20)
    transfer = jnp.broadcast_to(jnp.array(air, dtype=jnp.float32), grid + (n_bins,))

    n_samples = int(args.seconds * sr) // args.hop * args.hop
    burst_len = int(args.burst_ms / 1000 * sr)
    noise = jax.random.uniform(jax.random.key(args.seed), (burst_len,), minval=-1, maxval=1) * hann_window(burst_len)
    source = jnp.zeros(n_samples).at[:burst_len].set(noise)

    source_idx = to_voxel(source_m, dx, grid)
    mic_idx = to_voxel(mic_m, dx, grid)
    distance = np.linalg.norm((mic_idx - source_idx) * dx)

    print(f'speed of sound {c:.1f} m/s, link delay {link_delay} samples ({1000 * link_delay / sr:.2f} ms)')
    print(f'voxel size {dx:.3f} m, grid {grid} ({np.prod(grid)} voxels), room {np.array(grid) * dx} m')
    print(f'{n_bins} transfer bins, {sr / args.filter_length:.0f} Hz apart; '
          f'mesh spectrum repeats every {sr / link_delay:.0f} Hz; wave-accurate below ~{sr / (4 * link_delay):.0f} Hz')
    print('air transfer per link-step: ' + ', '.join(f'{f:.0f}Hz {t:.5f}' for f, t in zip(freqs, air)))
    print(f'source voxel {source_idx}, mic voxel {mic_idx}')

    sim = jax.jit(simulate, static_argnames=('hop', 'wall_reflection'))
    recorded = sim(
        transfer, source[None, :], jnp.array(source_idx[None]), jnp.array(mic_idx[None]),
        hop=args.hop, wall_reflection=args.wall_reflection)
    recorded = np.asarray(recorded[0])
    print('DONE')

    # the burst arrives at the mic `distance / c` seconds after it starts
    expected = distance / c * sr
    arrival = measured_arrival(np.asarray(noise), recorded)
    print(f'direct path {distance:.2f} m: expected arrival {1000 * expected / sr:.2f} ms, '
          f'measured {1000 * arrival / sr:.2f} ms, error {arrival - expected:+.1f} samples '
          f'({(arrival - expected) / link_delay:+.2f} link-steps)')

    if args.no_serve:
        return

    collection = LmdbCollection(path=args.conjure_collection)
    if args.wipe_conjure_data:
        print('Wiping previous experiment data')
        collection.destroy()
        collection = LmdbCollection(path=args.conjure_collection)

    source_logger, mic_logger = loggers(
        ['source', 'mic'],
        'audio/wav',
        lambda x: encode_audio(x, samplerate=sr),
        collection)

    spec_logger, = loggers(
        ['mic-spectrogram'],
        SupportedContentType.Spectrogram.value,
        lambda x: x,
        collection,
        NumpySerializer(),
        NumpyDeserializer())

    server = serve_conjure(
        [source_logger, mic_logger, spec_logger],
        port=args.port,
        n_workers=1,
        web_components_version='0.0.101')

    source_logger(max_norm(np.asarray(source)))
    mic_logger(max_norm(recorded))
    spec = np.log1p(stft_magnitude(recorded) * 100)
    spec_logger(spec / (spec.max() + 1e-12))

    print(f'Listening at http://localhost:{args.port}  (ctrl-c to exit)')
    input('Hit any key to finish')


if __name__ == '__main__':
    main()
