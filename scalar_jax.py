"""
An isolated test of a simple hypothesis: positioning a signal with a single,
scalar FFT shift is only learnable by gradient descent when the shifted
signal and its target overlap.  If the *loss* looks at the two signals at
many scales, all the way down to two samples, they overlap at *some* scale,
so the coarse scales should steer the shift into a basin that the fine
scales can then refine.

Each trial gets its own gammatone target, placed at a random (fractional)
position in a circular buffer.  The model renders a template starting at
sample 0 and rolls it by one learned scalar via `fft_shift`.  Many
independent trials run at once (vmapped); we report how many of them land on
the right answer.

  template
    fixed      the target's own gammatone; only the shift is learned (default).
    frequency  a gammatone whose center frequency is learned along with the
               shift, starting from a random frequency.
    free       a block of `--template-size` raw samples, learned along with
               the shift, starting from low-level noise.

What the multiscale loss looks at matters as much as the scales:

  representation
    raw          the signal itself.  Its carrier gives a local minimum at every
                 carrier period of misalignment wherever the two signals
                 overlap, and coarse scales low-pass it away.
    energy       x ** 2.  Still carries a ripple at twice the carrier
                 frequency, so it has a local minimum every carrier
                 *half*-period.
    envelope     |analytic signal| ** 2, which has no carrier at all.  For a
                 single, known template, matching the envelope implies
                 matching the signal, but it can't see the carrier, so it
                 can't learn one (default for `--template fixed`).
    spectrogram  STFT power (or magnitude): one carrier-free envelope per
                 frequency bin, so it locates energy in frequency as well as
                 time (default for the learned templates).  Phase is ignored.
                 `--spectrogram-exponent 2` (power, default) matters: with a
                 hann window at 75% overlap, the power spectrogram's total is
                 exactly proportional to the signal's energy (Parseval), so
                 it can't vary with shift or frequency.  The magnitude
                 spectrogram's total ripples slightly with position relative
                 to the hop grid and frequency relative to bin centers, and
                 wherever the two signals don't overlap, the L1 loss is just
                 the sum of the totals, so each ripple becomes a local
                 minimum.  (With magnitude, the free template barely learned.)

  scales
    blur      undecimated gaussian blurs with std 1, 2, 4 ... up to a quarter
              of the axis, plus the unblurred signal (default).  Every level
              is shift-equivariant, so the loss is a smooth function of
              shift.  For `spectrogram`, the blur runs along time (circular)
              and, with `--freq-blur` (default), along frequency too, with
              the same std in bins as in frames at each level.
    pyramid   binomial smoothing + decimation by 2, down to two samples.
              Decimation makes coarse levels shift-variant, which adds kinks
              (and occasionally spurious minima) on the grid of each level.
              1-D representations only.
    single    the finest scale only; the no-multiscale baseline.

Every level is compared with L1; levels preserve the signal's sum, so the
per-level losses are on comparable scales.  (L2 did much worse in a quick
check: far more spurious minima.)

A trial is solved when:
  fixed      the shift is within `--tolerance` samples
  frequency  ... and the frequency is within `--cents-tolerance` cents
  free       the envelopes align (circular cross-correlation peak) within
             `--tolerance` samples, and the relative L1 error of the
             STFT magnitudes is below `--spec-tolerance`.  The free template
             can put its content anywhere in its block, so the shift itself
             isn't comparable to the true position.

    python scalar_jax.py                              # fixed template, envelope blurs
    python scalar_jax.py --scales single              # baseline: finest scale only
    python scalar_jax.py --representation raw         # multiscale, but on the raw signal
    python scalar_jax.py --template frequency         # learn frequency + shift
    python scalar_jax.py --template frequency --no-freq-blur
    python scalar_jax.py --template free              # learn raw samples + shift
    python scalar_jax.py --landscape                  # plot loss vs. shift (fixed) or
                                                      # vs. shift and frequency (frequency)
"""

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax


def gammatone(n_samples: int, samplerate: int, freq, order: int = 4) -> jnp.ndarray:
    """
    A gammatone starting at sample 0, peak-normalized, zero-padded to
    `n_samples`.  Differentiable with respect to `freq`.
    """
    erb = 24.7 + 0.108 * freq
    b = 1.019 * erb
    t = jnp.arange(n_samples) / samplerate
    env = t ** (order - 1) * jnp.exp(-2 * jnp.pi * b * t)
    g = env * jnp.cos(2 * jnp.pi * freq * t)
    return g / (jnp.abs(g).max() + 1e-12)


def fft_shift(x: jnp.ndarray, shift: jnp.ndarray) -> jnp.ndarray:
    """
    Circularly shift `x` (along its last axis) forward by `shift` samples,
    which may be fractional, via a linear phase ramp in the frequency
    domain.  Matches `hyper_rnn.fft_shift`.
    """
    n = x.shape[-1]
    spec = jnp.fft.rfft(x)
    k = jnp.arange(spec.shape[-1])
    spec = spec * jnp.exp(-2j * jnp.pi * k * shift / n)
    return jnp.fft.irfft(spec, n=n)


def hann_window(n: int) -> jnp.ndarray:
    return 0.5 - 0.5 * jnp.cos(2 * jnp.pi * jnp.arange(n) / n)


# 5-tap binomial kernel, scaled to sum to 2, so that each smooth-and-decimate
# step preserves the total (sum) of the signal; the L1 loss at every level is
# then on a comparable scale
BINOMIAL = jnp.array([1., 4., 6., 4., 1.]) / 8


def downsample(x: jnp.ndarray) -> jnp.ndarray:
    """
    Circularly smooth with the binomial kernel, then keep every other sample.
    """
    smoothed = sum(w * jnp.roll(x, i - 2) for i, w in enumerate(BINOMIAL))
    return smoothed[::2]


def pyramid(x: jnp.ndarray, min_size: int = 2):
    levels = [x]
    while levels[-1].shape[-1] > min_size:
        levels.append(downsample(levels[-1]))
    return levels


def gaussian_blur(x: jnp.ndarray, sigma: float, axis: int, circular: bool) -> jnp.ndarray:
    """
    Gaussian blur along `axis` with std `sigma` (in samples), applied in the
    frequency domain.  DC gain is 1, so the sum is preserved, except for
    what leaks off the ends when not circular (the axis is zero-padded).
    """
    axis = axis % x.ndim
    m = x.shape[axis]
    size = m
    if not circular:
        pad = [(0, 0)] * x.ndim
        pad[axis] = (0, m)
        x = jnp.pad(x, pad)
        size = 2 * m
    spec = jnp.fft.rfft(x, axis=axis)
    k = jnp.arange(spec.shape[axis])
    gain = jnp.exp(-0.5 * (2 * jnp.pi * k * sigma / size) ** 2)
    gain = gain.reshape([-1 if i == axis else 1 for i in range(x.ndim)])
    y = jnp.fft.irfft(spec * gain, n=size, axis=axis)
    return y if circular else jax.lax.slice_in_dim(y, 0, m, axis=axis)


def blur_levels(x: jnp.ndarray):
    """
    The signal, then circular gaussian blurs of it with std 1, 2, 4 ... n/4
    samples, all at full resolution.
    """
    n = x.shape[-1]
    return [x] + [gaussian_blur(x, 2.0 ** i, -1, circular=True) for i in range(int(np.log2(n)) - 1)]


def spectrogram_blur_levels(spec: jnp.ndarray, freq_blur: bool):
    """
    `spec` is (frames, bins).  Level i blurs time (circularly) with std 2 ** i
    frames, up to a quarter of the frames, and, with `freq_blur`, frequency
    with std 2 ** i bins, up to a quarter of the bins.
    """
    n_frames, n_bins = spec.shape
    n_time = int(np.log2(n_frames)) - 1
    n_freq = int(np.log2(n_bins)) - 1 if freq_blur else 0
    levels = [spec]
    for i in range(max(n_time, n_freq)):
        x = spec
        x = gaussian_blur(x, 2.0 ** min(i, n_time - 1), 0, circular=True)
        if freq_blur:
            x = gaussian_blur(x, 2.0 ** min(i, n_freq - 1), 1, circular=False)
        levels.append(x)
    return levels


def analytic_envelope(x: jnp.ndarray) -> jnp.ndarray:
    """
    Squared magnitude of the analytic signal (positive frequencies doubled,
    negative frequencies zeroed), i.e. the energy envelope with no carrier.
    """
    n = x.shape[-1]
    spec = jnp.fft.fft(x)
    k = jnp.arange(n)
    gain = jnp.where(k == 0, 1., jnp.where(k < n // 2, 2., jnp.where(k == n // 2, 1., 0.)))
    return jnp.abs(jnp.fft.ifft(spec * gain)) ** 2


def stft_magnitude(x: jnp.ndarray, window: int, hop: int, exponent: float = 1.) -> jnp.ndarray:
    """
    Circular STFT magnitude ** `exponent`, (frames, bins).  Frames wrap
    around the buffer, so a shift by a multiple of `hop` is exactly a roll
    of the frames.
    """
    n = x.shape[-1]
    idx = (jnp.arange(n // hop)[:, None] * hop + jnp.arange(window)[None, :]) % n
    frames = x[idx] * hann_window(window)
    return jnp.abs(jnp.fft.rfft(frames, axis=-1, norm='ortho')) ** exponent


def make_per_scale_losses(
        representation: str,
        scales: str,
        window: int,
        hop: int,
        freq_blur: bool,
        spectrogram_exponent: float = 2.):
    """
    Returns `per_scale_losses(recon, target)`: the L1 distance at each
    scale, finest first.
    """
    if representation == 'spectrogram' and scales == 'pyramid':
        raise ValueError('--scales pyramid only supports 1-D representations')

    def represent(x):
        if representation == 'raw':
            return x
        if representation == 'energy':
            return x ** 2
        if representation == 'envelope':
            return analytic_envelope(x)
        if representation == 'spectrogram':
            return stft_magnitude(x, window, hop, spectrogram_exponent)
        raise ValueError(f'Unknown representation {representation}')

    def levels(x):
        if scales == 'single':
            return [x]
        if representation == 'spectrogram':
            return spectrogram_blur_levels(x, freq_blur)
        if scales == 'blur':
            return blur_levels(x)
        if scales == 'pyramid':
            return pyramid(x)
        raise ValueError(f'Unknown scales {scales}')

    def per_scale_losses(recon, target):
        rl, tl = levels(represent(recon)), levels(represent(target))
        return jnp.stack([jnp.abs(a - b).sum() for a, b in zip(rl, tl)])

    return per_scale_losses


def circular_error(estimate: jnp.ndarray, truth: jnp.ndarray, n: int) -> jnp.ndarray:
    """
    Signed position error, in samples, wrapped into [-n/2, n/2).
    """
    return (estimate - truth + n / 2) % n - n / 2


def alignment_offset(recon: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """
    The signed (integer) lag that best aligns `recon`'s envelope with
    `target`'s, via circular cross-correlation; 0 means aligned.
    """
    n = recon.shape[-1]
    er, et = analytic_envelope(recon), analytic_envelope(target)
    xc = jnp.fft.irfft(jnp.fft.rfft(et) * jnp.conj(jnp.fft.rfft(er)), n=n)
    lag = jnp.argmax(xc)
    return (lag + n // 2) % n - n // 2


def log_uniform(key, shape, low, high):
    return jnp.exp(jax.random.uniform(key, shape, minval=np.log(low), maxval=np.log(high)))


def make_trials(key, args):
    n = args.n_samples
    fkey, pkey, ikey, gkey, tkey = jax.random.split(key, 5)
    freqs = log_uniform(fkey, (args.n_trials,), args.min_freq, args.max_freq)
    true_positions = jax.random.uniform(pkey, (args.n_trials,), minval=0, maxval=n)
    targets = jax.vmap(lambda f, p: fft_shift(gammatone(n, args.samplerate, f), p))(freqs, true_positions)

    # the shift is learned as a fraction of the buffer
    params = {'shift': jax.random.uniform(ikey, (args.n_trials,))}
    if args.template == 'frequency':
        params['log_freq'] = jnp.log(log_uniform(gkey, (args.n_trials,), args.min_freq, args.max_freq))
    elif args.template == 'free':
        params['samples'] = jax.random.normal(tkey, (args.n_trials, args.template_size)) * args.template_init_scale
    return freqs, targets, true_positions, params


def make_render(args, freqs):
    """
    Returns `render(params) -> (n_trials, n_samples)`.
    """
    n = args.n_samples

    def templates(params):
        if args.template == 'fixed':
            return jax.vmap(lambda f: gammatone(n, args.samplerate, f))(freqs)
        if args.template == 'frequency':
            return jax.vmap(lambda lf: gammatone(n, args.samplerate, jnp.exp(lf)))(params['log_freq'])
        if args.template == 'free':
            return jnp.pad(params['samples'], ((0, 0), (0, n - args.template_size)))
        raise ValueError(f'Unknown template {args.template}')

    def render(params):
        return jax.vmap(fft_shift)(templates(params), params['shift'] * n)

    return render


def plot_landscape(args, per_scale_losses, freqs, targets, true_positions):
    from matplotlib import pyplot as plt

    n = args.n_samples
    target = targets[0]
    print(f'trial 0: {float(freqs[0]):.1f} Hz, true position {float(true_positions[0]):.2f}')

    if args.template == 'fixed':
        template = gammatone(n, args.samplerate, freqs[0])
        shifts = jnp.linspace(0, n, 2048, endpoint=False)
        losses = np.array(jax.lax.map(
            lambda s: per_scale_losses(fft_shift(template, s), target), shifts, batch_size=64))
        # normalize each level to [0, 1] so they can share an axis
        lo, hi = losses.min(axis=0), losses.max(axis=0)
        normed = (losses - lo) / (hi - lo + 1e-12)

        fig, (a, b) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        for i in range(normed.shape[1]):
            a.plot(shifts, normed[:, i] + i, linewidth=0.8)
        a.set_ylabel('per-level loss (normalized, offset; finest at bottom)')
        b.plot(shifts, losses.sum(axis=1))
        b.set_ylabel('total loss')
        b.set_xlabel('shift (samples)')
        for ax in (a, b):
            ax.axvline(float(true_positions[0]), color='k', linestyle='--', linewidth=0.8)
    else:
        shifts = jnp.linspace(0, n, 256, endpoint=False)
        log_freqs = jnp.linspace(np.log(args.min_freq), np.log(args.max_freq), 64)
        grid = jnp.stack(jnp.meshgrid(shifts, log_freqs, indexing='ij'), axis=-1).reshape(-1, 2)

        def total(p):
            recon = fft_shift(gammatone(n, args.samplerate, jnp.exp(p[1])), p[0])
            return per_scale_losses(recon, target).sum()

        losses = np.array(jax.lax.map(total, grid, batch_size=64)).reshape(len(shifts), len(log_freqs))
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(
            losses.T, origin='lower', aspect='auto',
            extent=[0, n, np.log2(args.min_freq), np.log2(args.max_freq)])
        ax.plot(float(true_positions[0]), np.log2(float(freqs[0])), 'rx')
        ax.set_xlabel('shift (samples)')
        ax.set_ylabel('log2 frequency (Hz)')
        fig.colorbar(im, label='total loss')
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n-samples', type=int, default=2 ** 15, help='circular buffer size; a power of two')
    parser.add_argument('--samplerate', type=int, default=22050)
    parser.add_argument('--min-freq', type=float, default=100.)
    parser.add_argument('--max-freq', type=float, default=5000.)
    parser.add_argument('--template', choices=['fixed', 'frequency', 'free'], default='fixed')
    parser.add_argument('--template-size', type=int, default=2048, help='--template free: learned block size')
    parser.add_argument('--template-init-scale', type=float, default=0.01, help='--template free: std of the initial noise')
    parser.add_argument(
        '--representation', choices=['raw', 'energy', 'envelope', 'spectrogram'], default=None,
        help='default: envelope for --template fixed, spectrogram otherwise')
    parser.add_argument('--scales', choices=['blur', 'pyramid', 'single'], default='blur')
    parser.add_argument('--window', type=int, default=512, help='spectrogram window size')
    parser.add_argument('--hop', type=int, default=128, help='spectrogram hop size')
    parser.add_argument(
        '--spectrogram-exponent', type=float, default=2.,
        help='2: power spectrogram (default); 1: magnitude')
    parser.add_argument(
        '--freq-blur', action=argparse.BooleanOptionalAction, default=True,
        help='spectrogram blur levels also blur along frequency')
    parser.add_argument('--n-trials', type=int, default=64)
    parser.add_argument('--n-iterations', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-2, help='initial shift lr, in units of the buffer size')
    parser.add_argument('--freq-lr', type=float, default=1e-2, help='initial lr for log frequency')
    parser.add_argument('--template-lr', type=float, default=1e-2, help='initial lr for free template samples')
    parser.add_argument(
        '--alternate', type=int, default=0,
        help='learned templates: alternate between updating only the shift and only the template, '
             'switching every this many iterations, starting with the shift; 0 (default) updates both together')
    parser.add_argument('--final-lr-ratio', type=float, default=1e-5, help='cosine decay ends at lr * this')
    parser.add_argument('--tolerance', type=float, default=0.5, help='position tolerance, in samples')
    parser.add_argument('--cents-tolerance', type=float, default=10.)
    parser.add_argument('--spec-tolerance', type=float, default=0.1)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--landscape', action='store_true',
        help='plot the loss for trial 0 (vs. shift, or vs. shift and frequency) and exit')
    args = parser.parse_args()

    n = args.n_samples
    if n & (n - 1):
        parser.error('--n-samples must be a power of two')
    if n % args.hop:
        parser.error('--hop must divide --n-samples')
    if args.template_size > n:
        parser.error('--template-size must not exceed --n-samples')
    if args.representation is None:
        args.representation = 'envelope' if args.template == 'fixed' else 'spectrogram'
    if args.representation == 'spectrogram' and args.scales == 'pyramid':
        parser.error('--scales pyramid only supports 1-D representations')
    if args.alternate < 0:
        parser.error('--alternate must be >= 0')
    if args.alternate and args.template == 'fixed':
        parser.error('--alternate needs a learned template')
    if args.landscape and args.template == 'free':
        parser.error('--landscape supports --template fixed or frequency')

    per_scale_losses = make_per_scale_losses(
        args.representation, args.scales, args.window, args.hop, args.freq_blur, args.spectrogram_exponent)

    freqs, targets, true_positions, params = make_trials(jax.random.key(args.seed), args)

    if args.landscape:
        plot_landscape(args, per_scale_losses, freqs, targets, true_positions)
        return

    render = make_render(args, freqs)

    def loss_func(params):
        losses = jax.vmap(per_scale_losses)(render(params), targets)
        # trials are independent, so summing keeps each one's gradient its own
        return losses.sum()

    # which parameter group(s) each iteration updates
    template_keys = [k for k in params if k != 'shift']
    if args.alternate:
        phases = [
            ['shift'] if (i // args.alternate) % 2 == 0 else template_keys
            for i in range(args.n_iterations)]
    else:
        phases = [list(params)] * args.n_iterations

    lrs = {'shift': args.lr, 'log_freq': args.freq_lr, 'samples': args.template_lr}
    # one optimizer per parameter, so a frozen parameter's adam state and lr
    # schedule only advance on the iterations that actually update it; adam
    # is elementwise, so every trial effectively has its own optimizer too
    optims = {
        k: optax.adam(optax.cosine_decay_schedule(
            lrs[k], max(1, sum(k in p for p in phases)), alpha=args.final_lr_ratio))
        for k in params}
    opt_state = {k: optims[k].init(params[k]) for k in params}

    def make_step(keys):
        @jax.jit
        def step(params, opt_state):
            loss, grads = jax.value_and_grad(loss_func)(params)
            params, opt_state = dict(params), dict(opt_state)
            for k in keys:
                updates, opt_state[k] = optims[k].update(grads[k], opt_state[k], params[k])
                params[k] = optax.apply_updates(params[k], updates)
            return params, opt_state, loss
        return step

    steps = {tuple(p): make_step(p) for p in phases}

    @jax.jit
    def metrics(params):
        recon = render(params)
        st = jax.vmap(lambda x: stft_magnitude(x, args.window, args.hop))(targets)
        sr = jax.vmap(lambda x: stft_magnitude(x, args.window, args.hop))(recon)
        spec_err = jnp.abs(sr - st).sum(axis=(1, 2)) / st.sum(axis=(1, 2))
        if args.template == 'free':
            pos_err = jax.vmap(alignment_offset)(recon, targets).astype(jnp.float32)
        else:
            pos_err = circular_error(params['shift'] * n, true_positions, n)
        cents = (
            1200 * jnp.log2(jnp.exp(params['log_freq']) / freqs)
            if args.template == 'frequency' else jnp.zeros_like(freqs))
        return pos_err, cents, spec_err

    def solved_mask(pos_err, cents, spec_err):
        solved = np.abs(pos_err) < args.tolerance
        if args.template == 'frequency':
            solved &= np.abs(cents) < args.cents_tolerance
        if args.template == 'free':
            solved &= spec_err < args.spec_tolerance
        return solved

    def report(i, params, loss):
        pos_err, cents, spec_err = (np.array(m) for m in metrics(params))
        solved = solved_mask(pos_err, cents, spec_err)
        line = (
            f'{i:6d}  loss {float(loss):12.4f}  '
            f'|{"offset" if args.template == "free" else "err"}| '
            f'median {np.median(np.abs(pos_err)):9.2f} max {np.abs(pos_err).max():9.2f}  ')
        if args.template == 'frequency':
            line += f'|cents| median {np.median(np.abs(cents)):8.1f} max {np.abs(cents).max():8.1f}  '
        line += f'spec err median {np.median(spec_err):.3f}  solved {solved.sum()}/{len(solved)}'
        print(line, flush=True)
        return pos_err, cents, spec_err

    print(
        f'{args.n_trials} trials, buffer {n} samples, template {args.template}, '
        f'representation {args.representation}, scales {args.scales}'
        + (f' (freq blur {"on" if args.freq_blur else "off"})'
           if args.representation == 'spectrogram' and args.scales == 'blur' else '')
        + (f', alternating every {args.alternate}' if args.alternate else ''),
        flush=True)

    result = report(0, params, loss_func(params))
    for i in range(1, args.n_iterations + 1):
        params, opt_state, loss = steps[tuple(phases[i - 1])](params, opt_state)
        if i % args.log_every == 0 or i == args.n_iterations:
            result = report(i, params, loss)

    pos_err, cents, spec_err = result
    unsolved = np.nonzero(~solved_mask(pos_err, cents, spec_err))[0]
    if len(unsolved):
        print('\nunsolved trials (target freq, position error/offset, cents, spec err):')
        for t in unsolved:
            print(
                f'  {int(t):4d}  {float(freqs[t]):8.1f} Hz  {pos_err[t]:10.2f}  '
                f'{cents[t]:8.1f}  {spec_err[t]:.3f}')


if __name__ == '__main__':
    main()
