"""
A sample-rate damped harmonic oscillator (resonator) bank that can be driven
by *any* forcing signal -- impulses (struck/plucked), noise (blown), or a
sawtooth (bowed/scraped) -- computed in parallel, with finite gradients for
any damping.

Why not `envelope * cos(phase)`?  `parallel.py`'s `damped_harmonic_oscillator`
(and the web demo's `stepNode`) leaky-integrates the forcing into an
envelope, then multiplies by a sinusoid at the node's frequency.  That
hears the forcing only through its content near DC, shifted up to the
node's frequency.  A resonator hears the forcing's content *near its own
frequency*.  For white noise (and impulses) the two are statistically the
same -- a frequency shift doesn't change white noise -- but for anything
with spectral structure they're not: a 196Hz sawtooth (a bow) driving a
node at 392Hz excites the resonator strongly, as its 2nd harmonic lands on
the node, while the envelope form barely responds, and mostly not at
392Hz.  Likewise, a resonator can be driven by another node's *audio*
(sympathetic resonance, body filtering), not just its control signals.

A damped oscillator, sampled, is a complex one-pole filter:

    z[n] = exp(-alpha[n] + i * omega[n]) * z[n - 1] + g[n] * f[n]
    x[n] = Im(z[n])

alpha: decay, nepers per sample; omega: frequency, radians per sample.
Driven by a unit impulse, x is exp(-alpha n) sin(omega n): a mass on a
spring given a velocity kick.  Both may vary per sample, so tension
(pitch) and damping modulation -- including audio-rate FM from another
layer -- are exact, with phase integrated (as the web demo does)
rather than computed as omega * t (as parallel.py does).  g = 1 - exp(-alpha) normalizes the
resonant peak gain to ~1, so loudness doesn't explode as damping -> 0.

It's a linear recurrence, h[n] = a[n] h[n-1] + b[n], so it runs as a
parallel scan (`linear_recurrence`).  Unlike `chunked_parallel`, which
divides by a running product of decay factors (the product underflows,
and the gradient of the division -- which squares it -- overflows), this
never divides:

  - within short chunks, sum in log space, with the chunk size chosen so
    the decay within a chunk spans at most exp(MAX_CHUNK_DECAY);
  - across chunks, a Hillis-Steele (doubling) scan, which only multiplies
    by decay factors of magnitude <= 1.

Run as a script for checks: agreement with a sequential reference, finite
gradients at extreme damping (where `parallel2.chunked_parallel`'s are
NaN), timing, and a few rendered examples of each excitation:

    python sample_rate_dho.py [examples output dir]
"""

import math
from typing import Optional

import torch
from torch.nn import functional as F

# the most a chunk's decay may span, in nepers: exp(30) ~ 1e13, leaving
# float32 (max ~3e38) ample headroom for the forcing's own magnitude
MAX_CHUNK_DECAY = 30.

# the longest chunk: within a chunk, phase accumulates to omega * chunk
# radians before cancelling, and float32's precision falls with the size
# of the angle (~6e-5 rad at 800 rad)
MAX_CHUNK = 256


# ---------------------------------------------------------------------------
# the scan

def _chunk_size(log_a: torch.Tensor) -> int:
    """the largest power of two (<= MAX_CHUNK) whose decay spans at most MAX_CHUNK_DECAY"""
    max_decay = float((-log_a.real.detach()).max().clamp(min=1e-12))
    size = 2 ** int(math.floor(math.log2(max(MAX_CHUNK_DECAY / max_decay, 1.))))
    return int(min(size, MAX_CHUNK))


def _shift(x: torch.Tensor, offset: int, fill: float) -> torch.Tensor:
    """x shifted right by `offset` along the last axis, filled with `fill`"""
    pad = torch.full((*x.shape[:-1], offset), fill, dtype=x.dtype, device=x.device)
    return torch.cat([pad, x[..., :-offset]], dim=-1)


def linear_recurrence(
        log_a: torch.Tensor,
        b: torch.Tensor,
        chunk_size: Optional[int] = None) -> torch.Tensor:
    """
    h[n] = exp(log_a[n]) * h[n - 1] + b[n], with h[-1] = 0.

    log_a: (..., n), real or complex, with real part <= 0 (|a| <= 1)
    b: broadcastable to log_a's shape
    chunk_size: by default, chosen from log_a (see `_chunk_size`)
    returns: (..., n), complex if either input is
    """
    dtype = torch.promote_types(log_a.dtype, b.dtype)
    log_a = log_a.to(dtype)
    b = torch.broadcast_to(b.to(dtype), log_a.shape)
    n = log_a.shape[-1]
    c = chunk_size or _chunk_size(log_a)

    pad = (-n) % c
    if pad:
        log_a = torch.cat([log_a, torch.zeros(*log_a.shape[:-1], pad, dtype=dtype, device=log_a.device)], -1)
        b = torch.cat([b, torch.zeros(*b.shape[:-1], pad, dtype=dtype, device=b.device)], -1)
    k = log_a.shape[-1] // c
    log_a = log_a.reshape(*log_a.shape[:-1], k, c)
    b = b.reshape(*b.shape[:-1], k, c)

    # within each chunk: h[i] = exp(S[i]) * sum_{j <= i} exp(-S[j]) b[j],
    # S the chunk-local cumulative log decay.  |exp(-S)| <= exp(MAX_CHUNK_DECAY)
    s = torch.cumsum(log_a, dim=-1)
    decay = torch.exp(s)
    local = decay * torch.cumsum(torch.exp(-s) * b, dim=-1)

    # across chunks: each chunk's total decay, and its final local state,
    # combined by an inclusive doubling scan of (a, h) pairs, whose
    # combine (a1, h1), (a2, h2) -> (a1 a2, a2 h1 + h2) only multiplies
    total, state = decay[..., -1], local[..., -1]
    offset = 1
    while offset < k:
        state = state + total * _shift(state, offset, 0.)
        total = total * _shift(total, offset, 1.)
        offset *= 2

    # each chunk starts from the previous chunk's final state
    carry = _shift(state, 1, 0.)
    out = local + carry[..., None] * decay
    return out.reshape(*out.shape[:-2], k * c)[..., :n]


def resonator(
        force: torch.Tensor,
        alpha: torch.Tensor,
        omega: torch.Tensor,
        normalize: bool = True,
        chunk_size: Optional[int] = None) -> torch.Tensor:
    """
    A bank of damped oscillators, each driven by its own force.

    force: (..., n)
    alpha: decay, nepers per sample, >= 0; broadcastable to force's shape
        (e.g. (..., 1) for constant, (..., n) for modulated)
    omega: frequency, radians per sample, in (0, pi); likewise
    normalize: scale the force by 1 - exp(-alpha), so the resonant peak
        gain is ~1, whatever the damping
    returns: (..., n), the displacement
    """
    alpha, omega = torch.broadcast_tensors(alpha, omega)
    log_a = torch.complex(-alpha, omega)
    if normalize:
        force = force * -torch.expm1(-alpha)
    shape = torch.broadcast_shapes(force.shape, log_a.shape)
    z = linear_recurrence(log_a.expand(shape), force, chunk_size)
    return z.imag


# ---------------------------------------------------------------------------
# physical parameterizations

def omega_from_tension(log10_tension: torch.Tensor, sample_rate: int, max_fraction: float = 0.95) -> torch.Tensor:
    """
    Radians per sample from log10(tension), tension = omega0 ** 2 in
    (rad/s) ** 2 (parallel.py's TENSION_EXPONENT_RANGE of (4, 9) spans ~16Hz
    to ~5kHz), capped at `max_fraction` of Nyquist.
    """
    omega = 10 ** (log10_tension / 2) / sample_rate
    return torch.clamp(omega, max=max_fraction * math.pi)


def alpha_from_t60(t60_seconds: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """nepers per sample for a 60dB decay over `t60_seconds`"""
    return math.log(1000.) / (t60_seconds * sample_rate)


# ---------------------------------------------------------------------------
# excitations, (..., n_frames) control signals -> (..., n_samples) forces

def _upsample(x: torch.Tensor, n: int) -> torch.Tensor:
    """(..., n_frames) -> (..., n), linearly interpolated"""
    shape = x.shape
    return F.interpolate(x.reshape(-1, 1, shape[-1]), size=n, mode='linear', align_corners=False).reshape(*shape[:-1], n)


def impulse_excitation(strikes: torch.Tensor, n: int) -> torch.Tensor:
    """
    Struck/plucked: each control frame's value as a single impulse at the
    start of its frame (zero-stuffed, not interpolated, so it stays sharp).
    """
    frames = strikes.shape[-1]
    factor = math.ceil(n / frames)
    out = torch.zeros(*strikes.shape, factor, dtype=strikes.dtype, device=strikes.device)
    out[..., 0] = strikes
    return out.reshape(*strikes.shape[:-1], frames * factor)[..., :n]


def noise_excitation(envelope: torch.Tensor, n: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """
    Blown: unit-variance white noise under a (linearly interpolated)
    envelope.  The resonator turns it into noise band-limited around each
    node's frequency, narrower as damping falls.
    """
    shape = (*envelope.shape[:-1], n)
    noise = (torch.rand(shape, generator=generator, device=envelope.device, dtype=envelope.dtype) * 2 - 1) * math.sqrt(3)
    return _upsample(envelope, n) * noise


def poly_blep(t: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """the polyBLEP correction for a unit step at phase 0, for phase t in [0, 1) advancing dt per sample"""
    a = t / dt
    b = (t - 1) / dt
    return torch.where(t < dt, 2 * a - a * a - 1, torch.where(t > 1 - dt, b * b + 2 * b + 1, torch.zeros_like(t)))


def sawtooth_excitation(
        envelope: torch.Tensor,
        frequency_hz: torch.Tensor,
        sample_rate: int,
        n: int,
        antialias: bool = True) -> torch.Tensor:
    """
    Bowed/scraped: a sawtooth (stick-slip) under an envelope, at a
    (linearly interpolated, modulatable) frequency.  Phase is integrated in
    float64, so it stays exact over long buffers, and gradients reach the
    frequency through it.  polyBLEP removes most of the aliasing.
    """
    dt = torch.clamp(_upsample(frequency_hz, n) / sample_rate, 1e-5, 0.5)
    phase = torch.cumsum(dt.double(), dim=-1)
    t = (phase - torch.floor(phase)).to(envelope.dtype)
    saw = 2 * t - 1
    if antialias:
        saw = saw - poly_blep(t, dt)
    return _upsample(envelope, n) * saw


# ---------------------------------------------------------------------------
# checks

def _sequential(log_a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """the plain recurrence, in float64, as a reference"""
    a = torch.exp(log_a.to(torch.complex128))
    b = torch.broadcast_to(b.to(torch.complex128), a.shape)
    h = torch.zeros(a.shape[:-1], dtype=torch.complex128, device=a.device)
    out = []
    for i in range(a.shape[-1]):
        h = a[..., i] * h + b[..., i]
        out.append(h)
    return torch.stack(out, -1)


def _check(out_dir: str):
    import time
    from pathlib import Path
    from scipy.io import wavfile

    torch.manual_seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sr = 22050

    print('1. agreement with a sequential float64 reference (max |error| / max |reference|)')
    n = 8192
    for label, alpha_max in [('slow decay', 1e-4), ('mixed', 0.05), ('very fast decay', 2.0)]:
        alpha = torch.rand(4, n, device=device) * alpha_max
        omega = torch.rand(4, 1, device=device) * 3 + 0.1 * torch.sin(torch.arange(n, device=device) / 50)
        log_a = torch.complex(-alpha, omega)
        b = torch.randn(4, n, device=device)
        ref = _sequential(log_a, b)
        err = (linear_recurrence(log_a, b).to(torch.complex128) - ref).abs().max() / ref.abs().max()
        print(f'   {label:16s} (alpha <= {alpha_max:g}/sample, chunk {_chunk_size(log_a)}): {err.item():.1e}')

    print('\n2. gradients at the damping that breaks parallel2.chunked_parallel (2^17 samples)')
    n = 2 ** 17
    for d in [0.999, 0.975, 0.5]:
        alpha = torch.full((1, 4, 1), -math.log(d), device=device, requires_grad=True)
        omega = torch.full((1, 4, 1), 0.3, device=device, requires_grad=True)
        force = torch.randn(1, 4, n, device=device, requires_grad=True)
        resonator(force, alpha, omega).pow(2).mean().backward()
        finite = all(torch.isfinite(t.grad).all().item() for t in (alpha, omega, force))
        print(f'   damping {d}/sample: gradients finite {finite}')

    print('\n3. time: forward + backward, batch 2 x 32 nodes x 2^17 samples, time-varying alpha and omega')
    shape = (2, 32, n)
    alpha = (torch.rand(2, 32, 1, device=device) * 1e-3).expand(shape).clone().requires_grad_()
    omega = (torch.rand(2, 32, 1, device=device) * 0.5 + 0.01 * torch.sin(torch.arange(n, device=device) / 1000)).requires_grad_()
    force = torch.randn(shape, device=device, requires_grad=True)
    for i in range(3):
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.time()
        resonator(force, alpha, omega).pow(2).mean().backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        if i == 2:
            peak = f', peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB' if device == 'cuda' else ''
            print(f'   {time.time() - start:.3f}s{peak}')

    print('\n4. rendering examples: 8 nodes, 3s, a harmonic series on 196Hz, each excitation')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 3 * sr
    frames = 64
    freqs = 196. * torch.arange(1, 9, device=device)
    omega = (2 * math.pi * freqs / sr)[:, None]
    alpha = alpha_from_t60(torch.full((8, 1), 1.5, device=device), sr)
    gate = torch.zeros(8, frames, device=device)
    gate[:, 4:44] = 1.
    excitations = {
        'impulse': impulse_excitation(torch.zeros(8, frames, device=device).index_fill_(1, torch.tensor([4, 20, 36], device=device), 1.), n),
        'noise': noise_excitation(gate, n),
        'saw': sawtooth_excitation(gate, torch.full((8, frames), 196., device=device), sr, n),
        # vibrato: omega modulated at audio rate, the exact case the
        # envelope * cos(phase) form handles with phase integration
        'saw_vibrato': None,
    }
    for name, force in excitations.items():
        o = omega
        if name == 'saw_vibrato':
            force = excitations['saw']
            t = torch.arange(n, device=device) / sr
            o = omega * (1 + 0.01 * torch.sin(2 * math.pi * 5.5 * t))
        audio = resonator(force, alpha, o).sum(0)
        audio = (audio / audio.abs().max().clamp(min=1e-8)).cpu().numpy()
        wavfile.write(out_dir / f'{name}.wav', sr, audio.astype('float32'))
    print(f'   wrote {", ".join(f"{k}.wav" for k in excitations)} to {out_dir}')


if __name__ == '__main__':
    import sys
    import tempfile
    _check(sys.argv[1] if len(sys.argv) > 1 else f'{tempfile.gettempdir()}/sample_rate_dho_examples')
