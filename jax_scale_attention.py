"""
A self-attention encoder for real-valued, wide-dynamic-range input (e.g. a
spectrogram residual), normalized at every layer without discarding scale.

The building block is `ScaleAwareNorm`, a self-conditioned AdaLN: each token
is RMS-normalized, and the log of the RMS it divided out is embedded (with
Fourier features) and used, alongside an optional conditioning vector, to
predict the normalized token's scale and shift.  So each sublayer sees
unit-scale input, but still knows how loud that input was.

`AttentionEncoder` is a drop-in alternative to `jax_ae.Encoder`:

    1. Each input frame is divided by its own RMS.  Its log level is kept
       only *relative* to the frames it can see (the RMS of its visible
       context), embedded, added to the residual stream, and used to
       condition every `ScaleAwareNorm`.  Pairwise level differences
       between frames also bias each head's attention logits (zero at
       init), so a head can learn to attend to louder or quieter frames.
    2. Pre-norm transformer blocks: `ScaleAwareNorm` -> multi-head attention
       with QK-norm (bounded attention logits, whatever the stream's scale)
       and rotary position embeddings -> residual; `ScaleAwareNorm` -> MLP ->
       residual.  The residual stream itself is never normalized.
    3. A final `ScaleAwareNorm` readout.  The encoder also returns each
       frame's absolute log level, so heads that need absolute scale
       (magnitude) can read it directly.

With `level_mode='relative'` (the default), absolute level enters nowhere
but that returned log level, so the features are invariant to a global gain
(magnitude can be exactly equivariant), and with the anti-causal mask, a
gain applied to one region changes nothing about the frames after it.
`level_mode='absolute'` also embeds each frame's absolute log level, giving
up that invariance, for when absolute level means something.
"""

import math
from typing import NamedTuple, Optional

import equinox as eqx
import jax
import jax.numpy as jnp


# frames are floored this far (in energy; -80 dB) below the loudest frame
# they can see; see `AttentionEncoder.__call__`
RELATIVE_FLOOR = 1e-8


def fourier_embed(s: jnp.ndarray, lo: float, hi: float, n_freqs: int) -> jnp.ndarray:
    """
    Embed a scalar `s`, clipped to [lo, hi] and rescaled to [-1, 1], as
    (2 * n_freqs + 1,): itself, then sines and cosines at octave-spaced
    frequencies (the finest resolving ~(hi - lo) / 2 ** (n_freqs + 1)).
    """
    u = (jnp.clip(s, lo, hi) - (hi + lo) / 2) / ((hi - lo) / 2)
    angles = (2.0 ** jnp.arange(n_freqs)) * jnp.pi * u
    return jnp.concatenate([u[None], jnp.sin(angles), jnp.cos(angles)])


def rms(x: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    return jnp.sqrt(jnp.mean(x ** 2, axis=-1) + eps)


class ScaleAwareNorm(eqx.Module):
    """
    Self-conditioned AdaLN.  For one token x (dim,):

        r = rms(x)
        gamma, beta = to_mod([fourier(log r), cond])
        returns (x / r) * (1 + gamma) + beta

    `to_mod` is zero-initialized, so this starts out as a plain RMSNorm.
    """
    to_mod: eqx.nn.Linear
    n_freqs: int = eqx.field(static=True)
    lo: float = eqx.field(static=True)
    hi: float = eqx.field(static=True)

    def __init__(self, dim: int, cond_dim: int, key, n_freqs: int = 6, lo: float = -8., hi: float = 8.):
        """
        lo, hi: the range of log rms the embedding resolves (clipped outside)
        """
        self.n_freqs, self.lo, self.hi = n_freqs, lo, hi
        to_mod = eqx.nn.Linear(2 * n_freqs + 1 + cond_dim, 2 * dim, key=key)
        self.to_mod = eqx.tree_at(
            lambda m: (m.weight, m.bias), to_mod,
            (jnp.zeros_like(to_mod.weight), jnp.zeros_like(to_mod.bias)))

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        x: (dim,)
        cond: (cond_dim,), may be empty
        """
        r = rms(x)
        emb = fourier_embed(jnp.log(r), self.lo, self.hi, self.n_freqs)
        gamma, beta = jnp.split(self.to_mod(jnp.concatenate([emb, cond])), 2)
        return (x / r) * (1 + gamma) + beta


def rotary(x: jnp.ndarray, base: float = 10000.) -> jnp.ndarray:
    """
    Rotary position embedding.
    x: (n_tokens, n_heads, head_dim), head_dim even
    """
    n, _, d = x.shape
    freqs = base ** (-jnp.arange(0, d, 2) / d)
    angles = jnp.arange(n)[:, None] * freqs[None, :]          # (n, d / 2)
    cos, sin = jnp.cos(angles)[:, None, :], jnp.sin(angles)[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).reshape(x.shape)


class QKNormAttention(eqx.Module):
    """
    Multi-head self-attention with QK-norm: queries and keys are
    RMS-normalized per head (with a learned gain), so the attention logits
    are bounded by a learned per-head temperature, whatever the scale of the
    tokens.  Values are not normalized, so they can still carry magnitude.
    """
    qkv: eqx.nn.Linear
    out: eqx.nn.Linear
    q_gain: jnp.ndarray
    k_gain: jnp.ndarray
    log_temperature: jnp.ndarray
    level_bias: jnp.ndarray
    n_heads: int = eqx.field(static=True)

    def __init__(self, dim: int, n_heads: int, pair_dim: int, key):
        if dim % n_heads != 0 or (dim // n_heads) % 2 != 0:
            raise ValueError('dim must split into n_heads heads of even size')
        k1, k2 = jax.random.split(key)
        head_dim = dim // n_heads
        self.n_heads = n_heads
        self.qkv = eqx.nn.Linear(dim, 3 * dim, key=k1)
        self.out = eqx.nn.Linear(dim, dim, key=k2)
        self.q_gain = jnp.ones(head_dim)
        self.k_gain = jnp.ones(head_dim)
        # unit-RMS q and k have a dot product of up to head_dim, so start at
        # the usual 1 / sqrt(head_dim) scaling
        self.log_temperature = jnp.full((n_heads,), -0.5 * jnp.log(head_dim))
        # per-head attention bias from the pairwise level-difference
        # embedding; zero at init
        self.level_bias = jnp.zeros((n_heads, pair_dim))

    def __call__(self, x: jnp.ndarray, mask: jnp.ndarray, pair: jnp.ndarray) -> jnp.ndarray:
        """
        x: (n_tokens, dim)
        mask: (n_tokens, n_tokens), True where query i may attend to key j
        pair: (n_tokens, n_tokens, pair_dim), embedded level differences
        """
        n, dim = x.shape
        q, k, v = jnp.split(jax.vmap(self.qkv)(x), 3, axis=-1)
        q, k, v = (t.reshape(n, self.n_heads, -1) for t in (q, k, v))

        q = rotary(q / rms(q)[..., None] * self.q_gain)
        k = rotary(k / rms(k)[..., None] * self.k_gain)

        logits = jnp.einsum('ihd,jhd->hij', q, k) * jnp.exp(self.log_temperature)[:, None, None]
        logits = logits + jnp.einsum('ijf,hf->hij', pair, self.level_bias)
        logits = jnp.where(mask[None], logits, -jnp.inf)
        weights = jax.nn.softmax(logits, axis=-1)
        y = jnp.einsum('hij,jhd->ihd', weights, v).reshape(n, dim)
        return jax.vmap(self.out)(y)


class Block(eqx.Module):
    norm1: ScaleAwareNorm
    attention: QKNormAttention
    norm2: ScaleAwareNorm
    mlp: eqx.nn.MLP

    def __init__(self, dim: int, n_heads: int, cond_dim: int, pair_dim: int, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = ScaleAwareNorm(dim, cond_dim, k1)
        self.attention = QKNormAttention(dim, n_heads, pair_dim, k2)
        self.norm2 = ScaleAwareNorm(dim, cond_dim, k3)
        self.mlp = eqx.nn.MLP(dim, dim, 4 * dim, depth=1, activation=jax.nn.gelu, key=k4)

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, mask: jnp.ndarray, pair: jnp.ndarray) -> jnp.ndarray:
        """
        x: (n_tokens, dim), the residual stream
        cond: (n_tokens, cond_dim)
        pair: (n_tokens, n_tokens, pair_dim)
        """
        x = x + self.attention(jax.vmap(self.norm1)(x, cond), mask, pair)
        return x + jax.vmap(self.mlp)(jax.vmap(self.norm2)(x, cond))


class AttentionEncoder(eqx.Module):
    """
    Drop-in alternative to `jax_ae.Encoder`, but fed the *unnormalized*
    spectrogram (it normalizes per frame itself), and returning each frame's
    log level alongside its features.
    """
    in_proj: eqx.nn.Linear
    level_proj: eqx.nn.Linear
    blocks: list
    final_norm: ScaleAwareNorm
    n_level_freqs: int = eqx.field(static=True)
    anticausal: bool = eqx.field(static=True)
    level_mode: str = eqx.field(static=True)
    # the range of absolute log levels resolved: a frame of unit-RMS (ortho-
    # normalized) STFT magnitudes has log level 0; zero-padded silence sits
    # at the bottom
    level_lo: float = eqx.field(static=True)
    level_hi: float = eqx.field(static=True)

    def __init__(
            self,
            n_coeffs: int,
            channels: int,
            n_layers: int,
            n_heads: int,
            key,
            anticausal: bool = True,
            level_mode: str = 'relative',
            n_level_freqs: int = 6,
            level_lo: float = -14.,
            level_hi: float = 6.):
        if level_mode not in ('relative', 'absolute'):
            raise ValueError(f'unknown level_mode {level_mode}')
        keys = jax.random.split(key, n_layers + 3)
        self.n_level_freqs = n_level_freqs
        self.anticausal = anticausal
        self.level_mode = level_mode
        self.level_lo, self.level_hi = level_lo, level_hi
        embed_dim = 2 * n_level_freqs + 1
        cond_dim = embed_dim * (2 if level_mode == 'absolute' else 1)
        self.in_proj = eqx.nn.Linear(n_coeffs, channels, key=keys[0])
        self.level_proj = eqx.nn.Linear(cond_dim, channels, key=keys[1])
        self.blocks = [Block(channels, n_heads, cond_dim, embed_dim, k) for k in keys[2:-1]]
        self.final_norm = ScaleAwareNorm(channels, cond_dim, keys[-1])

    def _embed(self, s: jnp.ndarray, lo: float, hi: float) -> jnp.ndarray:
        """s: any shape; returns s.shape + (2 * n_level_freqs + 1,)"""
        flat = jax.vmap(lambda x: fourier_embed(x, lo, hi, self.n_level_freqs))(s.reshape(-1))
        return flat.reshape(s.shape + (-1,))

    def __call__(self, spec: jnp.ndarray):
        """
        spec: (n_frames, n_coeffs), unnormalized
        returns: features (channels, n_frames), and each frame's absolute
            log level (n_frames,), clipped to [level_lo, level_hi]
        """
        n = spec.shape[0]
        energy = jnp.mean(spec ** 2, axis=-1)                       # (n_frames,)

        if self.anticausal:
            # each frame sees itself and what follows, as in `jax_ae.Encoder`
            mask = jnp.triu(jnp.ones((n, n), dtype=bool))
            peak = jax.lax.cummax(energy, reverse=True)
            # the mean energy of each frame's visible context: it and what follows
            visible = jnp.cumsum(energy[::-1])[::-1] / jnp.arange(n, 0, -1)
        else:
            mask = jnp.ones((n, n), dtype=bool)
            peak = jnp.full((n,), jnp.max(energy))
            visible = jnp.full((n,), jnp.mean(energy))

        # floor each frame's energy RELATIVE_FLOOR below the loudest frame it
        # can see, rather than at an absolute value, so the floor scales with
        # gain and relative levels (silence included) are exactly
        # gain-invariant, and depend on nothing a frame can't see
        floor = peak * RELATIVE_FLOOR + 1e-30
        energy = energy + floor
        log_level = 0.5 * jnp.log(energy)
        log_visible = 0.5 * jnp.log(visible + floor)
        rel_lo = 0.5 * math.log(RELATIVE_FLOOR) - 1

        # each frame's level relative to its visible context (at most
        # log(n) / 2: the context includes the frame itself)
        cond = self._embed(log_level - log_visible, rel_lo, 0.5 * math.log(n))
        if self.level_mode == 'absolute':
            cond = jnp.concatenate([self._embed(log_level, self.level_lo, self.level_hi), cond], axis=-1)
        # pairwise level differences, key j minus query i
        pair = self._embed(log_level[None, :] - log_level[:, None], rel_lo, -rel_lo)

        x = jax.vmap(self.in_proj)(spec / jnp.sqrt(energy)[:, None]) + jax.vmap(self.level_proj)(cond)

        for block in self.blocks:
            x = block(x, cond, mask, pair)

        x = jax.vmap(self.final_norm)(x, cond)
        return x.T, jnp.clip(log_level, self.level_lo, self.level_hi)


# ---------------------------------------------------------------------------
# shared plumbing for models that offer `AttentionEncoder` as an alternative
# encoder (`claude_jax_physical.py`, `goo_physical_train.py`)

class AttentionConfig(NamedTuple):
    n_layers: int
    n_heads: int
    anticausal: bool
    level_mode: str
    # spectral magnitudes are raised to this power before the encoder sees
    # them: a power law (unlike log(x + eps) or log1p) scales with gain, so
    # the encoder's level handling stays exact, while still compressing
    # each frame's spectral tilt
    exponent: float


def add_attention_arguments(parser, default_encoder: str = 'conv'):
    parser.add_argument('--encoder', choices=['conv', 'attention'], default=default_encoder,
                        help="'attention': a transformer with scale-aware (self-conditioned AdaLN) norms "
                             "and QK-norm attention; see jax_scale_attention.py")
    parser.add_argument('--attention-layers', type=int, default=6, help="'attention' encoder only")
    parser.add_argument('--attention-heads', type=int, default=4,
                        help="'attention' encoder only; channels / heads must be even")
    parser.add_argument('--attention-mask', choices=['anticausal', 'full'], default='anticausal',
                        help="'attention' encoder only: 'anticausal' (each frame sees itself and what "
                             "follows) or 'full'")
    parser.add_argument('--attention-levels', choices=['relative', 'absolute'], default='relative',
                        help="'attention' encoder only: 'relative' keeps the encoder's features invariant "
                             "to gain (absolute level reaches only the heads, as an explicit log level); "
                             "'absolute' also embeds each frame's absolute level into the features")
    parser.add_argument('--attention-exponent', type=float, default=0.5,
                        help="'attention' encoder only: spectral magnitudes ** this, as its input")


def attention_config_from_args(args) -> Optional[AttentionConfig]:
    """None for `--encoder conv`"""
    if args.encoder != 'attention':
        return None
    return AttentionConfig(
        n_layers=args.attention_layers,
        n_heads=args.attention_heads,
        anticausal=args.attention_mask == 'anticausal',
        level_mode=args.attention_levels,
        exponent=args.attention_exponent)


def make_attention_encoder(n_coeffs: int, channels: int, config: AttentionConfig, key) -> AttentionEncoder:
    return AttentionEncoder(
        n_coeffs, channels, n_layers=config.n_layers, n_heads=config.n_heads, key=key,
        anticausal=config.anticausal, level_mode=config.level_mode)


def pooled_log_level(log_level: jnp.ndarray) -> jnp.ndarray:
    """(n_frames,) per-frame log levels -> () the log RMS over all frames"""
    return 0.5 * (jax.nn.logsumexp(2 * log_level) - math.log(log_level.shape[-1]))
