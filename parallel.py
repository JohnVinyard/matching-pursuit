import base64
import math
import struct
from builtins import enumerate
from dataclasses import dataclass
from typing import Iterable, Tuple, Callable, List

from conjure import LmdbCollection, SupportedContentType, NumpySerializer, NumpyDeserializer, IdentitySerializer, IdentityDeserializer, loggers, serve_conjure
import numpy as np
from torch import nn

import torch
from torch.nn import functional as F
from random import randint

import conjure
from conjure.logger import encode_audio
from torch.optim import Adam, Optimizer
from config.dotenv import Config
from data.audioiter import AudioIterator
from modules import sparsify, interpolate_last_axis, max_norm
from modules.activation import unit_sine
from modules.anticausal import AntiCausalAnalysis
from modules.decompose import fft_resample
from modules.multibanddict import flattened_multiband_spectrogram, multiband_spectrogram
from modules.normalization import unit_norm
from modules.reverb import NeuralReverb
from modules.transfer import fft_convolve
from modules.upsample import ensure_last_axis_length
from spiking import SpikingModel
from util import device, count_parameters
from util.overfit import overfit_model
from torch.nn.utils.clip_grad import clip_grad_value_, clip_grad_norm_
from torch.nn.utils.weight_norm import weight_norm
from argparse import ArgumentParser

from matplotlib import pyplot as plt

from util.weight_init import make_initializer
from sample_rate_dho import impulse_excitation, linear_recurrence, noise_excitation, sawtooth_excitation

Solution = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

MASS_COEFF = 1
BASE_RESONANCE = 0.02
FORWARD_FORCE = False
CHANNELS = 128

LEARNING_RATE = 1e-4

TENSION_EXPONENT_RANGE = (4.0, 9.0)

# if True, mix together all layers' outputs (scaled by LAYER_MIX, a learned
# per-layer weight) before reverb, instead of using only the last layer's
# output
SUM_LAYER_OUTPUTS = False

# if False (--no-reverb), the wet branch at the end of the chain is dropped:
# no room impulse response convolution, so the output is the (summed) DHO
# output times the dry/wet mix's *dry* weight (and final_gain), exactly as
# with reverb, minus the wet term.  Keeping the learned dry gain means the
# only difference is the rooms, not the output level.  Applies in every
# mode, including the synthetic targets of the self-supervised modes (whose
# parameter loss then also ignores the room terms, and the wet weight).  For
# checking how much the model leans on the rooms' own resonances rather than
# the DHO model.  The reverb modules and room heads are still built (and
# saved), just unused.
USE_REVERB = True

# if True, tension is a combination of fixed tension_dictionary atoms (see
# make_tension_dictionary/compute_tension_from_dictionary); if False, tension
# is produced directly as `compute_tension(raw_scalar)` (10 ** t), the way it
# worked before the dictionary was introduced
USE_TENSION_DICTIONARY = False

# if False, damp_mod and tension_mod are ignored in execute_parallel_layer
# (as if neither were ever passed in), regardless of what any caller
# generates for them. Since all three modes (train, overfit, overfit_direct)
# funnel through execute_parallel_layer, this one flag disables both for all
# of them. Does not affect tension_modifier, the layer-to-layer FM routing.
APPLY_DAMP_AND_TENSION_MOD = True

# InstrumentAutoencoder's self.tension/self.damp heads (ControlSignalCreator
# instances) produce raw, unscaled (post-relu, hard-sparsified) activations
# with no inherent bound -- unlike LayerController.materialize_tension_mod/
# materialize_damping_mod and random_synthetic_control_signals's known
# tension_mod/damp_mod, which both already scale down to 0.001 before use.
# Applying that same, already-established scale here means: (1) the
# estimated tension_mod/damp_mod self_supervised/self_supervised_only
# compare against known_tension_mod/known_damp_mod (via instrument_param_loss)
# land in the same regime those known values do, instead of being off by
# ~1000x; and (2) execute_parallel_layer's
# `tension = tension * compute_tension(log_modifier)` sees a small,
# physically-reasonable per-step nudge, rather than an unscaled, order-1
# log_modifier that compute_tension's live `10 ** x` would turn into a full
# order-of-magnitude (or worse) multiplicative shock each step.
TENSION_MOD_SCALE = 0.001
DAMP_MOD_SCALE = 0.001

# which analysis/encoder architecture InstrumentAutoencoder uses, one of:
#   'unet'        -- UNetAnalysis: the instrument latent comes from the
#                     U-Net's pooled bottleneck (replacing the separate
#                     latents/latents_attn attention-pooling heads), and
#                     cs/tension_mod/damp_mod are derived (via the existing
#                     ControlSignalCreator heads) from the U-Net's final,
#                     full-size decoder output.
#   'conv'        -- Analysis: a dilated AntiCausalAnalysis conv stack (no
#                     self-attention, despite the old flag name USE_UNET_
#                     ENCODER=False implying otherwise); the instrument
#                     latent is pooled from its output via the latents/
#                     latents_attn attention-pooling heads.
#   'transformer' -- TransformerAnalysis: a PyTorch port of
#                     jax_scale_attention.AttentionEncoder -- self-attention
#                     with no plain norm at all, only ScaleAwareNorm
#                     (self-conditioned AdaLN) and QK-norm attention (see
#                     that file's module docstring). Structured like 'conv':
#                     pooled via the same latents/latents_attn heads.
# 'conv' and 'transformer' are otherwise drop-in replacements for each
# other and for 'unet'.
ENCODER_ARCH = 'conv'

TRANSFORMER_N_LAYERS = 6
TRANSFORMER_N_HEADS = 4
# 'anticausal' (each frame sees itself and what follows, matching Analysis/
# UNetAnalysis's own causality convention) or False (full, unmasked attention)
TRANSFORMER_ANTICAUSAL = True
# 'relative' (absolute level enters nowhere but the returned log_level, so
# features are invariant to a global gain) or 'absolute' (also conditions on
# each frame's absolute level, giving up that invariance)
TRANSFORMER_LEVEL_MODE = 'relative'

# if True (and ENCODER_ARCH is 'conv' or 'transformer', so the latents_attn
# attention-pooling head is in use), unit_norm the attention-pooling weights
# (attn) over the time axis before pooling (latents = torch.mean(l * attn,
# dim=-1)) -- keeps the pooled instrument latent's scale independent of
# however large/small latents_attn's raw ControlSignalCreator output happens
# to be, rather than growing or shrinking with it. No effect when
# ENCODER_ARCH is 'unet', since that path has no attn/latents_attn at all.
USE_ATTENTION_NORM = True

N_NODES = 16

# analysis arch
DO_NORM = False


# sparse
SPARSIFY = True
SOFT_SPARSE = False
SPARSITY_LOSS_COEFF = 1
N_TO_KEEP = 128

# diagnostic-only, imperfect by design: when True, max_norm is applied to
# recon/synth/synth_recon (but not the real/synthetic *targets* they're
# compared against for the loss, apart from synthetic_audio itself, which
# also gets max_norm'd here since -- unlike the real AudioIterator batch,
# which already comes in max_norm'd -- it's never normalized at
# synthesis time) immediately before computing loss_func, so the
# reconstruction loss only ever sees peak-normalized signals on both sides.
# If this noticeably shrinks the loss (or stabilizes/improves training),
# that's a sign the model was mostly struggling to match overall loudness,
# rather than shape/structure.
MAX_NORM_BEFORE_LOSS = False

# weight on the auxiliary "information" loss (see information_loss/
# entropy_deficit), which pushes the target/recon spectrogram residual
# toward unstructured, uniform noise, regardless of its scale -- ported
# from jax_ae.py's --information-coeff. 0 disables it entirely.
INFORMATION_COEFF = 0

FILTER_SIZE = 256

# number of directly-defined/learned filter taps -- the actual filter
# (FILTER_SIZE long) is reached via FFT upsampling (see compute_filters),
# never learned directly at full size. Since FFT upsampling only ever
# populates the lowest N_FILTER_TAPS // 2 + 1 frequency bins and leaves the
# rest structurally at zero, it's impossible for the resulting filter to
# contain energy above that cutoff -- i.e. it is guaranteed low-pass,
# matching how low-pass response dominates passive physical systems -- while
# also cutting the number of parameters way down from FILTER_SIZE.
N_FILTER_TAPS = 64

# clamp the oscillator's angular frequency below Nyquist in
# damped_harmonic_oscillator (ported from jax_ae.py's own version of this
# function). Without it, tension (and so omega) is otherwise unbounded
# above, and an oscillator running above Nyquist doesn't just sound
# implausibly high-pitched -- it aliases, folding back down into the
# audible range, which is a worse failure mode than simply capping the
# frequency.
CLAMP_OMEGA_TO_NYQUIST = True

# the lowest frequency a node can ring at, in Hz at 22050 Hz (the rate the
# web demo and share links assume).  omega = sqrt(|tension - x ** 2|), with
# x = damping / (2 * mass), is exactly 0 wherever tension == x ** 2 -- which
# FM, sweeping tension every sample, crosses easily once a small mass makes
# x ** 2 ~ tension -- and there both the output's x / omega term and sqrt's
# gradient are infinite, so the forward pass and every gradient go NaN.
# Flooring omega here keeps both finite (x / omega <= x / floor) and, being
# a clamp, just zeroes omega's gradient below it.  20 Hz is below anything
# audible as pitch, so it shouldn't change how working nodes sound.
MIN_OMEGA_HZ = 20.
MIN_OMEGA_SAMPLERATE = 22050

# where the oscillators' excitation is integrated (--integrator):
#   'control_rate' -- `sequential` leaky-integrates the forces into an
#                     envelope at control rate, which is linearly
#                     interpolated to audio rate and multiplied by a
#                     sinusoid (damped_harmonic_oscillator).  Anything in
#                     the drive faster than the control rate is lost.
#   'sample_rate'  -- each node is a true resonator, run at audio rate as a
#                     parallel scan (see sample_rate_oscillator and
#                     sample_rate_dho.py): damping and tension (including
#                     the routed, audio-rate FM) apply per sample, and the
#                     drive is a learned mix of audio-rate excitations (see
#                     EXCITATIONS).
#                     Same parameters, same output form, so the two can be
#                     A/B'd directly.
INTEGRATOR = 'control_rate'

# 'sample_rate' only: the excitations the (routed) control-rate forces can
# drive each node with.  Which one -- or what blend -- is no longer a CLI
# choice: it's part of the instrument, learned like every other instrument
# parameter.  Both the blend and the bow frequency are ONE value for the
# whole instrument, shared by every node of every layer
# (InstrumentAutoencoder.excitation_mix / to_bow_tension, from the latent;
# LayerController.excitation_logits / bow_t_raw in overfit_direct), to
# match the web demo, which has a single, global excitation mode and a
# single bow (pulse) rate, and couldn't emulate per-node ones.
# instrument.excitation_mix holds the softmax weights over:
#   'impulse' -- a strike at the start of each control frame (struck/plucked)
#   'noise'   -- white noise under the same forces, linearly interpolated
#                (blown, the forces as breath pressure).  The resonator
#                band-limits it around each node's (modulated) frequency.
#   'bow'     -- a sawtooth (stick-slip) at the instrument's bow frequency
#                (instrument.bow_tension), under the same forces as bow
#                pressure (bowed/scraped).  Its harmonics excite whichever
#                nodes they land near.
# Noise and bow are scaled so one control frame of either carries the same
# energy as one strike of the same force (see sample_rate_oscillator), so
# the softmax weights trade off like-for-like.  With --integrator
# control_rate none of this is heard: that path only integrates an envelope.
# Share links don't carry either yet.  (The web demo's bow is a pulse
# train at pulseRate Hz, not a sawtooth.)
EXCITATIONS = ('impulse', 'noise', 'bow')

# if True, InstrumentAutoencoder starts each hypernetwork head at a plausible
# operating point (see InstrumentHyperNetwork.init_operating_point), by
# setting its bias, rather than at ~0.  At init the encoder's latent is
# tiny (rms ~1e-4), so every head outputs ~its bias, and init_func zeroes
# them all: gains, output gains, filter/noise/bow mixes, routers, wet/dry and
# mass all start at ~1e-5, the output at ~1e-19, and most gradients below
# Adam's eps (1e-8), where Adam's steps shrink to nothing -- layer 0's
# exactly zero.  (to_tension's bias was already set this way.)
INIT_OPERATING_POINT = True

# Every layer but the last is heard only as FM on the next layer's tension
# (unless SUM_LAYER_OUTPUTS), through tanh(x * gains) * output_gain and an
# identity tension_router.  At init the drive is small (the untrained
# encoder's forces peak ~0.005, Layer's directly-optimized ones ~0.01), so
# with gains ~1 that output -- the FM depth, in log10 units of tension --
# peaks at only ~0.02-0.08, and with --integrator sample_rate those layers'
# gradients are 1e3-1e5x smaller than the last layer's.  So, with that
# integrator, their gains start FM_LAYER_GAIN times higher, U(0.5, 1.5) *
# FM_LAYER_GAIN: at 15, peak FM is ~0.25 log10.  Not with control_rate:
# there FM moves omega * t directly, so the earlier layers' tension
# gradients are already larger than the last layer's, and the boost made a
# 40-step check train worse (51% of the starting loss vs 32%).  The tanh still bounds the output, so the most FM
# a layer can apply (~output_gain, i.e. ~1 log10) is unchanged.  1 turns it
# off.  (jax_ae.ParallelGenerator's equivalent fix sets its tension router
# and output gain instead: there those, not the drive, were the small terms.)
FM_LAYER_GAIN = 15.
# the same, for overfit_direct's Layers, whose initial drive is ~10x
# smaller still (FM peak ~0.002 at gains ~1): 150 puts it at ~0.27 log10
FM_LAYER_GAIN_DIRECT = 150.


def damping_nonlinearity(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)
    # return torch.clamp(x, 0, 1)


def compute_tension(t: torch.Tensor) -> torch.Tensor:
    return 10**t


@dataclass
class HyperParameters:
    n_samples: int
    n_frames: int
    base_resonance: float
    mass_coeff: float


@dataclass
class StaticTensors:
    """
    Tensors that are static and do not require any gradients throughout
    the entire experiment.
    """
    time: torch.Tensor
    damping: torch.Tensor
    initial_displacement: torch.Tensor


@dataclass
class PerformanceTensors:
    """
    Tensors that represent actions from something with agency, like a human
    performer
    """
    forces: torch.Tensor
    damp_mod: torch.Tensor
    tension_mod: torch.Tensor


@dataclass
class InstrumentDefinitionTensors:
    """
    Tensors representing the physical properties of a subset of a physical
    object
    """
    mass: torch.Tensor
    tension: torch.Tensor
    filters: torch.Tensor
    filters_mix: torch.Tensor
    force_router: torch.Tensor
    tension_router: torch.Tensor
    damping: torch.Tensor
    gains: torch.Tensor
    output_gain: torch.Tensor
    # see EXCITATIONS.  Both are one value for the whole instrument, shared
    # across layers, so they're set by whoever owns the whole instrument
    # (InstrumentAutoencoder, LayerController), not by each layer's
    # hypernetwork:
    # softmax weights over EXCITATIONS, (batch, 1, 3)
    excitation_mix: torch.Tensor = None
    # the bow frequency, as a real-space tension (omega ** 2, in the same
    # units as `tension`), (batch, 1, 1)
    bow_tension: torch.Tensor = None

    def display_shapes(self):
        print(f'''
            {self.mass.shape}
            {self.tension.shape}
            {self.filters.shape}
            {self.filters_mix.shape}
            {self.force_router.shape}
            {self.tension_router.shape}
            {self.damping.shape}
            {self.excitation_mix.shape}
            {self.bow_tension.shape}
            {self.output_gain.shape}
        ''')


class ParameterGenerator(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
            
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = nn.Parameter(torch.zeros(1).fill_(1))

        ln1 = weight_norm(nn.Linear(in_channels, out_channels))
        self.ln1 = ln1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x)
        return x * self.scale


class Analysis(nn.Module):
    def __init__(self, n_samples: int, frame_size: int, channels: int):
        super().__init__()
        self.n_samples = n_samples
        self.frame_size = frame_size
        self.window_size = self.frame_size * 4
        self.channels = channels
        self.n_coeffs = self.window_size // 2 + 1

        
        self.network = AntiCausalAnalysis(
            self.n_coeffs,
            channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 1],
            do_norm=DO_NORM,
            pos_encodings=False,
            with_activation_norm=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        spec = stft(x, ws=self.window_size, step=self.frame_size, pad=True).view(
            batch, -1, self.n_coeffs).permute(0, 2, 1)


        x = self.network(spec)
        return x


class UNetAnalysis(nn.Module):
    """
    Alternate encoder to Analysis (see ENCODER_ARCH): a 1D U-Net over
    the same STFT-magnitude representation Analysis consumes. Exposes two
    outputs instead of one: a pooled bottleneck (the instrument latent,
    replacing the separate latents/latents_attn attention-pooling heads)
    and a full-size decoder output (the same role Analysis's own output
    plays -- still run through the existing ControlSignalCreator heads for
    cs/tension_mod/damp_mod).
    """

    def __init__(self, n_samples: int, frame_size: int, channels: int, depth: int = 4):
        super().__init__()
        self.n_samples = n_samples
        self.frame_size = frame_size
        self.window_size = self.frame_size * 4
        self.channels = channels
        self.n_coeffs = self.window_size // 2 + 1
        self.depth = depth

        self.proj = weight_norm(nn.Conv1d(self.n_coeffs, channels, 1, 1, 0))

        self.down = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, 4, stride=2, padding=1))
            for _ in range(depth)
        ])
        self.upsample = nn.ModuleList([
            weight_norm(nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1))
            for _ in range(depth)
        ])
        self.up_proj = nn.ModuleList([
            weight_norm(nn.Conv1d(channels * 2, channels, 3, 1, 1))
            for _ in range(depth)
        ])

        # gated by DO_NORM (see forward) -- matches AntiCausalBlock's own
        # do_norm-gated BatchNorm1d convention, so the one flag controls
        # normalization in both encoders
        self.down_norms = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(depth)])
        self.up_norms = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(depth)])

        # the "final dense full-sized output layer" -- cs/tension_mod/damp_mod
        # are derived from this, via the existing ControlSignalCreator heads
        self.final = weight_norm(nn.Conv1d(channels, channels, 1, 1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = x.shape[0]
        spec = stft(x, ws=self.window_size, step=self.frame_size, pad=True).view(
            batch, -1, self.n_coeffs).permute(0, 2, 1)

        x = self.proj(spec)

        skips = []
        for down, norm in zip(self.down, self.down_norms):
            skips.append(x)
            x = F.silu(down(x))
            if DO_NORM:
                x = norm(x)

        # pooled bottleneck -> the instrument latent
        bottleneck = torch.mean(x, dim=-1)

        for upsample, up_proj, norm, skip in zip(self.upsample, self.up_proj, self.up_norms, reversed(skips)):
            x = F.silu(upsample(x))
            # strided conv/deconv lengths don't always divide n_frames
            # exactly; align defensively rather than assume exact arithmetic
            x = ensure_last_axis_length(x, desired_size=skip.shape[-1])
            x = torch.cat([x, skip], dim=1)
            x = F.silu(up_proj(x))
            if DO_NORM:
                x = norm(x)

        x = self.final(x)

        return bottleneck, x


# --- TransformerAnalysis: PyTorch port of jax_scale_attention.py --------
#
# See that file's module docstring for the full rationale. In short: a
# self-attention encoder for real-valued, wide-dynamic-range input, whose
# only normalization is ScaleAwareNorm -- a self-conditioned AdaLN where
# each token is RMS-normalized and the log of the RMS it divided out is
# Fourier-embedded and used (with an optional conditioning vector) to
# predict the normalized token's scale/shift -- and QK-norm attention,
# where queries/keys are RMS-normalized per head (with a learned gain), so
# attention logits stay bounded regardless of the stream's scale.

# frames are floored this far (in energy; -80 dB) below the loudest frame
# they can see; see TransformerAnalysis.forward
RELATIVE_FLOOR = 1e-8


def fourier_embed(s: torch.Tensor, lo: float, hi: float, n_freqs: int) -> torch.Tensor:
    """
    Embed `s` (any shape), clipped to [lo, hi] and rescaled to [-1, 1], as
    (..., 2 * n_freqs + 1): itself, then sines and cosines at octave-spaced
    frequencies (the finest resolving ~(hi - lo) / 2 ** (n_freqs + 1)).
    """
    u = (torch.clamp(s, lo, hi) - (hi + lo) / 2) / ((hi - lo) / 2)
    freqs = (2.0 ** torch.arange(n_freqs, device=s.device, dtype=s.dtype)) * math.pi
    angles = u[..., None] * freqs
    return torch.cat([u[..., None], torch.sin(angles), torch.cos(angles)], dim=-1)


def rms(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.sqrt(torch.mean(x ** 2, dim=-1) + eps)


def rotary(x: torch.Tensor, base: float = 10000.) -> torch.Tensor:
    """
    Rotary position embedding, over the second-to-last axis (the token/time
    axis). x: (..., n_tokens, n_heads, head_dim), head_dim even.
    """
    n, h, d = x.shape[-3:]
    freqs = base ** (-torch.arange(0, d, 2, device=x.device, dtype=x.dtype) / d)
    angles = torch.arange(n, device=x.device, dtype=x.dtype)[:, None] * freqs[None, :]
    cos, sin = torch.cos(angles)[:, None, :], torch.sin(angles)[:, None, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).reshape(x.shape)


class ScaleAwareNorm(nn.Module):
    """
    Self-conditioned AdaLN. For one token x (..., dim):

        r = rms(x)
        gamma, beta = to_mod([fourier(log r), cond])
        returns (x / r) * (1 + gamma) + beta

    `to_mod` is zero-initialized, so this starts out as a plain RMSNorm.
    """

    def __init__(self, dim: int, cond_dim: int, n_freqs: int = 6, lo: float = -8., hi: float = 8.):
        """lo, hi: the range of log rms the embedding resolves (clipped outside)"""
        super().__init__()
        self.n_freqs, self.lo, self.hi = n_freqs, lo, hi
        self.to_mod = nn.Linear(2 * n_freqs + 1 + cond_dim, 2 * dim)
        nn.init.zeros_(self.to_mod.weight)
        nn.init.zeros_(self.to_mod.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (..., dim); cond: (..., cond_dim), may be empty on the last axis"""
        r = rms(x)
        emb = fourier_embed(torch.log(r), self.lo, self.hi, self.n_freqs)
        gamma, beta = self.to_mod(torch.cat([emb, cond], dim=-1)).chunk(2, dim=-1)
        return (x / r[..., None]) * (1 + gamma) + beta


class QKNormAttention(nn.Module):
    """
    Multi-head self-attention with QK-norm: queries and keys are
    RMS-normalized per head (with a learned gain), so the attention logits
    are bounded by a learned per-head temperature, whatever the scale of
    the tokens. Values are not normalized, so they can still carry
    magnitude.
    """

    def __init__(self, dim: int, n_heads: int, pair_dim: int):
        super().__init__()
        if dim % n_heads != 0 or (dim // n_heads) % 2 != 0:
            raise ValueError('dim must split into n_heads heads of even size')
        head_dim = dim // n_heads
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.q_gain = nn.Parameter(torch.ones(head_dim))
        self.k_gain = nn.Parameter(torch.ones(head_dim))
        # unit-RMS q and k have a dot product of up to head_dim, so start at
        # the usual 1 / sqrt(head_dim) scaling
        self.log_temperature = nn.Parameter(torch.full((n_heads,), -0.5 * math.log(head_dim)))
        # per-head attention bias from the pairwise level-difference
        # embedding; zero at init
        self.level_bias = nn.Parameter(torch.zeros(n_heads, pair_dim))

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """
        x: (b, n_tokens, dim)
        mask: (n_tokens, n_tokens), True where query i may attend to key j
        pair: (b, n_tokens, n_tokens, pair_dim), embedded level differences
        """
        b, n, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = (t.view(b, n, self.n_heads, -1) for t in (q, k, v))

        q = rotary(q / rms(q)[..., None] * self.q_gain)
        k = rotary(k / rms(k)[..., None] * self.k_gain)

        logits = torch.einsum('bihd,bjhd->bhij', q, k) * torch.exp(self.log_temperature)[None, :, None, None]
        logits = logits + torch.einsum('bijf,hf->bhij', pair, self.level_bias)
        logits = logits.masked_fill(~mask[None, None], float('-inf'))
        weights = torch.softmax(logits, dim=-1)
        y = torch.einsum('bhij,bjhd->bihd', weights, v).reshape(b, n, dim)
        return self.out(y)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, cond_dim: int, pair_dim: int):
        super().__init__()
        self.norm1 = ScaleAwareNorm(dim, cond_dim)
        self.attention = QKNormAttention(dim, n_heads, pair_dim)
        self.norm2 = ScaleAwareNorm(dim, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(
            self, x: torch.Tensor, cond: torch.Tensor,
            mask: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        """x: (b, n_tokens, dim), the residual stream; cond: (b, n_tokens, cond_dim)"""
        x = x + self.attention(self.norm1(x, cond), mask, pair)
        return x + self.mlp(self.norm2(x, cond))


class TransformerAnalysis(nn.Module):
    """
    Alternate encoder to Analysis/UNetAnalysis (see ENCODER_ARCH): a PyTorch
    port of jax_scale_attention.AttentionEncoder. Structured like Analysis
    (not UNetAnalysis): returns a per-frame feature map, pooled into the
    instrument latent by the existing latents/latents_attn attention-pooling
    heads (see InstrumentAutoencoder), rather than exposing its own pooled
    bottleneck. Also returns each frame's absolute log level (clipped to
    [level_lo, level_hi]), unused by InstrumentAutoencoder today but kept
    for parity with the jax original and potential future use (e.g. a
    magnitude/loudness head that needs the level the per-frame RMS
    normalization below removes).
    """

    def __init__(
            self, n_samples: int, frame_size: int, channels: int,
            n_layers: int = 6, n_heads: int = 4, anticausal: bool = True,
            level_mode: str = 'relative', n_level_freqs: int = 6,
            level_lo: float = -14., level_hi: float = 6.):
        super().__init__()
        if level_mode not in ('relative', 'absolute'):
            raise ValueError(f'unknown level_mode {level_mode}')
        self.n_samples = n_samples
        self.frame_size = frame_size
        self.window_size = frame_size * 4
        self.channels = channels
        self.n_coeffs = self.window_size // 2 + 1
        self.anticausal = anticausal
        self.level_mode = level_mode
        self.n_level_freqs = n_level_freqs
        self.level_lo, self.level_hi = level_lo, level_hi

        embed_dim = 2 * n_level_freqs + 1
        cond_dim = embed_dim * (2 if level_mode == 'absolute' else 1)
        self.in_proj = nn.Linear(self.n_coeffs, channels)
        self.level_proj = nn.Linear(cond_dim, channels)
        self.blocks = nn.ModuleList([
            TransformerBlock(channels, n_heads, cond_dim, embed_dim)
            for _ in range(n_layers)
        ])
        self.final_norm = ScaleAwareNorm(channels, cond_dim)

    def _embed(self, s: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return fourier_embed(s, lo, hi, self.n_level_freqs)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (batch, 1, n_samples), unnormalized
        returns: features (batch, channels, n_frames), and each frame's
            absolute log level (batch, n_frames), clipped to [level_lo, level_hi]
        """
        batch = x.shape[0]
        spec = stft(x, ws=self.window_size, step=self.frame_size, pad=True).view(
            batch, -1, self.n_coeffs)
        n = spec.shape[1]
        energy = torch.mean(spec ** 2, dim=-1)

        if self.anticausal:
            # each frame sees itself and what follows, as in Analysis/UNetAnalysis
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device))
            peak, _ = torch.cummax(energy.flip(-1), dim=-1)
            peak = peak.flip(-1)
            # the mean energy of each frame's visible context: it and what follows
            cum = torch.cumsum(energy.flip(-1), dim=-1).flip(-1)
            counts = torch.arange(n, 0, -1, device=x.device, dtype=energy.dtype)
            visible = cum / counts
        else:
            mask = torch.ones(n, n, dtype=torch.bool, device=x.device)
            peak = energy.max(dim=-1, keepdim=True).values.expand(-1, n)
            visible = energy.mean(dim=-1, keepdim=True).expand(-1, n)

        # floor each frame's energy RELATIVE_FLOOR below the loudest frame it
        # can see, rather than at an absolute value, so the floor scales with
        # gain and relative levels (silence included) are exactly
        # gain-invariant, and depend on nothing a frame can't see
        floor = peak * RELATIVE_FLOOR + 1e-30
        energy = energy + floor
        log_level = 0.5 * torch.log(energy)
        log_visible = 0.5 * torch.log(visible + floor)
        rel_lo = 0.5 * math.log(RELATIVE_FLOOR) - 1

        # each frame's level relative to its visible context (at most
        # log(n) / 2: the context includes the frame itself)
        cond = self._embed(log_level - log_visible, rel_lo, 0.5 * math.log(n))
        if self.level_mode == 'absolute':
            cond = torch.cat([self._embed(log_level, self.level_lo, self.level_hi), cond], dim=-1)
        # pairwise level differences, key j minus query i
        pair = self._embed(log_level[:, None, :] - log_level[:, :, None], rel_lo, -rel_lo)

        h = self.in_proj(spec / torch.sqrt(energy)[..., None]) + self.level_proj(cond)

        for block in self.blocks:
            h = block(h, cond, mask, pair)

        h = self.final_norm(h, cond)
        return h.permute(0, 2, 1), torch.clamp(log_level, self.level_lo, self.level_hi)


def make_tension_dictionary(
        size: int,
        exponent_range: Tuple[float, float],
        seed: int) -> torch.Tensor:
    """
    Fixed, seeded dictionary of candidate (already real-space, i.e.
    already-exponentiated) tension values -- see the comment in
    InstrumentHyperNetwork.__init__ for the full rationale.
    """
    gen = torch.Generator().manual_seed(seed)
    exponents = torch.empty(size).uniform_(*exponent_range, generator=gen)
    return 10 ** exponents


def compute_mass(m_raw: torch.Tensor) -> torch.Tensor:  
    """
    Shared by InstrumentHyperNetwork (hypernetwork-generated m_raw) and Layer
    (directly-optimized m_raw), so the two stay in sync as this formula keeps
    getting tuned, rather than silently drifting apart.
    """
    return 1e-5 + torch.abs(m_raw)


def compute_tension_from_dictionary(
        t_logits: torch.Tensor,
        tension_dictionary: torch.Tensor) -> torch.Tensor:
    """
    Combines fixed dictionary atoms (see make_tension_dictionary) using
    per-node weights, instead of pushing a single scalar through a live
    `10 ** t`. Shared by InstrumentHyperNetwork and Layer; see
    USE_TENSION_DICTIONARY for the alternative, direct-tension path (just
    `compute_tension` applied to a single raw per-node parameter).
    """
    t_weights = torch.softmax(t_logits, dim=-1)
    return torch.einsum('bnk,k->bn', t_weights, tension_dictionary).unsqueeze(-1)


def compute_filters(filter_taps: torch.Tensor, filter_size: int) -> torch.Tensor:
    """
    Upsamples a small number of directly-defined/learned filter taps
    (N_FILTER_TAPS long) to the full filter_size via FFT-domain zero-padding
    (fft_resample, is_lowest_band=True), instead of learning filter_size
    taps directly. The frequency bins above the taps' own Nyquist are never
    populated -- not just zero-initialized, structurally zero -- so the
    resulting filter is guaranteed low-pass regardless of how training moves
    the taps. Shared by InstrumentHyperNetwork, Layer, and
    random_synthetic_instrument, so all three stay in sync.
    """
    return fft_resample(filter_taps, desired_size=filter_size, is_lowest_band=True)


class InstrumentHyperNetwork(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            n_nodes: int,
            filter_size: int,
            n_filter_taps: int = N_FILTER_TAPS,
            tension_dict_size: int = 1024,
            tension_exponent_range: Tuple[float, float] = TENSION_EXPONENT_RANGE,
            tension_dict_seed: int = 0):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.router_size = self.n_nodes ** 2
        self.filter_size = filter_size
        self.n_filter_taps = n_filter_taps
        self.tension_dict_size = tension_dict_size

        # see make_tension_dictionary/compute_tension_from_dictionary above:
        # a live exponential (`10 ** t`) has a self-referential gradient
        # (d(10^t)/dt = ln(10) * 10^t -- the gradient magnitude *is* the
        # current output, so it grows as t grows, causing runaway parameter
        # updates); combining fixed dictionary atoms instead replaces that
        # with a small number of *fixed*, per-atom gradients (d(tension)/dw_k
        # = D_k, constant across training), trading a dynamic, self-
        # amplifying instability for the ordinary kind of per-parameter scale
        # heterogeneity that Adam already handles well. USE_TENSION_DICTIONARY
        # switches between this and the simpler, direct `10 ** t` path.
        if USE_TENSION_DICTIONARY:
            self.register_buffer(
                'tension_dictionary',
                make_tension_dictionary(tension_dict_size, tension_exponent_range, tension_dict_seed))
            self.to_tension = ParameterGenerator(latent_dim, n_nodes * tension_dict_size)
        else:
            self.to_tension = ParameterGenerator(latent_dim, n_nodes)

        self.to_masses = ParameterGenerator(latent_dim, n_nodes)
        self.damping = ParameterGenerator(latent_dim, n_nodes)

        self.gain = ParameterGenerator(latent_dim, n_nodes)
        # a single scalar gain (per batch item, not per node), applied as
        # the final multiplicative stage before a layer's output
        self.output_gain = ParameterGenerator(latent_dim, 1)

        self.filters = ParameterGenerator(latent_dim, n_nodes * n_filter_taps)
        self.filters_mix = ParameterGenerator(latent_dim, n_nodes * 2)

        self.force_router = ParameterGenerator(latent_dim, self.router_size)
        self.tension_router = ParameterGenerator(latent_dim, self.router_size)

        # the excitation blend and bow frequency are per instrument, not per
        # layer, so they live on InstrumentAutoencoder (see EXCITATIONS)

    @torch.no_grad()
    def init_operating_point(self):
        """
        Start each head at the same kind of instrument `Layer`'s directly-
        optimized parameters and random_synthetic_instrument start from,
        via its bias (at init, each head's output is ~its bias; see
        INIT_OPERATING_POINT):
          - mass: |m_raw| ~ U(0, 1), as Layer, so 1 / (2 * mass) no longer
            dwarfs sqrt(tension) and the drive isn't divided by ~1e-5
          - gains, output_gain: U(0.5, 1.5), as Layer and synthetic
          - force_router, tension_router: identity, as Layer and synthetic
          - filters_mix: (dry 1, filtered 0) -- the one departure: Layer and
            synthetic use U(-0.01, 0.01), which would scale every layer's
            output by ~1e-2 again
        to_tension is set by InstrumentAutoencoder, as before.
        """
        n = self.n_nodes
        eye = torch.eye(n, device=self.gain.ln1.bias.device).flatten()
        self.to_masses.ln1.bias.uniform_(-1, 1)
        self.gain.ln1.bias.uniform_(0.5, 1.5)
        self.output_gain.ln1.bias.uniform_(0.5, 1.5)
        self.force_router.ln1.bias.copy_(eye)
        self.tension_router.ln1.bias.copy_(eye)
        self.filters_mix.ln1.bias.view(n, 2).copy_(
            torch.tensor([1., 0.], device=eye.device).expand(n, 2))

    def forward(self, latent: torch.Tensor) -> InstrumentDefinitionTensors:
        batch, latent_dim = latent.shape

        g = self.gain.forward(latent)
        og = self.output_gain.forward(latent)

        if USE_TENSION_DICTIONARY:
            t_logits = self.to_tension(latent).view(batch, self.n_nodes, self.tension_dict_size)
            t = compute_tension_from_dictionary(t_logits, self.tension_dictionary)
        else:
            t_raw = self.to_tension(latent).view(batch, self.n_nodes, 1)
            t = compute_tension(t_raw)

        m_raw = self.to_masses(latent).view(batch, self.n_nodes, 1)
        mass = compute_mass(m_raw)

        d = self.damping(latent).view(batch, self.n_nodes, 1)

        filt_taps = self.filters(latent).view(batch, self.n_nodes, self.n_filter_taps)
        filt = compute_filters(filt_taps, self.filter_size)
        mx = self.filters_mix(latent).view(batch, self.n_nodes, 2)

        fr = self.force_router(latent).view(batch, self.n_nodes, self.n_nodes)
        tr = self.tension_router(latent).view(
            batch, self.n_nodes, self.n_nodes)


        return InstrumentDefinitionTensors(
            mass=mass,
            tension=t,
            damping=d,
            filters=filt,
            filters_mix=mx,
            force_router=fr,
            tension_router=tr,
            gains=g,
            output_gain=og,
        )


class ControlSignalCreator(nn.Module):

    def __init__(self, in_channels: int, control_channels: int, n_to_keep: int, sparsify: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.control_channels = control_channels
        self.n_to_keep = n_to_keep

        network = weight_norm(nn.Conv1d(
            in_channels, control_channels, kernel_size=8, stride=1, padding=0))
        self.network = network 
        self.sparsify = sparsify

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 7))
        x = self.network(x)

        # x = torch.abs(x)
        # x = x - x.mean(dim=(1, 2), keepdim=True)
        # x = torch.relu(x)
        
        
        if self.sparsify:
            x = sparsify(x, n_to_keep=self.n_to_keep, soft=SOFT_SPARSE)
        return x


init_func = make_initializer(0.04)


class InstrumentAutoencoder(nn.Module):

    def __init__(
            self,
            n_samples: int,
            n_nodes: int,
            control_rate: int,
            n_layers: int,
            channels: int,
            filter_size: int,
            n_to_keep: int,
            n_filter_taps: int = N_FILTER_TAPS,
            base_resonance: float = 0.5,
            tension_dict_size: int = 1024,
            tension_exponent_range: Tuple[float, float] = TENSION_EXPONENT_RANGE,
            tension_dict_seed: int = 0):

        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.n_layers = n_layers
        self.channels = channels
        self.n_nodes = n_nodes
        self.filter_size = filter_size
        self.n_filter_taps = n_filter_taps
        self.n_to_keep = n_to_keep

        self.hyper_params = HyperParameters(
            n_samples=n_samples,
            n_frames=self.n_frames,
            base_resonance=base_resonance,
            mass_coeff=MASS_COEFF
        )

        d = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('d', d)

        _id = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('_id', _id)

        t = torch.linspace(0, 10, self.n_samples)
        self.register_buffer('t', t)

        influence_decay = torch.linspace(1, 0, self.n_frames) ** 2
        self.influence_decay = nn.Parameter(influence_decay)

        if ENCODER_ARCH == 'unet':
            self.analysis = UNetAnalysis(n_samples, control_rate, channels)
        elif ENCODER_ARCH == 'transformer':
            self.analysis = TransformerAnalysis(
                n_samples, control_rate, channels,
                n_layers=TRANSFORMER_N_LAYERS,
                n_heads=TRANSFORMER_N_HEADS,
                anticausal=TRANSFORMER_ANTICAUSAL,
                level_mode=TRANSFORMER_LEVEL_MODE)
        else:
            self.analysis = Analysis(n_samples, control_rate, channels)

        # self.to_inp_weights = ParameterGenerator(channels, channels**2)
        # self.to_hidden_weights = ParameterGenerator(channels, channels**2)
        # self.to_output_weights = ParameterGenerator(channels, channels**2)

        self.hyper_networks = nn.ModuleList([
            InstrumentHyperNetwork(
                channels, n_nodes, filter_size,
                n_filter_taps=n_filter_taps,
                tension_dict_size=tension_dict_size,
                tension_exponent_range=tension_exponent_range,
                # distinct seed per layer, so each layer's dictionary is an
                # independent draw rather than every layer sharing identical
                # atoms
                tension_dict_seed=tension_dict_seed + i)
            for i in range(n_layers)
        ])

        if ENCODER_ARCH != 'unet':
            # with UNetAnalysis, the pooled bottleneck is the instrument
            # latent directly -- no separate attention-pooling heads needed.
            # 'conv' and 'transformer' both instead return a per-frame
            # feature map, pooled here.
            self.latents = ControlSignalCreator(
                channels, channels, n_to_keep=n_to_keep, sparsify=False)

            self.latents_attn = ControlSignalCreator(
                channels, channels, n_to_keep=n_to_keep, sparsify=False)

        self.control = ControlSignalCreator(
            channels, n_nodes, n_to_keep=n_to_keep, sparsify=SPARSIFY)
        self.tension = ControlSignalCreator(
            channels, n_nodes, n_to_keep=16, sparsify=True)
        self.damp = ControlSignalCreator(
            channels, n_nodes, n_to_keep=16, sparsify=True)

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate=22050, n_samples=n_samples)

        self.room_mix = ParameterGenerator(channels, self.verb.n_rooms)
        self.wet_dry = ParameterGenerator(channels, 2)

        # one excitation blend and one bow frequency (a direct `10 ** t`
        # tension) for the whole instrument (see EXCITATIONS)
        self.excitation_mix = ParameterGenerator(channels, len(EXCITATIONS))
        self.to_bow_tension = ParameterGenerator(channels, 1)

        # used only when SUM_LAYER_OUTPUTS is True, to scale/mix each layer's
        # own output into the final signal, instead of using only the last
        # layer's output
        self.layer_mix = nn.Parameter(torch.zeros(n_layers).uniform_(-0.01, 0.01))

        # a single scalar gain, shared across every hyper_networks layer
        # instance (unlike each InstrumentDefinitionTensors.output_gain,
        # applied node-wise inside execute_parallel_layer before the layers
        # are mixed/summed), applied once as the very last step before the
        # final audio is returned
        self.final_gain = nn.Parameter(torch.zeros(1).uniform_(0.5, 1.5))

        self.apply(init_func)

        if not USE_TENSION_DICTIONARY:
            # init_func (above) just zeroed every head's bias, including
            # to_tension's -- re-spread each node's starting tension across
            # the full physically-plausible range (matching Layer's direct
            # t_raw init and random_synthetic_instrument's known-tension
            # generation) instead of leaving every node at bias=0, i.e.
            # tension=compute_tension(0)=1, many orders of magnitude below
            # where the self-supervised task needs it to end up. Note this
            # sets the *bias* of to_tension's underlying linear layer, so
            # the actual initial tension per node is this plus a
            # latent-dependent term (weight @ latent); it isn't a hard
            # guarantee the way the dictionary path's softmax combination is
            for hyper in self.hyper_networks:
                with torch.no_grad():
                    hyper.to_tension.ln1.bias.uniform_(*tension_exponent_range)

        # likewise, start the instrument's bow frequency somewhere in the
        # same range as node tensions, so the bow's harmonics can land on
        # nodes from the start (see EXCITATIONS)
        with torch.no_grad():
            self.to_bow_tension.ln1.bias.uniform_(*tension_exponent_range)

        if INIT_OPERATING_POINT:
            for hyper in self.hyper_networks:
                hyper.init_operating_point()
            with torch.no_grad():
                if INTEGRATOR == 'sample_rate' and not SUM_LAYER_OUTPUTS:
                    for hyper in self.hyper_networks[:-1]:
                        hyper.gain.ln1.bias.uniform_(0.5 * FM_LAYER_GAIN, 1.5 * FM_LAYER_GAIN)
                # rooms, as synthetic: U(0, 1) (through a relu); the dry
                # path fully on, and some reverb
                self.room_mix.ln1.bias.uniform_(0, 1)
                self.wet_dry.ln1.bias.copy_(torch.tensor([1., 0.5]))
                # an even blend of EXCITATIONS
                self.excitation_mix.ln1.bias.zero_()

    def compute_excitation(self, latents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        (batch, latent_dim) -> the instrument-wide excitation (see
        EXCITATIONS): softmax weights, (batch, 1, len(EXCITATIONS)), and bow
        tension, (batch, 1, 1)
        """
        mix = torch.softmax(self.excitation_mix(latents), dim=-1)[:, None, :]
        bow_tension = compute_tension(self.to_bow_tension(latents))[:, None, :]
        return mix, bow_tension

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if ENCODER_ARCH == 'unet':
            latents, x = self.analysis(x)
        else:
            if ENCODER_ARCH == 'transformer':
                x, _log_level = self.analysis(x)
            else:
                x = self.analysis(x)

            l = self.latents(x)
            attn = self.latents_attn(x)
            if USE_ATTENTION_NORM:
                attn = unit_norm(attn, dim=-1)

            # for now, just choose the very first latent.  This could
            # be an average
            latents = torch.mean(l * attn, dim=-1)
            # latents = l[:, :, 0]

        cs = self.control(x)

        # print(l.shape, latents.shape, cs.shape)

        # iw = self.to_hidden_weights(latents).view(-1, self.channels, self.channels)
        # hw = self.to_hidden_weights(latents).view(-1, self.channels, self.channels)
        # ow = self.to_hidden_weights(latents).view(-1, self.channels, self.channels)

        # tm = execute_rnn(hw, iw, ow, cs)
        # mixed = tm

        # print(tm.shape)

        t = self.tension(x) * TENSION_MOD_SCALE
        d = self.damp(x) * DAMP_MOD_SCALE

        tm = None
        layer_outputs = []

        static = StaticTensors(
            time=self.t,
            damping=self.d,
            initial_displacement=self._id
        )

        excitation_mix, bow_tension = self.compute_excitation(latents)

        for layer in self.hyper_networks:

            params: InstrumentDefinitionTensors = layer.forward(latents)
            params.excitation_mix = excitation_mix
            params.bow_tension = bow_tension
            diff = 1 - self.hyper_params.base_resonance
            params.damping = self.hyper_params.base_resonance + (damping_nonlinearity(params.damping) * diff * 0.9998)
            # params.mass = 1e-5 + F.softplus(params.mass)
            # params.mass = torch.abs(params.mass)

            if FORWARD_FORCE:
                tm, cs = execute_parallel_layer(
                    instrument=params,
                    hyper=self.hyper_params,
                    static=static,
                    forces=cs,
                    damp_mod=d,
                    tension_mod=t,
                    tension_modifier=tm
                )
            else:
                tm, _cs = execute_parallel_layer(
                    instrument=params,
                    hyper=self.hyper_params,
                    static=static,
                    forces=cs,
                    damp_mod=d,
                    tension_mod=t,
                    tension_modifier=tm
                )

            # tm still chains into the next layer's tension_modifier either
            # way -- this only affects what ends up in the final mix below
            layer_outputs.append(tm)

        if SUM_LAYER_OUTPUTS:
            stacked = torch.stack(layer_outputs, dim=-1)
            tm = torch.einsum('bctl,l->bct', stacked, self.layer_mix)
        else:
            tm = layer_outputs[-1]

        tm = torch.sum(tm, dim=1, keepdim=True)

        if USE_REVERB:
            rooms = torch.relu(self.room_mix.forward(latents))
            mx = self.wet_dry.forward(latents)

            wet = self.verb.forward(tm, rooms)

            stacked = torch.stack([tm, wet], dim=-1)

            mixed = torch.einsum('bctm,bm->bct', stacked, mx)
        else:
            mixed = tm * self.wet_dry.forward(latents)[:, 0, None, None]

        mixed = mixed * self.final_gain

        return cs, mixed

    def estimate_params(
            self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[InstrumentDefinitionTensors],
               torch.Tensor, torch.Tensor]:
        """
        Mirrors the encoding portion of forward() (analysis -> latents ->
        cs/tension_mod/damp_mod -> per-layer hypernetwork params -> room_mix/
        wet_dry), but stops before execute_parallel_layer/reverb/mixing and
        returns the raw estimated pieces instead. Used by self_supervised to
        compare estimated vs. known parameters for synthetic examples -- not
        called by forward(), and does not change forward()'s behavior.
        """
        if ENCODER_ARCH == 'unet':
            latents, x = self.analysis(x)
        else:
            if ENCODER_ARCH == 'transformer':
                x, _log_level = self.analysis(x)
            else:
                x = self.analysis(x)
            l = self.latents(x)
            attn = self.latents_attn(x)
            if USE_ATTENTION_NORM:
                attn = unit_norm(attn, dim=-1)
            latents = torch.mean(l * attn, dim=-1)

        cs = self.control(x)
        t = self.tension(x) * TENSION_MOD_SCALE
        d = self.damp(x) * DAMP_MOD_SCALE

        excitation_mix, bow_tension = self.compute_excitation(latents)
        params_list = []
        for layer in self.hyper_networks:
            params: InstrumentDefinitionTensors = layer.forward(latents)
            params.excitation_mix = excitation_mix
            params.bow_tension = bow_tension
            diff = 1 - self.hyper_params.base_resonance
            params.damping = self.hyper_params.base_resonance + (torch.sigmoid(params.damping) * diff * 0.9998)
            params_list.append(params)

        rooms = torch.relu(self.room_mix.forward(latents))
        mx = self.wet_dry.forward(latents)

        return cs, t, d, params_list, rooms, mx

    def random_forward(self, x: torch.Tensor, n_to_keep: int = None) -> torch.Tensor:
        """
        Mirrors LayerController.random_forward: holds the current (estimated
        from x) instrument definition/tension/damping/rooms/wet-dry mix
        fixed, but drives it with a fresh, highly-sparse random control
        signal instead of the encoder-derived one -- a sanity check on
        whether the learned instrument sounds plausible independent of the
        specific input it was estimated from.
        """
        cs, t, d, params_list, rooms, mx = self.estimate_params(x)

        forces = torch.zeros_like(cs).uniform_(0, 1)
        forces = sparsify(forces, n_to_keep=n_to_keep)

        static = StaticTensors(
            time=self.t,
            damping=self.d,
            initial_displacement=self._id
        )

        tm = None
        layer_outputs = []
        for params in params_list:
            tm, _ = execute_parallel_layer(
                instrument=params,
                hyper=self.hyper_params,
                static=static,
                forces=forces,
                damp_mod=d,
                tension_mod=t,
                tension_modifier=tm
            )
            layer_outputs.append(tm)

        if SUM_LAYER_OUTPUTS:
            stacked = torch.stack(layer_outputs, dim=-1)
            tm = torch.einsum('bctl,l->bct', stacked, self.layer_mix)
        else:
            tm = layer_outputs[-1]

        tm = torch.sum(tm, dim=1, keepdim=True)

        if USE_REVERB:
            wet = self.verb.forward(tm, rooms)
            stacked = torch.stack([tm, wet], dim=-1)
            mixed = torch.einsum('bctm,bm->bct', stacked, mx)
        else:
            mixed = tm * mx[:, 0, None, None]
        mixed = mixed * self.final_gain

        return mixed


def execute_parallel_layer(
        hyper: HyperParameters,
        static: StaticTensors,
        instrument: InstrumentDefinitionTensors,
        forces: torch.Tensor,
        tension_modifier: torch.Tensor = None,
        damp_mod: torch.Tensor = None,
        tension_mod: torch.Tensor = None) -> [torch.Tensor, torch.Tensor]:

    # print(instrument.display_shapes())

    if not APPLY_DAMP_AND_TENSION_MOD:
        damp_mod = None
        tension_mod = None

    forces = torch.einsum('bct,bcd->bdt', forces, instrument.force_router)

    d = instrument.damping.repeat(1, 1, hyper.n_frames)
    if damp_mod is not None:
        d = torch.clamp(d + damp_mod, 0, 1)

    # damp = torch.abs(d)

    # damp = interpolate_last_axis(damp, desired_size=hyper.n_samples)

    mass = instrument.mass * hyper.mass_coeff   

    if INTEGRATOR == 'control_rate':
        energy = sequential(forces / mass, d)
        energy = interpolate_last_axis(energy, desired_size=hyper.n_samples)

    # the problem with this idea is that the spectrum here is not
    # variable over time
    # noisy_energy = torch.zeros_like(energy).uniform_(-1, 1)
    # noisy_energy = noisy_energy * energy
    # energy = torch.stack([energy, noisy_energy], dim=-1)
    # energy = torch.einsum('bctm,bcm->bct', energy, instrument.noise_mix)

    # instrument.tension is already real-space (a softmax combination of the
    # fixed tension_dictionary atoms, see InstrumentHyperNetwork) -- no
    # `10 ** t` here. Modifiers still accumulate in log-space as before (an
    # additive shift in log-space is a scale-invariant, proportional effect
    # in real space regardless of the base tension's magnitude), and get
    # applied as a single multiplicative `compute_tension(...)` factor. This
    # confines the live exponential to the small, already-bounded modifier
    # terms instead of the full wide-range base tension value.
    tension = instrument.tension
    log_modifier = None

    if tension_modifier is not None:
        tension_modifier = torch.einsum(
            'bct,bcd->bdt', tension_modifier, instrument.tension_router)
        log_modifier = tension_modifier

    if tension_mod is not None:
        tm = interpolate_last_axis(tension_mod, desired_size=hyper.n_samples)
        log_modifier = tm if log_modifier is None else log_modifier + tm

    if log_modifier is not None:
        tension = tension * compute_tension(log_modifier)

    if INTEGRATOR == 'sample_rate':
        x = sample_rate_oscillator(
            hyper=hyper,
            static=static,
            forces=forces,
            damping=d,
            mass=mass,
            tension=tension,
            excitation_mix=instrument.excitation_mix,
            bow_tension=instrument.bow_tension,
        )
    else:
        x = damped_harmonic_oscillator(
            energy=energy,
            time=static.time,
            mass=mass,
            damping=static.damping,
            tension=tension,
            initial_displacement=static.initial_displacement,
            clamp_to_nyquist=CLAMP_OMEGA_TO_NYQUIST,
            min_omega_per_sample=min_omega_per_sample(),
        )

    filt = fft_convolve(x, ensure_last_axis_length(
        unit_norm(instrument.filters), desired_size=hyper.n_samples))
    x = torch.stack([x, filt], dim=-1)

    x = torch.einsum('bctm,bcm->bct', x, instrument.filters_mix)

    x = torch.tanh(x * instrument.gains[..., None])

    # final, scalar (per batch item, not per node) gain stage, applied last
    x = x * instrument.output_gain[..., None]

    return x, forces


def l0_norm(x: torch.Tensor):
    mask = (x > 0).float()

    forward = mask
    backward = x

    y = backward + (forward - backward).detach()

    return y.sum()


def l1_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.abs(x).sum()


def avg_norm(x: torch.Tensor) -> torch.Tensor:
    """
    The L2 norm along the time (last) dimension, averaged over all
    remaining (batch/channel) dimensions -- a single scalar summarizing a
    batch of audio clips' overall amplitude, for comparing real vs.
    reconstructed (or synthetic vs. its reconstruction) signal scale.
    Logged alongside the losses to help diagnose amplitude mismatches, e.g.
    when changing MASS_COEFF.
    """
    return torch.norm(x, dim=-1).mean()


def maybe_max_norm(x: torch.Tensor) -> torch.Tensor:
    """
    See MAX_NORM_BEFORE_LOSS.
    """
    return max_norm(x) if MAX_NORM_BEFORE_LOSS else x


def stft(
        x: torch.Tensor,
        ws: int = 512,
        step: int = 256,
        pad: bool = False):
    frames = x.shape[-1] // step

    if pad:
        x = F.pad(x, (0, ws))
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = torch.abs(x)
    x = x[:, :, :frames, :]
    return x


@torch.jit.script
def compute_omega(
        tension: torch.Tensor,
        x: torch.Tensor,
        dt: torch.Tensor,
        min_omega_per_sample: float,
        clamp_to_nyquist: bool) -> torch.Tensor:
    """
    A node's angular frequency, in radians per synthetic time unit,
    sqrt(|tension - x ** 2|), floored at min_omega_per_sample / dt (radians
    per sample -> per time unit; see MIN_OMEGA_HZ: unfloored, it's 0 at
    tension == x ** 2, and both the output and the gradient go NaN) and, if
    clamp_to_nyquist, capped below Nyquist.  dt: the synthetic time step
    per sample.  Shared by both integrators; the settings are arguments
    because it's scripted (no module globals).
    """
    min_omega = min_omega_per_sample / dt
    omega = torch.sqrt(torch.maximum(torch.abs(tension - (x ** 2)), min_omega ** 2))
    if clamp_to_nyquist:
        omega = torch.minimum(omega, 0.95 * math.pi / dt)
    return omega


def min_omega_per_sample() -> float:
    """MIN_OMEGA_HZ in radians per sample, for compute_omega"""
    return 2 * math.pi * MIN_OMEGA_HZ / MIN_OMEGA_SAMPLERATE


@torch.jit.script
def damped_harmonic_oscillator(
        energy: torch.Tensor,
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor,
        clamp_to_nyquist: bool,
        min_omega_per_sample: float
) -> torch.Tensor:
    x = (damping / (2 * mass))

    omega = compute_omega(tension, x, time[1] - time[0], min_omega_per_sample, clamp_to_nyquist)

    phi = torch.atan2(
        (x * initial_displacement),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    # z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    z = a * energy * torch.cos(omega * time - phi)
    return z


def sample_rate_oscillator(
        hyper: HyperParameters,
        static: StaticTensors,
        forces: torch.Tensor,
        damping: torch.Tensor,
        mass: torch.Tensor,
        tension: torch.Tensor,
        excitation_mix: torch.Tensor = None,
        bow_tension: torch.Tensor = None) -> torch.Tensor:
    """
    INTEGRATOR = 'sample_rate': the same parameters as the control-rate path
    (`sequential` + damped_harmonic_oscillator), integrated per sample.

    forces: (batch, nodes, n_frames), already routed
    damping: (batch, nodes, n_frames), the per-control-frame decay
        multiplier `sequential` applies
    mass: (batch, nodes, 1)
    tension: (batch, nodes, 1) or (batch, nodes, n_samples), modifiers
        (tension_mod, routed FM) already applied
    excitation_mix: (batch, 1, 3) (or (batch, nodes, 3)), softmax weights
        over EXCITATIONS
    bow_tension: (batch, 1, 1) (or (batch, nodes, 1)), the bow frequency
        (see EXCITATIONS)
    With neither given (e.g. a Layer run on its own), strikes only.
    returns: (batch, nodes, n_samples)

    Each node is a phasor z, rotating by omega and decaying by alpha each
    sample, kicked by the drive:

        z[n] = exp(-alpha[n] + i * omega[n] * dt) * z[n - 1] + drive[n] / mass

    run as a parallel scan (sample_rate_dho.linear_recurrence).  The
    parameters map over unchanged:
      - damping d per control frame -> alpha = -log(d) / control_rate per
        sample, linearly interpolated, so damp_mod lands per sample
      - omega: exactly damped_harmonic_oscillator's, sqrt(|tension - x^2|)
        with x = static damping / (2 * mass), clamped below Nyquist, in
        parallel.py's synthetic time units (static.time), times dt to make
        it radians per sample.  Tension may vary per sample, so the routed
        FM is applied at audio rate, with phase integrated.
      - output: damped_harmonic_oscillator's a * E * cos(theta - phi) is,
        expanded, E * (cos(theta) + (x / omega) * sin(theta)), i.e.
        Re(z) + (x / omega) * Im(z): the same form, with the envelope E and
        phase theta now coming out of the resonator.
    So with an all-impulse excitation_mix, a strike sounds as before -- a decaying
    sinusoid at amplitude force / mass -- but the drive is now heard
    through the node's resonance, which is what lets noise (and later,
    other audio-rate drives) excite it physically.
    """
    n = hyper.n_samples
    control_rate = n // hyper.n_frames
    dt = static.time[1] - static.time[0]

    strike = impulse_excitation(forces, n)
    if excitation_mix is None:
        drive = strike
    else:
        # a strike puts force ** 2 of energy into a frame; unit-variance
        # noise under the same (interpolated) force puts ~control_rate *
        # force ** 2, and a unit sawtooth (variance 1/3) a third of that.
        # Scale both down to the strike's, so excitation_mix's weights
        # compare like-for-like (as jax_ae.ParallelGenerator.excitation does)
        sustain_scale = 1 / math.sqrt(control_rate)
        breath = noise_excitation(forces, n) * sustain_scale
        # the bow's frequency, in the same units as the nodes' (radians per
        # synthetic time unit, uncapped by mass), to cycles per sample.
        # sawtooth_excitation integrates phase (in float64) and caps the
        # rate at Nyquist; passing sample_rate=1 means "per sample"
        bow_cycles = torch.sqrt(bow_tension) * dt / (2 * math.pi)
        bow_cycles = bow_cycles.expand(*forces.shape[:2], 1)
        bow = sawtooth_excitation(forces, bow_cycles, sample_rate=1, n=n) * (math.sqrt(3) * sustain_scale)
        drive = torch.einsum(
            'bctm,bcm->bct', torch.stack([strike, breath, bow], dim=-1), excitation_mix)
    drive = drive / mass

    alpha = -torch.log(torch.clamp(damping, 1e-6, 1.)) / control_rate
    alpha = interpolate_last_axis(alpha, desired_size=n)

    x = static.damping / (2 * mass)
    omega = compute_omega(tension, x, dt, min_omega_per_sample(), CLAMP_OMEGA_TO_NYQUIST)

    alpha, omega_per_sample = torch.broadcast_tensors(alpha, omega * dt)
    z = linear_recurrence(torch.complex(-alpha, omega_per_sample), drive)
    return z.real + (x / omega) * z.imag


@torch.jit.script
def sequential(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(forces)
    for i in range(forces.shape[-1]):
        if i == 0:
            output[..., i] = forces[..., i]
        else:
            output[..., i] = (
                forces[..., i] + output[..., i - 1]) * damping[..., i]
    return output


def parallel_sr_independent(
        forces: torch.Tensor,
        lambda_: torch.Tensor,  # continuous-time damping rate (1/sec)
        sample_rate: float,
        eps: float = 1e-8,
) -> torch.Tensor:
    """
    forces:    (..., T)
    lambda_:   (..., T)   continuous-time damping rates
    returns:   (..., T)
    """
    dt = 1.0 / sample_rate

    # Discretize the continuous-time system
    alpha = torch.exp(-lambda_ * dt)

    # Stable computation of beta
    beta = torch.where(
        lambda_.abs() > eps,
        (1.0 - alpha) / lambda_,
        torch.full_like(lambda_, dt),  # limit as lambda -> 0
    )

    # Recurrence: o[n] = alpha[n] * o[n-1] + beta[n] * forces[n]
    b = beta * forces

    # Parallel scan formulation
    p = torch.cumprod(alpha, dim=-1)
    s = torch.cumsum(b / (p + 1e-12), dim=-1)

    return p * s


def new_parallel(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    print(forces.shape, damping.shape)
    raise NotImplementedError('')


def parallel(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    # a[i] = d[i]
    # b[i] = d[i] * f[i]
    b = damping * forces

    # p[i] = prod_{j<=i} d[j]
    p = torch.cumprod(damping, dim=-1)

    # sum_{k<=i} b[k] / p[k]
    s = torch.cumsum(b / p, dim=-1)

    # o[i] = p[i] * s[i]
    return p * s


def chunked_parallel(
        forces: torch.Tensor,
        damping: torch.Tensor,
        chunk_size: int = 2048) -> torch.Tensor:
    """
    Drop-in, numerically stable replacement for `parallel()`.

    `parallel()`'s cumprod/cumsum trick is exact in infinite precision, but
    in float32 it fails once cumprod(damping) decays into the subnormal
    range: dividing by a subnormal `p` overflows to inf, poisoning every
    later step via the running cumsum. This happens well before `p` reaches
    exact zero (e.g. lambda=80 @ 44.1kHz starts producing non-finite output
    around N=65536, long before cumprod itself underflows to 0).

    The fix is the standard chunked/blocked parallel scan used in chunked
    linear-attention/SSM implementations: run the cumprod/cumsum trick
    independently within fixed-size chunks (small enough that `p` never
    leaves the normal float32 range), then propagate the running state
    across chunk boundaries with a short sequential loop over chunks
    (cheap: n_samples / chunk_size steps, not n_samples steps).
    """
    orig_len = forces.shape[-1]
    pad = (-orig_len) % chunk_size

    if pad > 0:
        forces = F.pad(forces, (0, pad))
        damping = F.pad(damping, (0, pad), value=1.0)

    batch_shape = forces.shape[:-1]
    n_chunks = forces.shape[-1] // chunk_size

    f = forces.reshape(*batch_shape, n_chunks, chunk_size)
    d = damping.reshape(*batch_shape, n_chunks, chunk_size)

    # within-chunk parallel scan (chunk_size is chosen small enough that
    # `p` stays within the normal float32 range, so no epsilon is needed)
    b = d * f
    p = torch.cumprod(d, dim=-1)
    s = torch.cumsum(b / p, dim=-1)
    local_out = p * s

    # state to carry from the end of one chunk into the next: total decay
    # factor and final output value of each chunk
    chunk_damp_total = p[..., -1]
    chunk_final_local = local_out[..., -1]

    # sequential scan *across chunks only* (n_chunks steps, not n_samples)
    carry = torch.zeros_like(chunk_damp_total[..., 0])
    carries = []
    for c in range(n_chunks):
        carries.append(carry)
        carry = carry * chunk_damp_total[..., c] + chunk_final_local[..., c]
    carry_in = torch.stack(carries, dim=-1)

    out = local_out + carry_in[..., None] * p
    out = out.reshape(*batch_shape, n_chunks * chunk_size)
    return out[..., :orig_len]


class Layer(nn.Module):
    def __init__(
            self,
            n_nodes: int,
            n_samples: int,
            control_rate: int,
            filter_size: int,
            n_filter_taps: int = N_FILTER_TAPS,
            tension_dict_size: int = 1024,
            tension_exponent_range: Tuple[float, float] = TENSION_EXPONENT_RANGE,
            tension_dict_seed: int = 0):

        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.filter_size = filter_size
        self.n_filter_taps = n_filter_taps

        # static hyperparameter stuff
        self.mass_coeff = 1
        d = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('d', d)

        _id = torch.zeros(1, self.n_nodes, 1).fill_(1)
        self.register_buffer('_id', _id)

        t = torch.linspace(0, 10, self.n_samples)
        self.register_buffer('t', t)

        # same tension/mass formula as InstrumentHyperNetwork (see
        # compute_mass/compute_tension_from_dictionary/USE_TENSION_DICTIONARY),
        # but t_logits (or t_raw)/m_raw are directly optimized parameters
        # here instead of a hypernetwork's output -- this is the "no encoder,
        # just the weights that would be produced" path, so it needs to stay
        # in sync with that formula rather than drift onto its own.
        if USE_TENSION_DICTIONARY:
            self.register_buffer(
                'tension_dictionary',
                make_tension_dictionary(tension_dict_size, tension_exponent_range, tension_dict_seed))
            self.t_logits = nn.Parameter(torch.zeros(
                1, self.n_nodes, tension_dict_size).uniform_(-1, 1))
        else:
            # init directly into the physically-plausible tension range
            # (matching random_synthetic_instrument's known-tension
            # generation), rather than near t_raw=0 i.e. tension=1 -- compute_
            # tension's live `10 ** t` has a self-amplifying gradient, so
            # starting every node 4-9 orders of magnitude below where it
            # needs to end up means training has to climb out of that
            # regime before it can even begin to fit a real tension value
            self.t_raw = nn.Parameter(torch.zeros(
                1, self.n_nodes, 1).uniform_(*tension_exponent_range))

        self.m_raw = nn.Parameter(torch.zeros(
            1, self.n_nodes, 1).uniform_(-1, 1))

        self.damp = nn.Parameter(torch.zeros(
            1, self.n_nodes, 1).uniform_(1e-12, 0.9999))

        self.filt_taps = nn.Parameter(torch.zeros(
            1, self.n_nodes, self.n_filter_taps).uniform_(-0.01, 0.01))
        self.filt_mix = nn.Parameter(torch.zeros(
            1, self.n_nodes, 2).uniform_(-0.01, 0.01))

        # DECISION:  Should there be separate routing matrices for force and tension, or just one?

        self.force_router = nn.Parameter(torch.zeros(
            1, self.n_nodes, self.n_nodes).uniform_(-0.01, 0.01) + torch.eye(self.n_nodes, self.n_nodes))
        self.tension_router = nn.Parameter(torch.zeros(
            1, self.n_nodes, self.n_nodes).uniform_(-0.01, 0.01) + torch.eye(self.n_nodes, self.n_nodes))

        self.gain = nn.Parameter(torch.zeros(1, self.n_nodes).uniform_(0.5, 1.5))
        self.output_gain = nn.Parameter(torch.zeros(1, 1).uniform_(0.5, 1.5))

        self.base_resonance = 0.02
        self.max_resonance = 0.999
        self.diff = (1 - self.base_resonance)

    def total_mass_cost(self):
        return (torch.sigmoid(self.m_raw) * self.mass_coeff).sum()

    def total_tension_cost(self):
        return torch.abs(self.t_logits if USE_TENSION_DICTIONARY else self.t_raw).sum()

    def total_damp_cost(self):
        damp = self.base_resonance + (torch.sigmoid(self.damp) * self.diff)
        return damp.sum()

    def forward(
            self,
            forces: torch.Tensor,
            tension_modifier: torch.Tensor = None,
            damp_mod: torch.Tensor = None,
            tension_mod: torch.Tensor = None,
            excitation_mix: torch.Tensor = None,
            bow_tension: torch.Tensor = None) -> torch.Tensor:
        """
        excitation_mix: (1, 1, len(EXCITATIONS)) softmax weights, and
        bow_tension: (1, 1, 1), the instrument-wide excitation shared by
            every layer (LayerController.excitation_logits/bow_t_raw); only
            heard with --integrator sample_rate
        """

        hyper = HyperParameters(
            n_samples=self.n_samples,
            n_frames=self.n_frames,
            base_resonance=self.base_resonance,
            mass_coeff=self.mass_coeff
        )

        static = StaticTensors(
            time=self.t,
            damping=self.d,
            initial_displacement=self._id
        )

        if USE_TENSION_DICTIONARY:
            t = compute_tension_from_dictionary(self.t_logits, self.tension_dictionary)
        else:
            t = compute_tension(self.t_raw)
        mass = compute_mass(self.m_raw)
        filt = compute_filters(self.filt_taps, self.filter_size)

        instrument = InstrumentDefinitionTensors(
            mass=mass,
            tension=t,
            filters=filt,
            filters_mix=self.filt_mix,
            force_router=self.force_router,
            tension_router=self.tension_router,
            damping=self.damp,
            gains=self.gain,
            output_gain=self.output_gain,
            excitation_mix=excitation_mix,
            bow_tension=bow_tension,
        )

        x, _ = execute_parallel_layer(
            hyper=hyper,
            static=static,
            instrument=instrument,
            forces=forces,
            tension_modifier=tension_modifier,
            damp_mod=damp_mod,
            tension_mod=tension_mod,
        )
        return x


class LayerController(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_nodes: int,
            n_samples: int,
            control_rate: int,
            n_to_keep: int = 1024,
            filter_size: int = 32,
            n_filter_taps: int = 8):

        super().__init__()
        self.n_layers = n_layers
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.n_to_keep = n_to_keep
        self.filter_size = filter_size
        self.n_filter_taps = n_filter_taps

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate=22050, n_samples=n_samples)
        self.room_mix = nn.Parameter(torch.zeros(
            1, self.verb.n_rooms).uniform_(-1, 1))
        self.wet_dry_mix = nn.Parameter(torch.zeros(2).uniform_(-0.01, 0.01))

        self.forces = nn.Parameter(torch.zeros(
            1, n_nodes, self.n_frames).uniform_(-0.01, 0.01))
        self.damp_mod = nn.Parameter(torch.zeros(
            1, n_nodes, self.n_frames).uniform_(-1, 1))
        self.tension_mod = nn.Parameter(torch.zeros(
            1, n_nodes, self.n_frames).uniform_(-1, 1))

        self.layers: Iterable[Layer] = nn.ModuleList([
            Layer(
                n_nodes,
                n_samples,
                control_rate,
                filter_size,
                n_filter_taps=n_filter_taps,
                tension_dict_seed=randint(0, int(1e5)) + i)
            for i in range(self.n_layers)
        ])
        # only the last layer is mixed into the output; the rest are heard
        # as FM, so start them louder (see FM_LAYER_GAIN_DIRECT)
        if INTEGRATOR == 'sample_rate':
            with torch.no_grad():
                for layer in self.layers[:-1]:
                    layer.gain.mul_(FM_LAYER_GAIN_DIRECT)

        self.mix = nn.Parameter(torch.zeros(
            self.n_layers).uniform_(-0.01, 0.01))

        # one excitation blend and bow frequency for the whole instrument,
        # shared by every layer (see EXCITATIONS): logits, starting as an
        # even blend, and a log10 tension somewhere in the nodes' range
        self.excitation_logits = nn.Parameter(torch.zeros(1, 1, len(EXCITATIONS)))
        self.bow_t_raw = nn.Parameter(torch.zeros(1, 1, 1).uniform_(*TENSION_EXPONENT_RANGE))

        # a single scalar gain, shared across every layer instance (unlike
        # each Layer's own per-layer output_gain, applied node-wise inside
        # execute_parallel_layer before the layers are mixed/summed), applied
        # once as the very last step before the final audio is returned
        self.final_gain = nn.Parameter(torch.zeros(1).uniform_(0.5, 1.5))

    @property
    def total_params(self):
        return \
            64 + 16 + 16 \
            + sum([count_parameters(x) for x in self.layers]) \
            + self.room_mix.numel() \
            + self.wet_dry_mix.numel() \
            + self.mix.numel()

    def total_mass_cost(self):
        return sum([l.total_mass_cost() for l in self.layers])

    def total_tension_cost(self):
        return sum([l.total_tension_cost() for l in self.layers])

    def total_damp_cost(self):
        return sum([l.total_damp_cost() for l in self.layers])

    def materialize_damping_mod(self):
        dm = self.damp_mod * 0.001
        dm = sparsify(dm, n_to_keep=16, salience=torch.abs(dm))
        return dm

    def materialize_tension_mod(self):
        tm = self.tension_mod * 0.001
        tm = sparsify(tm, n_to_keep=16, salience=torch.abs(tm))
        return tm

    def materialize_forces(
            self,
            forces: torch.Tensor = None,
            do_upsample: bool = True,
            n_to_keep: int = None) -> torch.Tensor:

        if forces is not None:
            f = forces
        else:
            f = self.forces

        f = torch.abs(f)
        f = f - f.mean()
        f = torch.relu(f)

        f = sparsify(f, n_to_keep=n_to_keep or self.n_to_keep)
        return f

    def compression_ratio(self):
        # For each sparse event, we need amplitude, time, and control_plane dimension

        params_per_event = 3
        params = count_parameters(self.layers) + \
            (self.n_to_keep * params_per_event)

        return params / self.n_samples

    def random_forward(self, sum_output: bool = True, n_to_keep: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            f = torch.zeros_like(self.forces).uniform_(
                0, self.forces.max().item()) ** 2
            f = sparsify(f, n_to_keep=n_to_keep)
        except:
            f = torch.zeros_like(self.forces).uniform_(0, 1)

        return self.forward(sum_output=sum_output, forces=f, n_to_keep=n_to_keep)

    def forward(
            self,
            sum_output: bool = True,
            forces: torch.Tensor = None,
            n_to_keep: int = None) -> Tuple[torch.Tensor, torch.Tensor]:

        if forces is not None:
            sparse_forces = self.materialize_forces(
                forces, n_to_keep=n_to_keep)
        else:
            sparse_forces = self.materialize_forces(
                self.forces, n_to_keep=n_to_keep)

        dm = self.materialize_damping_mod()
        tension_mod = self.materialize_tension_mod()

        excitation_mix = torch.softmax(self.excitation_logits, dim=-1)
        bow_tension = compute_tension(self.bow_t_raw)

        tm = None
        outputs = []
        for i, layer in enumerate(self.layers):

            tm = layer.forward(
                forces=sparse_forces,
                tension_modifier=tm,
                damp_mod=dm,
                tension_mod=tension_mod,
                excitation_mix=excitation_mix,
                bow_tension=bow_tension,
            )
            outputs.append(tm)

        # DECISION: Does the conv reverb cause issues because it's outside the energy calculation?
        if USE_REVERB:
            # apply conv reverb
            wet = self.verb.forward(tm, self.room_mix)

            # wet/dry reverb mix
            x = torch.stack([tm, wet], dim=-1)
            x = x * self.wet_dry_mix
            tm = torch.sum(x, dim=-1)
        else:
            tm = tm * self.wet_dry_mix[0]

        if sum_output:
            tm = torch.sum(tm, dim=1, keepdim=True)

        tm = tm * self.final_gain

        return tm, sparse_forces

# spiking = SpikingModel(64, 64, 64, 64, 64).to(device)

def loss_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    
    # base_loss = spiking.compute_multiband_loss(a, b, hard=True)

    # a = flattened_multiband_spectrogram(
    #     a, {'xs': (64, 16)}, smallest_band_size=512)
    # b = flattened_multiband_spectrogram(
    #     b, {'xs': (64, 16)}, smallest_band_size=512)

    a = stft(a, 2048, 256, pad=False)
    b = stft(b, 2048, 256, pad=False)

    base_loss = torch.abs(a - b).sum()
    return base_loss


def entropy_deficit(spec: torch.Tensor) -> torch.Tensor:
    """
    How far a (magnitude) spectrogram is from flat, i.e., from uniform
    noise, regardless of its overall scale: 1 - (entropy of the
    spectrogram, normalized to sum to one) / (the maximum possible entropy,
    log of the number of cells). 0 when already flat/noise-like, 1 when all
    energy sits in a single cell. Negative values are clipped to zero
    first, so an over-subtracted residual counts as a hole, not energy.
    Ported from jax_ae.py's entropy_deficit.
    """
    a = torch.clamp(spec, min=0).reshape(-1)
    p = a / (a.sum() + 1e-12)
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return 1. - entropy / math.log(a.numel())


def information_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    The auxiliary "information" loss from jax_ae.py, adapted to parallel.py's
    single-shot (non-greedy) reconstruction: the entropy deficit of the
    *spectrogram* residual (a's magnitude spectrogram minus b's, clipped to
    >= 0) between a real/synthetic target `a` and its reconstruction `b`.
    Pushing this toward zero asks the residual to look like unstructured,
    uniform noise -- regardless of scale -- rather than asking it to be
    small (that's loss_func's job).
    """
    a = stft(a, 2048, 256, pad=False)
    b = stft(b, 2048, 256, pad=False)
    residual = torch.clamp(a - b, min=0)
    return entropy_deficit(residual)


def random_synthetic_excitation(batch_size: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    one random excitation per instrument (see EXCITATIONS): a blend,
    (batch, 1, 3), and a bow tension, (batch, 1, 1)
    """
    mix = torch.softmax(
        torch.zeros(batch_size, 1, len(EXCITATIONS), device=device).uniform_(-2, 2), dim=-1)
    bow_tension = compute_tension(
        torch.zeros(batch_size, 1, 1, device=device).uniform_(*TENSION_EXPONENT_RANGE))
    return mix, bow_tension


def random_synthetic_instrument(
        batch_size: int,
        n_nodes: int,
        filter_size: int,
        base_resonance: float,
        device,
        n_filter_taps: int = N_FILTER_TAPS,
        excitation: Tuple[torch.Tensor, torch.Tensor] = None) -> InstrumentDefinitionTensors:
    """
    excitation: (excitation_mix, bow_tension), from
    random_synthetic_excitation -- pass the same one to every layer, so the
    synthetic instrument has a single excitation like the model's.  If None,
    this layer draws its own.

    Randomly samples one layer's worth of known InstrumentDefinitionTensors
    for self_supervised's synthetic-data parameter-recovery loss. Ranges
    mirror Layer's own randomly-initialized parameters (i.e. "what a
    plausible instrument looks like"), and mass/tension/filters reuse the
    same compute_mass/compute_tension/compute_filters helpers the model
    itself uses, so the known values stay in the same distribution family
    the model is actually trying to recover, rather than drifting onto
    their own convention -- in particular, known filters are low-pass by
    construction too (see compute_filters), not plain white noise.
    """
    diff = 1 - base_resonance

    mass = compute_mass(torch.zeros(batch_size, n_nodes, 1, device=device).uniform_(-3, 3))
    tension = compute_tension(torch.zeros(batch_size, n_nodes, 1, device=device).uniform_(*TENSION_EXPONENT_RANGE))
    damping = base_resonance + (
        torch.sigmoid(torch.zeros(batch_size, n_nodes, 1, device=device).uniform_(-3, 3)) * diff * 0.9998)

    filter_taps = torch.zeros(batch_size, n_nodes, n_filter_taps, device=device).uniform_(-0.01, 0.01)
    filters = compute_filters(filter_taps, filter_size)
    filters_mix = torch.zeros(batch_size, n_nodes, 2, device=device).uniform_(-0.01, 0.01)

    eye = torch.eye(n_nodes, device=device)
    force_router = torch.zeros(batch_size, n_nodes, n_nodes, device=device).uniform_(-0.01, 0.01) + eye
    tension_router = torch.zeros(batch_size, n_nodes, n_nodes, device=device).uniform_(-0.01, 0.01) + eye

    gains = torch.zeros(batch_size, n_nodes, device=device).uniform_(0.5, 1.5)
    output_gain = torch.zeros(batch_size, 1, device=device).uniform_(0.5, 1.5)

    if excitation is None:
        excitation = random_synthetic_excitation(batch_size, device)
    excitation_mix, bow_tension = excitation

    return InstrumentDefinitionTensors(
        mass=mass,
        tension=tension,
        filters=filters,
        filters_mix=filters_mix,
        force_router=force_router,
        tension_router=tension_router,
        damping=damping,
        gains=gains,
        output_gain=output_gain,
        excitation_mix=excitation_mix,
        bow_tension=bow_tension,
    )


def random_synthetic_control_signals(
        batch_size: int,
        n_nodes: int,
        n_frames: int,
        device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Randomly samples known control signals (forces, tension_mod, damp_mod)
    for self_supervised -- sparse, like real control signals are meant to
    be, rather than dense noise. `cs` (forces) gets its own sparsity budget,
    randomized per call across a 16-32 nonzero-element range, rather than
    reusing the model's overall n_to_keep (which is typically much larger);
    tension_mod/damp_mod match the n_to_keep=16 the model's own
    self.tension/self.damp heads already use.
    """
    def sparse_signal(scale: float, n_to_keep: int):
        x = torch.zeros(batch_size, n_nodes, n_frames, device=device).uniform_(0, 1)
        x = sparsify(x, n_to_keep=n_to_keep)
        return x * scale

    cs = sparse_signal(1.0, n_to_keep=randint(16, 32))
    tension_mod = sparse_signal(0.001, n_to_keep=16)
    damp_mod = sparse_signal(0.001, n_to_keep=16)
    return cs, tension_mod, damp_mod


def synthesize_known_signal(
        hyper: HyperParameters,
        cs: torch.Tensor,
        tension_mod: torch.Tensor,
        damp_mod: torch.Tensor,
        params_list: List[InstrumentDefinitionTensors],
        rooms: torch.Tensor,
        wet_dry_mix: torch.Tensor,
        verb: NeuralReverb,
        device) -> torch.Tensor:
    """
    Runs known instrument params/control signals through the same
    execute_parallel_layer stack InstrumentAutoencoder.forward uses, then
    applies reverb exactly as forward() does -- but using known rooms/
    wet_dry_mix (sampled alongside the other known parameters) rather than
    values derived from an encoder latent, since there's no such latent on
    the known/synthetic side. `verb` is the model's own NeuralReverb module
    (stateless wrt latents, so safe to reuse directly here), passed in so
    the synthetic target uses the same impulse responses the model does.
    """
    batch, n_nodes, _ = cs.shape

    static = StaticTensors(
        time=torch.linspace(0, 10, hyper.n_samples, device=device),
        damping=torch.zeros(1, n_nodes, 1, device=device).fill_(1),
        initial_displacement=torch.zeros(1, n_nodes, 1, device=device).fill_(1)
    )

    tm = None
    layer_outputs = []
    for params in params_list:
        if FORWARD_FORCE:
            tm, cs = execute_parallel_layer(
                instrument=params, hyper=hyper, static=static, forces=cs,
                damp_mod=damp_mod, tension_mod=tension_mod, tension_modifier=tm)
        else:
            tm, _cs = execute_parallel_layer(
                instrument=params, hyper=hyper, static=static, forces=cs,
                damp_mod=damp_mod, tension_mod=tension_mod, tension_modifier=tm)
        layer_outputs.append(tm)

    if SUM_LAYER_OUTPUTS:
        stacked = torch.stack(layer_outputs, dim=-1)
        uniform_mix = torch.ones(len(layer_outputs), device=device) / len(layer_outputs)
        tm = torch.einsum('bctl,l->bct', stacked, uniform_mix)
    else:
        tm = layer_outputs[-1]

    tm = torch.sum(tm, dim=1, keepdim=True)

    if not USE_REVERB:
        return tm * wet_dry_mix[:, 0, None, None]

    wet = verb.forward(tm, rooms)
    stacked = torch.stack([tm, wet], dim=-1)
    mixed = torch.einsum('bctm,bm->bct', stacked, wet_dry_mix)

    return mixed


def instrument_param_loss(
        known_cs: torch.Tensor, est_cs: torch.Tensor,
        known_tension_mod: torch.Tensor, est_tension_mod: torch.Tensor,
        known_damp_mod: torch.Tensor, est_damp_mod: torch.Tensor,
        known_params: List[InstrumentDefinitionTensors],
        est_params: List[InstrumentDefinitionTensors],
        known_rooms: torch.Tensor, est_rooms: torch.Tensor,
        known_wet_dry: torch.Tensor, est_wet_dry: torch.Tensor) -> torch.Tensor:
    """
    Compares estimated vs. known parameters/control signals for
    self_supervised's synthetic examples.

    The node axis (force_router's columns; tension_router's rows and
    columns; mass/tension/damping/gains; tension_mod/damp_mod) and the
    control-channel axis (cs; force_router's rows) are each only defined up
    to an arbitrary permutation -- nothing in the architecture ties either
    one to our synthetic generation's node/channel index order, since the
    model's own output only ever depends on these through operations
    (summing over nodes, routing-matrix contractions) that don't care about
    that ordering. A raw element-wise comparison would only be correctly
    zero at the one labeling, out of (n_nodes!)**2 equally valid ones, that
    happens to match ours -- so instead this compares permutation-invariant
    statistics in layers of decreasing strength:

    - force_router/tension_router: singular values. Permutation matrices
      are orthogonal, and SVD is invariant under A -> U A V for any
      orthogonal U, V, so this is an exact invariant to independent row/
      column permutation, no matching required.
    - cs (channel axis = control-channel identity) and tension_mod/
      damp_mod (channel axis = node identity): summed over that axis,
      giving a time series that's exactly invariant to it -- at the cost of
      no longer supervising *which* channel/node does what, only the net
      aggregate signal.
    - mass/tension/damping/gains: sorted along the node axis. Exact and
      assignment-free per field in isolation, but sorting each field
      independently doesn't guarantee the same permutation was used across
      fields, so this does not check whether e.g. the right mass got
      paired with the right damping within a node.

    excitation_mix and bow_tension are one value per instrument (shared by
    every layer), so they have no node permutation to worry about: compared
    directly, once (bow tension in log-space, as tension below).

    filters/filters_mix are left out of this loss for now: they're
    vector-valued per node, so there's no simple scalar sort key, and a real
    fix needs actual bipartite (Hungarian) matching on a combined per-node
    descriptor -- left for later rather than papering over it with an
    arbitrary/unstable proxy.

    Tension is compared in log-space: its real-space range (~1e4-1e8) is
    many orders of magnitude wider than every other field here, so a plain
    L1 difference on raw tension values would swamp the loss and make
    every other term irrelevant by comparison.
    """
    loss = (
        torch.abs(known_cs.sum(dim=1) - est_cs.sum(dim=1)).sum()
        + torch.abs(known_tension_mod.sum(dim=1) - est_tension_mod.sum(dim=1)).sum()
        + torch.abs(known_damp_mod.sum(dim=1) - est_damp_mod.sum(dim=1)).sum()
    )
    # with --no-reverb, neither the known nor the estimated rooms, nor the
    # wet weight, are heard, so there's nothing there to recover; the dry
    # weight still is
    if USE_REVERB:
        loss = loss \
            + torch.abs(known_rooms - est_rooms).sum() \
            + torch.abs(known_wet_dry - est_wet_dry).sum()
    else:
        loss = loss + torch.abs(known_wet_dry[:, 0] - est_wet_dry[:, 0]).sum()

    loss = loss \
        + torch.abs(known_params[0].excitation_mix - est_params[0].excitation_mix).sum() \
        + torch.abs(
            torch.log10(known_params[0].bow_tension + 1e-8)
            - torch.log10(est_params[0].bow_tension + 1e-8)).sum()

    for kp, ep in zip(known_params, est_params):
        known_mass = torch.sort(kp.mass, dim=1).values
        est_mass = torch.sort(ep.mass, dim=1).values

        known_log_tension = torch.sort(torch.log10(kp.tension + 1e-8), dim=1).values
        est_log_tension = torch.sort(torch.log10(ep.tension + 1e-8), dim=1).values

        known_damping = torch.sort(kp.damping, dim=1).values
        est_damping = torch.sort(ep.damping, dim=1).values

        known_gains = torch.sort(kp.gains, dim=1).values
        est_gains = torch.sort(ep.gains, dim=1).values

        known_force_router_svals = torch.linalg.svdvals(kp.force_router)
        est_force_router_svals = torch.linalg.svdvals(ep.force_router)

        known_tension_router_svals = torch.linalg.svdvals(kp.tension_router)
        est_tension_router_svals = torch.linalg.svdvals(ep.tension_router)

        loss = loss \
            + torch.abs(known_mass - est_mass).sum() \
            + torch.abs(known_log_tension - est_log_tension).sum() \
            + torch.abs(known_damping - est_damping).sum() \
            + torch.abs(known_gains - est_gains).sum() \
            + torch.abs(known_force_router_svals - est_force_router_svals).sum() \
            + torch.abs(known_tension_router_svals - est_tension_router_svals).sum() \
            + torch.abs(kp.output_gain - ep.output_gain).sum()

    return loss


def train_ae(batch_size: int, n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):
    control_rate = 512
    samplerate = 22050
    base_resonance = BASE_RESONANCE

    analysis_model = InstrumentAutoencoder(
        n_samples=n_samples,
        n_nodes=n_nodes,
        control_rate=control_rate,
        n_layers=n_layers,
        channels=CHANNELS,
        filter_size=FILTER_SIZE,
        n_to_keep=n_to_keep,
        base_resonance=base_resonance,
        tension_dict_seed=randint(0, int(1e5))).to(device)

    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True)

    collection = LmdbCollection(path='parallel')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        orig_audio,
        recon_audio,
    ],
        port=9998,
        n_workers=1,
        web_components_version='0.0.101'
    )

    optim = Adam(analysis_model.parameters(), lr=LEARNING_RATE)

    for i, batch in enumerate(iter(stream)):
        optim.zero_grad()
        batch = batch.view(-1, 1, n_samples).to(device)
        orig_audio(batch[0, ...])

        control_signal, recon = analysis_model.forward(batch)

        recon_audio(max_norm(recon[0, ...]))
        
        # if we are "hard" sparsifying, then don't bother with sparsity loss
        sparsity_coeff = 0 if SPARSIFY else 1
        info_loss = information_loss(batch, maybe_max_norm(recon))
        loss = loss_func(batch, maybe_max_norm(recon)) \
            + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff) \
            + (info_loss * INFORMATION_COEFF)

        sparsity = (control_signal > 0).sum() / control_signal.numel()

        loss.backward()

        valid_gradients = all(
            not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in analysis_model.parameters() if p.grad is not None
        )

        if valid_gradients:
            # torch.nn.utils.clip_grad_norm_(analysis_model.parameters(), 0.1)
            torch.nn.utils.clip_grad_value_(analysis_model.parameters(), 0.5)
            optim.step()
        else:
            print("Warning: Detected NaN/Inf values in gradients. Skipping step.")
            continue

        # optim.step()

        print(
            i, loss.item(), sparsity.item(),
            'info', info_loss.item(),
            'real_norm', avg_norm(batch).item(),
            'recon_norm', avg_norm(recon).item())


def self_supervised(batch_size: int, n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):
    """
    Augments train_ae's real-example reconstruction training with synthetic
    examples: known instrument parameters/control signals generate a known
    audio signal, which is then run back through the *same* model (exactly
    as any real example would be). For synthetic examples only, we also
    know the ground truth, so in addition to the usual reconstruction loss
    we can directly supervise parameter recovery via instrument_param_loss.
    Real examples are handled identically to train_ae -- this mode doesn't
    change that path at all, it just adds a second kind of example each step.
    """
    control_rate = 512
    samplerate = 22050
    base_resonance = BASE_RESONANCE
    filter_size = FILTER_SIZE
    n_frames = n_samples // control_rate

    analysis_model = InstrumentAutoencoder(
        n_samples=n_samples,
        n_nodes=n_nodes,
        control_rate=control_rate,
        n_layers=n_layers,
        channels=CHANNELS,
        filter_size=filter_size,
        n_to_keep=n_to_keep,
        base_resonance=base_resonance,
        tension_dict_seed=randint(0, int(1e5))).to(device)

    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True)

    collection = LmdbCollection(path='parallel_self_supervised')

    recon_audio, orig_audio, synth_audio, synth_recon_audio = loggers(
        ['recon', 'orig', 'synth', 'synth-recon'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        orig_audio,
        recon_audio,
        synth_audio,
        synth_recon_audio,
    ],
        port=9998,
        n_workers=1,
        web_components_version='0.0.101'
    )

    optim = Adam(analysis_model.parameters(), lr=LEARNING_RATE)

    for i, batch in enumerate(iter(stream)):
        optim.zero_grad()
        batch = batch.view(-1, 1, n_samples).to(device)
        orig_audio(batch[0, ...])

        # --- real example: identical to train_ae ---
        control_signal, recon = analysis_model.forward(batch)
        recon_audio(max_norm(recon[0, ...]))

        sparsity_coeff = 0 if SPARSIFY else 1
        real_info_loss = information_loss(batch, maybe_max_norm(recon))
        real_loss = loss_func(batch, maybe_max_norm(recon)) \
            + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff) \
            + (real_info_loss * INFORMATION_COEFF)
        sparsity = (control_signal > 0).sum() / control_signal.numel()

        # --- synthetic example: known params/control signals -> known audio ---
        known_cs, known_tension_mod, known_damp_mod = random_synthetic_control_signals(
            batch_size=batch.shape[0], n_nodes=n_nodes, n_frames=n_frames,
            device=batch.device)
        known_excitation = random_synthetic_excitation(batch.shape[0], batch.device)
        known_params = [
            random_synthetic_instrument(
                batch_size=batch.shape[0], n_nodes=n_nodes, filter_size=filter_size,
                base_resonance=base_resonance, device=batch.device,
                excitation=known_excitation)
            for _ in range(n_layers)
        ]
        known_rooms = torch.zeros(
            batch.shape[0], analysis_model.verb.n_rooms, device=batch.device).uniform_(0, 1)
        known_wet_dry = torch.zeros(batch.shape[0], 2, device=batch.device).uniform_(-1, 1)

        synthetic_audio = synthesize_known_signal(
            hyper=analysis_model.hyper_params,
            cs=known_cs, tension_mod=known_tension_mod, damp_mod=known_damp_mod,
            params_list=known_params, rooms=known_rooms, wet_dry_mix=known_wet_dry,
            verb=analysis_model.verb, device=batch.device)
        synth_audio(synthetic_audio[0, ...])

        synth_control_signal, synth_recon = analysis_model.forward(synthetic_audio)
        synth_recon_audio(max_norm(synth_recon[0, ...]))
        synth_info_loss = information_loss(maybe_max_norm(synthetic_audio), maybe_max_norm(synth_recon))
        synth_recon_loss = loss_func(maybe_max_norm(synthetic_audio), maybe_max_norm(synth_recon)) \
            + (synth_info_loss * INFORMATION_COEFF)

        est_cs, est_tension_mod, est_damp_mod, est_params, est_rooms, est_wet_dry = \
            analysis_model.estimate_params(synthetic_audio)
        param_loss = instrument_param_loss(
            known_cs, est_cs, known_tension_mod, est_tension_mod, known_damp_mod, est_damp_mod,
            known_params, est_params, known_rooms, est_rooms, known_wet_dry, est_wet_dry)

        loss = real_loss + synth_recon_loss + param_loss

        loss.backward()

        valid_gradients = all(
            not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in analysis_model.parameters() if p.grad is not None
        )

        if valid_gradients:
            torch.nn.utils.clip_grad_value_(analysis_model.parameters(), 0.5)
            optim.step()
        else:
            print("Warning: Detected NaN/Inf values in gradients. Skipping step.")
            continue

        print(
            i,
            'real', real_loss.item(),
            'synth_recon', synth_recon_loss.item(),
            'param', param_loss.item(),
            'sparsity', sparsity.item(),
            'real_info', real_info_loss.item(),
            'synth_info', synth_info_loss.item(),
            'real_norm', avg_norm(batch).item(),
            'recon_norm', avg_norm(recon).item(),
            'synth_norm', avg_norm(synthetic_audio).item(),
            'synth_recon_norm', avg_norm(synth_recon).item())


def self_supervised_only(batch_size: int, n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):
    """
    Same synthetic-data task as self_supervised (known params/control
    signals -> known audio -> reconstruction + parameter-recovery loss),
    but with no real-audio example and no AudioIterator -- every step is
    synthetic only. Useful in isolation, to check whether the model can
    recover known parameters at all before mixing that signal in with
    real-example reconstruction.
    """
    control_rate = 512
    base_resonance = BASE_RESONANCE
    filter_size = FILTER_SIZE
    n_frames = n_samples // control_rate

    analysis_model = InstrumentAutoencoder(
        n_samples=n_samples,
        n_nodes=n_nodes,
        control_rate=control_rate,
        n_layers=n_layers,
        channels=CHANNELS,
        filter_size=filter_size,
        n_to_keep=n_to_keep,
        base_resonance=base_resonance,
        tension_dict_seed=randint(0, int(1e5))).to(device)

    collection = LmdbCollection(path='parallel_self_supervised_only')

    synth_audio, synth_recon_audio = loggers(
        ['synth', 'synth-recon'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        synth_audio,
        synth_recon_audio,
    ],
        port=9998,
        n_workers=1,
        web_components_version='0.0.101'
    )

    optim = Adam(analysis_model.parameters(), lr=LEARNING_RATE)

    i = 0
    while True:
        optim.zero_grad()

        known_cs, known_tension_mod, known_damp_mod = random_synthetic_control_signals(
            batch_size=batch_size, n_nodes=n_nodes, n_frames=n_frames,
            device=device)
        known_excitation = random_synthetic_excitation(batch_size, device)
        known_params = [
            random_synthetic_instrument(
                batch_size=batch_size, n_nodes=n_nodes, filter_size=filter_size,
                base_resonance=base_resonance, device=device,
                excitation=known_excitation)
            for _ in range(n_layers)
        ]
        known_rooms = torch.zeros(
            batch_size, analysis_model.verb.n_rooms, device=device).uniform_(0, 1)
        known_wet_dry = torch.zeros(batch_size, 2, device=device).uniform_(-1, 1)

        synthetic_audio = synthesize_known_signal(
            hyper=analysis_model.hyper_params,
            cs=known_cs, tension_mod=known_tension_mod, damp_mod=known_damp_mod,
            params_list=known_params, rooms=known_rooms, wet_dry_mix=known_wet_dry,
            verb=analysis_model.verb, device=device)
        synth_audio(synthetic_audio[0, ...])

        synth_control_signal, synth_recon = analysis_model.forward(synthetic_audio)
        synth_recon_audio(max_norm(synth_recon[0, ...]))
        synth_info_loss = information_loss(maybe_max_norm(synthetic_audio), maybe_max_norm(synth_recon))
        synth_recon_loss = loss_func(maybe_max_norm(synthetic_audio), maybe_max_norm(synth_recon)) \
            + (synth_info_loss * INFORMATION_COEFF)

        est_cs, est_tension_mod, est_damp_mod, est_params, est_rooms, est_wet_dry = \
            analysis_model.estimate_params(synthetic_audio)
        param_loss = instrument_param_loss(
            known_cs, est_cs, known_tension_mod, est_tension_mod, known_damp_mod, est_damp_mod,
            known_params, est_params, known_rooms, est_rooms, known_wet_dry, est_wet_dry)

        loss = synth_recon_loss + param_loss

        loss.backward()

        valid_gradients = all(
            not (torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            for p in analysis_model.parameters() if p.grad is not None
        )

        if valid_gradients:
            torch.nn.utils.clip_grad_value_(analysis_model.parameters(), 0.5)
            optim.step()
        else:
            print("Warning: Detected NaN/Inf values in gradients. Skipping step.")
            i += 1
            continue

        print(
            i,
            'synth_recon', synth_recon_loss.item(),
            'param', param_loss.item(),
            'synth_info', synth_info_loss.item(),
            'synth_norm', avg_norm(synthetic_audio).item(),
            'synth_recon_norm', avg_norm(synth_recon).item())

        i += 1


def overfit_autoencoder(n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):
    control_rate = 512

    analysis_model = InstrumentAutoencoder(
        n_samples=n_samples,
        n_nodes=n_nodes,
        control_rate=control_rate,
        n_layers=n_layers,
        channels=CHANNELS,
        filter_size=FILTER_SIZE,
        n_to_keep=n_to_keep,
        tension_dict_seed=randint(0, int(1e5))).to(device)

    def logger_factory(collection):
        rnd, = conjure.loggers(
            ['random'],
            SupportedContentType.Audio.value,
            encode_audio,
            collection,
            IdentitySerializer(),
            IdentityDeserializer())

        return [rnd]

    def training_loop_hook(
            iteration: int,
            loggers: List[conjure.Conjure],
            model: InstrumentAutoencoder,
            optim: Optimizer,
            target: torch.Tensor):

        rnd_logger = loggers[-1]

        with torch.no_grad():
            # highly-sparse (4-16 events) random control signal, to sanity
            # check that the current estimated instrument sounds plausible
            # when driven by something other than the encoder's own forces
            rnd = model.random_forward(target, n_to_keep=randint(4, 16))
            rnd_logger(max_norm(rnd))

    def model_eval(model: InstrumentAutoencoder, _, target: torch.Tensor):
        control_signal, recon = model.forward(target)

        sparsity = (control_signal > 0).sum() / control_signal.numel()
        info_loss = information_loss(target, recon)
        print(sparsity.item(), 'info', info_loss.item())

        # if we are "hard" sparsifying, then don't bother with sparsity loss
        sparsity_coeff = 0 if SPARSIFY else 1
        loss = loss_func(target, recon) \
            + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff) \
            + (info_loss * INFORMATION_COEFF)
        return recon, loss

    overfit_model(
        n_samples=n_samples,
        model=analysis_model,
        loss_func=loss_func,
        logger_factory=logger_factory,
        training_loop_hook=training_loop_hook,
        model_eval=model_eval,
        collection_name='parallel',
        learning_rate=LEARNING_RATE,
        port=9998
    )


# --- web demo share-link export ---------------------------------------------
#
# Hand-ports demos/dho-synth/main.js's encodeShareState binary format, so a
# training run can periodically emit a share URL for the *instrument* (per-
# node physics params + layer-2 routing matrices) overfit_osc is currently
# fitting -- not the performance (LayerController.forces, the fitted sparse
# excitation events), which has no slot in that format. The point is to let
# you play a *new* performance on whatever instrument the model has
# converged on, not to replay its fit.
#
# This assumes the web demo's layer1/layer2 topology, which was built to
# mirror this exact Layer/LayerController pair, but a few things still need
# converting rather than copying -- see each helper's docstring below. There
# is no shared source of truth between this and main.js, so keep the two in
# lockstep by hand if either format changes.
# v2 adds forceRouter1 (layer 1's own self-routing matrix, n x n -- see
# _layer_web_nodes), matching main.js's SHARE_VERSION bump.
SHARE_VERSION_WEB = 2
SHARE_FILTER_TAPS = 32  # must match demos/dho-synth/main.js's FILTER_TAPS
SHARE_ROUTER_QUANT_RANGE = 32  # must match main.js's ROUTER_QUANT_RANGE
SHARE_EXCITATION_MODE_IDS = ['sustained', 'impulse', 'pulse', 'noise']


def _share_quantize_u8(v: float, lo: float, hi: float) -> int:
    return max(0, min(255, round(((v - lo) / (hi - lo)) * 255)))


def _share_quantize_log_u8(v: float, lo: float, hi: float) -> int:
    return _share_quantize_u8(math.log10(v), math.log10(lo), math.log10(hi))


def _share_quantize_i8(v: float) -> int:
    return max(-127, min(127, round(v * 127)))


def _share_quantize_i16(v: float, rng: float) -> int:
    return max(-32767, min(32767, round((v / rng) * 32767)))


def _share_encode_node(
        mass: float,
        tensionExp: float,
        dampLambda: float,
        excitationGain: float,
        gain: float,
        filterTaps: List[float],
        filterMix: List[float]) -> bytes:
    header = struct.pack('<5f', mass, tensionExp, dampLambda, excitationGain, gain)
    taps = struct.pack(
        f'<{SHARE_FILTER_TAPS}b', *[_share_quantize_i8(v) for v in filterTaps])
    mix = struct.pack('<2b', _share_quantize_i8(filterMix[0]), _share_quantize_i8(filterMix[1]))
    return header + taps + mix


def _layer_web_nodes(
        layer: 'Layer',
        tension_offset: float,
        control_rate: int,
        samplerate: int) -> List[dict]:
    """
    Extracts one Layer's per-node params into the shape/units
    demos/dho-synth/main.js's Node class expects:

    - mass/tensionExp/filters/filterMix/gain already use the exact same
      formulas/conventions on both sides (compute_mass, 10**tensionExp,
      fft_resample(is_lowest_band=True) from n_filter_taps=8 up to
      filter_size=32, unconstrained linear filter/gain blends) -- just
      unwrapped from their (1, n_nodes, ...) tensor shapes.
    - tensionExp gets `tension_offset` added: parallel.py's oscillator phase
      runs on a `t = linspace(0, 10, n_samples)` clock, not the real
      samplerate, so its physics clock runs faster than real/wall-clock time
      by a fixed, derivable ratio; see encode_share_state_url.
    - dampLambda converts parallel.py's raw per-control-frame multiplicative
      decay (`layer.damp`, applied once per control_rate-sample frame, then
      linearly upsampled) into the web's continuous per-sample exponential
      rate, by matching the decay over one frame's worth of real time. This
      reproduces the overall decay slope, not parallel.py's own (coarser,
      linearly-interpolated) envelope shape.
    - excitationGain has no analog in parallel.py (force_router's entries
      already *are* the per-node gain, with no separate scalar on top): now
      that both layers' full force_router matrices have a home in the web
      format (forceRouter1/forceRouter2, see encode_share_state_url),
      excitationGain is just left at the web's own identity default (1.0)
      for every node, on both layers, to avoid double-applying gain.
    """
    n = layer.n_nodes
    with torch.no_grad():
        mass = compute_mass(layer.m_raw).reshape(n).cpu().numpy()
        tension_exp = layer.t_raw.reshape(n).cpu().numpy() + tension_offset

        damp_raw = np.clip(layer.damp.reshape(n).cpu().numpy(), 1e-6, 0.999999)
        damp_lambda = -np.log(damp_raw) * samplerate / control_rate

        gain = layer.gain.reshape(n).cpu().numpy()

        filt = unit_norm(compute_filters(layer.filt_taps, layer.filter_size), dim=-1)
        filt = filt.reshape(n, -1).cpu().numpy()
        filt_mix = layer.filt_mix.reshape(n, 2).cpu().numpy()

        excitation_gain = np.ones(n)

    return [
        dict(
            mass=float(mass[j]),
            tensionExp=float(tension_exp[j]),
            dampLambda=float(damp_lambda[j]),
            excitationGain=float(excitation_gain[j]),
            gain=float(gain[j]),
            filterTaps=filt[j].tolist(),
            filterMix=filt_mix[j].tolist(),
        )
        for j in range(n)
    ]


def encode_share_state_url(controller: 'LayerController', samplerate: int = 22050) -> str:
    """
    Builds a `?p=...` query string (the fragment main.js's buildShareUrl
    appends to the page URL) encoding `controller`'s current instrument, in
    the exact binary layout main.js's encodeShareState produces. Assumes
    exactly 2 layers (layer 1 = FM modulator, layer 2 = the only audible
    one), matching the web demo's topology, and the direct (non-dictionary)
    tension path (USE_TENSION_DICTIONARY=False).

    Deliberately does not touch reverb (reverb file index 255 = "none"; the
    web leaves whatever IR is currently loaded alone in that case) and
    leaves sparsity/pulseRate/volume/modDepth at harmless defaults -- none of
    those describe the instrument itself, only UI/performance state that a
    share link can't usefully carry anyway (see this section's docstring re:
    LayerController.forces).
    """
    assert not USE_TENSION_DICTIONARY, \
        'share export assumes the direct (non-dictionary) tension path'
    assert controller.n_layers == 2, \
        'share export assumes exactly 2 layers, matching the web demo (layer1=modulator, layer2=audible)'

    layer1, layer2 = controller.layers[0], controller.layers[1]
    n = controller.n_nodes

    # parallel.py's oscillator phase runs on a `linspace(0, 10, n_samples)`
    # clock, independent of `samplerate` -- i.e. its physics clock advances
    # `ratio` physics-seconds per real second. The web's clock always tracks
    # real time 1:1, so matching omega = sqrt(tension - ...) requires scaling
    # tension (and hence tensionExp, in log space) by ratio**2.
    ratio = (10.0 * samplerate) / controller.n_samples
    tension_offset = 2 * math.log10(ratio)

    nodes1 = _layer_web_nodes(layer1, tension_offset, controller.control_rate, samplerate)
    nodes2 = _layer_web_nodes(layer2, tension_offset, controller.control_rate, samplerate)

    with torch.no_grad():
        # the web's tensionRouter multiplies layer 1's *pre*-gain/tanh output;
        # parallel.py's tension_modifier is layer 1's *fully processed*
        # output (post filter-mix, tanh(gain), output_gain). Scaling by
        # layer1's own output_gain roughly compensates for that extra
        # squashing stage so the ported FM depth is in the right ballpark.
        fm_scale = layer1.output_gain.reshape(-1)[0].item()
        # layer 1's own self-routing (see _layer_web_nodes and main.js's
        # forceRouter1) -- previously had no destination slot and was only
        # approximated via its diagonal into excitationGain; now exported in
        # full, same as layer 2's.
        force_router1 = layer1.force_router.reshape(n, n).cpu().numpy()
        force_router2 = layer2.force_router.reshape(n, n).cpu().numpy()
        tension_router = layer2.tension_router.reshape(n, n).cpu().numpy() * fm_scale

    header = struct.pack(
        '<BBBBBBBBBBB',
        SHARE_VERSION_WEB,
        n, n,
        0,  # strategy: 'random' -- only affects future Randomize clicks, not this instrument
        SHARE_EXCITATION_MODE_IDS.index('sustained'),
        255,  # reverb file index: none selected -- see docstring above
        _share_quantize_u8(0.5, 0, 0.95),  # sparsity: unused once an explicit instrument is loaded
        _share_quantize_u8(100, 20, 400),  # pulseRate: unused unless excitationMode='pulse'
        _share_quantize_u8(0.6, 0, 10),  # volume: a reasonable default master gain
        _share_quantize_log_u8(1.0, 0.1, 10),  # modDepth: neutral default
        _share_quantize_u8(0.0, 0, 1),  # reverbMix: harmless with no reverb file selected
    )

    body = bytearray(header)
    for node in nodes1 + nodes2:
        body += _share_encode_node(**node)

    n_clipped = 0

    def encode_router_row(row) -> bytes:
        nonlocal n_clipped
        packed = bytearray()
        for v in row:
            if abs(v) > SHARE_ROUTER_QUANT_RANGE:
                n_clipped += 1
            packed += struct.pack('<h', _share_quantize_i16(v, SHARE_ROUTER_QUANT_RANGE))
        return packed

    for row in tension_router:
        body += encode_router_row(row)
    for row in force_router2:
        body += encode_router_row(row)
    for row in force_router1:
        body += encode_router_row(row)

    if n_clipped:
        print(f'[share] warning: {n_clipped} routing weight(s) exceeded the quantization '
              f'range (+/-{SHARE_ROUTER_QUANT_RANGE}) and were clipped')

    encoded = base64.urlsafe_b64encode(bytes(body)).decode('ascii').rstrip('=')
    return f'?p={encoded}'


def _copy_to_clipboard(text: str) -> bool:
    """
    Best-effort clipboard copy via a hidden Tk window -- no extra dependency
    beyond the stdlib, since neither pyperclip nor xclip/xsel is installed
    here. Silently (but audibly, via a printed warning) does nothing useful
    if no display is available, e.g. a headless/SSH training box -- the
    console printout in training_loop_hook is always the reliable fallback.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text)
        root.update()
        root.destroy()
        return True
    except Exception as e:
        print(f'[share] could not copy to clipboard: {e}')
        return False


SHARE_HTML_PATH = 'parallel.html'
SHARE_HTML_DEFAULT_BASE_URL = 'https://blog.cochlea.xyz/dho-synth'


def _write_share_html(query: str, iteration: int, path: str = SHARE_HTML_PATH) -> None:
    """
    (Over)writes a tiny standalone HTML page with a clickable link to the web
    demo loaded with `query` (a `?p=...` string from encode_share_state_url).
    The base URL is editable via a text input; the link's displayed text is
    an abbreviated form of the full URL, with a QR code (via the
    qrcode-generator CDN script) for opening it on a phone.
    """
    import json
    page = f'''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>parallel share link</title>
<style>
  body {{ font-family: sans-serif; margin: 2em; }}
  input {{ width: 40em; max-width: 100%; font-family: monospace; }}
  a {{ font-family: monospace; }}
  #qr svg {{ width: 320px; max-width: 100%; height: auto; }}
</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/qrcode-generator/1.4.4/qrcode.min.js"></script>
</head>
<body>
<label>Base URL <input id="base" type="text" value="{SHARE_HTML_DEFAULT_BASE_URL}"></label>
<p>Iteration {iteration}: <a id="link" href="#" target="_blank" rel="noopener"></a></p>
<div id="qr"></div>
<script>
  const query = {json.dumps(query)};
  const base = document.getElementById('base');
  const link = document.getElementById('link');
  function abbreviate(url) {{
    const i = url.indexOf('?p=');
    const head = url.slice(0, i + 3);
    const p = url.slice(i + 3);
    return p.length <= 24 ? url : head + p.slice(0, 12) + '\\u2026' + p.slice(-8);
  }}
  function update() {{
    const url = base.value.trim() + query;
    link.href = url;
    link.textContent = abbreviate(url);
    link.title = url;
    const qrEl = document.getElementById('qr');
    try {{
      // type 0 = smallest QR version that fits; 'L' = max data capacity
      const qr = qrcode(0, 'L');
      qr.addData(url);
      qr.make();
      qrEl.innerHTML = qr.createSvgTag({{ cellSize: 4, margin: 4, scalable: true }});
    }} catch (e) {{
      qrEl.textContent = 'QR code unavailable: ' + e;
    }}
  }}
  base.addEventListener('input', update);
  update();
</script>
</body>
</html>
'''
    with open(path, 'w') as f:
        f.write(page)


# how often (in iterations) overfit_osc emits a share link for the
# instrument it's currently fitting; see encode_share_state_url
SHARE_EXPORT_EVERY = 5000


def signed_control_image(x: torch.Tensor) -> torch.Tensor:
    """
    A signed (batch, nodes, frames) control signal (damp_mod, tension_mod)
    as a [0, 1] image for the Spectrogram logger, which expects magnitudes:
    (batch, 2 * nodes, frames), the positive part in the top half of the
    rows and the negative part (flipped to positive) in the bottom half,
    node i at rows i and nodes + i.  Scaled by the largest |value|, so the
    image shows shape, not level (overfit_osc prints the level).
    """
    x = x / (x.abs().max() + 1e-12)
    return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)


def overfit_osc(n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int, share_html: bool = False):

    control_rate = 512

    controller = LayerController(
        n_layers=n_layers,
        n_nodes=n_nodes,
        n_samples=n_samples,
        control_rate=control_rate,
        n_to_keep=n_to_keep
    ).to(device)

    def logger_factory(collection):

        frce, = conjure.loggers(
            ['forces'],
            SupportedContentType.Spectrogram.value,
            lambda x: x.data.cpu().numpy()[0],
            collection,
            NumpySerializer(),
            NumpyDeserializer()
        )

        # the damping and tension mods actually applied (materialized:
        # scaled and sparsified), as signed_control_image: rises in the top
        # half of the rows, falls in the bottom half
        dmod, tmod = conjure.loggers(
            ['dampmod', 'tensionmod'],
            SupportedContentType.Spectrogram.value,
            lambda x: x.data.cpu().numpy()[0],
            collection,
            NumpySerializer(),
            NumpyDeserializer()
        )

        rnd,  = conjure.loggers(
            ['random'],
            SupportedContentType.Audio.value,
            encode_audio,
            collection,
            IdentitySerializer(),
            IdentityDeserializer())

        return [frce, dmod, tmod, rnd]

    def training_loop_hook(
            iteration: int,
            loggers: List[conjure.Conjure],
            model: LayerController,
            optim: Optimizer,
            target: torch.Tensor):

        # params_count = count_parameters(model)
        print('N PARAMS', model.total_params)

        # if iteration == 5000:
        #     print('Changing learning rate')
        #     for g in optim.param_groups:
        #         g['lr'] = LEARNING_RATE

        # in logger_factory's order
        frce_logger, dmod_logger, tmod_logger, rnd_logger = loggers

        with torch.no_grad():
            # highly-sparse (4-16 events) random control signal, to sanity
            # check that the learned instrument sounds plausible when driven
            # by something other than its own optimized forces
            rnd, _ = model.random_forward(n_to_keep=randint(4, 16))
            rnd_logger(max_norm(rnd))
            f = model.materialize_forces(do_upsample=False)
            f = f / (f.max() + 1e-12)
            frce_logger(f)

            dm = model.materialize_damping_mod()
            tm = model.materialize_tension_mod()
            dmod_logger(signed_control_image(dm))
            tmod_logger(signed_control_image(tm))
            # the images are normalized, so print the levels: damp_mod is
            # added to each node's per-frame damping multiplier, tension_mod
            # to its log10 tension (0.1 is ~a whole tone)
            print(
                f'damp_mod max |.| {dm.abs().max().item():.3g} ({int((dm != 0).sum())} nonzero), '
                f'tension_mod max |.| {tm.abs().max().item():.3g} ({int((tm != 0).sum())} nonzero)')

        if iteration > 0 and iteration % SHARE_EXPORT_EVERY == 0:
            try:
                query = encode_share_state_url(model)
                print(f'[share] iteration {iteration}: {query}')
                _copy_to_clipboard(query)
                if share_html:
                    _write_share_html(query, iteration)
            except Exception as e:
                print(f'[share] failed to export share link at iteration {iteration}: {e}')

    def model_eval(model: LayerController, _, target: torch.Tensor):

        recon, control_signal = model.forward()

        # if we are "hard" sparsifying, then don't bother with sparsity loss
        sparsity_coeff = 0 if SPARSIFY else 1
        info_loss = information_loss(target, recon)
        print('info', info_loss.item())
        loss = loss_func(target, recon) \
            + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff) \
            + (info_loss * INFORMATION_COEFF)
        return recon, loss

    overfit_model(
        n_samples=n_samples,
        model=controller,
        loss_func=loss_func,
        model_eval=model_eval,
        collection_name='parallel',
        logger_factory=logger_factory,
        training_loop_hook=training_loop_hook,
        learning_rate=LEARNING_RATE,
        port=9998
    )


@torch.jit.script
def execute_rnn(
        hidden_weights: torch.Tensor,
        input_weights: torch.Tensor,
        output_weights: torch.Tensor,
        control_signal: torch.Tensor):

    # print(hidden_weights.shape, input_weights.shape, output_weights.shape, control_signal.shape)

    batch, input_dim, output_dim = hidden_weights.shape
    batch, input_dim, output_dim = output_weights.shape
    batch, input_dim, time = control_signal.shape

    output = []
    hidden = torch.zeros(batch, input_dim, 1, device=hidden_weights.device)

    for i in range(time):
        input_t = control_signal[:, :, i: i + 1]

        # map input to the hidden state
        inp = torch.einsum('bit,bio->bot', input_t, input_weights)

        # transform the previous hidden state
        h = torch.einsum('bit,bio->bot', hidden, hidden_weights)

        # new hidden state inp + hidden mapping
        hidden = torch.tanh(inp + h)

        # finally, map hidden state to the output
        o = torch.einsum('bit,bio->bot', hidden, output_weights)
        o = torch.sin(o)

        output.append(o)

    output = torch.cat(output, dim=-1)
    # return output
    return output.permute(0, 2, 1).reshape(batch, 1, -1)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'overfit', 'overfit_direct', 'self_supervised', 'self_supervised_only', 'test'], required=True)
    parser.add_argument(
        '--no-reverb', action='store_true',
        help="drop the reverb's wet branch (room impulse response convolution) at the end of the "
             "chain, keeping only the dry signal at its learned dry gain, in every mode (see USE_REVERB)")
    parser.add_argument(
        '--integrator', choices=['control_rate', 'sample_rate'], default=INTEGRATOR,
        help='where the oscillators integrate their excitation (see INTEGRATOR)')
    parser.add_argument(
        '--share-html', action='store_true',
        help=f'overfit_direct only: (over)write {SHARE_HTML_PATH} with a clickable share link each time one is exported')

    args = parser.parse_args()

    # module scope, so this rebinds the global every function reads
    USE_REVERB = not args.no_reverb
    INTEGRATOR = args.integrator

    n_to_keep = N_TO_KEEP

    if args.mode == 'train':
        train_ae(
            batch_size=2,
            n_nodes=N_NODES,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=N_TO_KEEP)
    elif args.mode == 'self_supervised':
        self_supervised(
            batch_size=2,
            n_nodes=N_NODES,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=N_TO_KEEP)
    elif args.mode == 'self_supervised_only':
        self_supervised_only(
            batch_size=2,
            n_nodes=N_NODES,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=N_TO_KEEP)
    elif args.mode == 'overfit':
        overfit_autoencoder(
            n_nodes=N_NODES,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=N_TO_KEEP)
    elif args.mode == 'overfit_direct':
        # no encoder/hypernetwork -- directly optimizes the instrument
        # parameters (and the control signal itself) that the hypernetwork
        # path would otherwise generate, as a sanity check isolating whether
        # difficulty comes from the physics parameterization itself or from
        # the encoder/hypernetwork on top of it
        overfit_osc(
            n_nodes=N_NODES,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=N_TO_KEEP,
            share_html=args.share_html)
    elif args.mode == 'test':
        batch = 3
        dim = 64
        time_dim = 1024

        control_signal = torch.zeros(3, dim, time_dim)
        input_weights = torch.zeros(3, dim, dim)
        hidden_weights = torch.zeros(3, dim, dim)
        output_weights = torch.zeros(3, dim, dim)
        o = execute_rnn(
            hidden_weights, input_weights, output_weights, control_signal)
        print(o.shape, dim*dim)
