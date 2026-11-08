import math
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

# if True, use UNetAnalysis instead of Analysis as the encoder: the
# instrument latent comes from the U-Net's pooled bottleneck (replacing the
# separate latents/latents_attn attention-pooling heads), and cs/tension_mod/
# damp_mod are still derived (via the existing ControlSignalCreator heads)
# from the U-Net's final, full-size decoder output instead of Analysis's
# AntiCausalAnalysis output -- otherwise a drop-in replacement.
USE_UNET_ENCODER = True

N_NODES = 32

# analysis arch
DO_NORM = True


# sparse
SPARSIFY = False
SOFT_SPARSE = False
SPARSITY_LOSS_COEFF = 1
N_TO_KEEP = 128

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
    noise_mix: torch.Tensor
    gains: torch.Tensor
    output_gain: torch.Tensor

    def display_shapes(self):
        print(f'''
            {self.mass.shape}
            {self.tension.shape}
            {self.filters.shape}
            {self.filters_mix.shape}
            {self.force_router.shape}
            {self.tension_router.shape}
            {self.damping.shape}
            {self.noise_mix.shape}
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
    Alternate encoder to Analysis (see USE_UNET_ENCODER): a 1D U-Net over
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

        self.noise_mix = ParameterGenerator(latent_dim, n_nodes * 2)

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

        nm = self.noise_mix(latent).view(batch, self.n_nodes, 2)

        return InstrumentDefinitionTensors(
            mass=mass,
            tension=t,
            damping=d,
            filters=filt,
            filters_mix=mx,
            force_router=fr,
            tension_router=tr,
            noise_mix=nm,
            gains=g,
            output_gain=og
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

        x = torch.abs(x)
        x = x - x.mean(dim=(1, 2), keepdim=True)
        x = torch.relu(x)
        
        
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

        if USE_UNET_ENCODER:
            self.analysis = UNetAnalysis(n_samples, control_rate, channels)
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

        if not USE_UNET_ENCODER:
            # with UNetAnalysis, the pooled bottleneck is the instrument
            # latent directly -- no separate attention-pooling heads needed
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

        # used only when SUM_LAYER_OUTPUTS is True, to scale/mix each layer's
        # own output into the final signal, instead of using only the last
        # layer's output
        self.layer_mix = nn.Parameter(torch.zeros(n_layers).uniform_(-0.01, 0.01))

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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if USE_UNET_ENCODER:
            latents, x = self.analysis(x)
        else:
            x = self.analysis(x)

            l = self.latents(x)
            attn = self.latents_attn(x)

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

        for layer in self.hyper_networks:

            params: InstrumentDefinitionTensors = layer.forward(latents)
            diff = 1 - self.hyper_params.base_resonance
            params.damping = self.hyper_params.base_resonance + (torch.sigmoid(params.damping) * diff * 0.9998)
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

        rooms = torch.relu(self.room_mix.forward(latents))
        mx = self.wet_dry.forward(latents)

        wet = self.verb.forward(tm, rooms)

        stacked = torch.stack([tm, wet], dim=-1)

        mixed = torch.einsum('bctm,bm->bct', stacked, mx)

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
        if USE_UNET_ENCODER:
            latents, x = self.analysis(x)
        else:
            x = self.analysis(x)
            l = self.latents(x)
            attn = self.latents_attn(x)
            latents = torch.mean(l * attn, dim=-1)

        cs = self.control(x)
        t = self.tension(x) * TENSION_MOD_SCALE
        d = self.damp(x) * DAMP_MOD_SCALE

        params_list = []
        for layer in self.hyper_networks:
            params: InstrumentDefinitionTensors = layer.forward(latents)
            diff = 1 - self.hyper_params.base_resonance
            params.damping = self.hyper_params.base_resonance + (torch.sigmoid(params.damping) * diff * 0.9998)
            params_list.append(params)

        rooms = torch.relu(self.room_mix.forward(latents))
        mx = self.wet_dry.forward(latents)

        return cs, t, d, params_list, rooms, mx


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

    x = damped_harmonic_oscillator(
        energy=energy,
        time=static.time,
        mass=mass,
        damping=static.damping,
        tension=tension,
        initial_displacement=static.initial_displacement,
        clamp_to_nyquist=CLAMP_OMEGA_TO_NYQUIST,
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
def damped_harmonic_oscillator(
        energy: torch.Tensor,
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor,
        clamp_to_nyquist: bool
) -> torch.Tensor:
    x = (damping / (2 * mass))

    omega = torch.sqrt(torch.abs(tension - (x ** 2)))

    if clamp_to_nyquist:
        dt = time[1] - time[0]
        nyquist = 0.95 * math.pi / dt
        omega = torch.minimum(omega, nyquist)

    phi = torch.atan2(
        (x * initial_displacement),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    # z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    z = a * energy * torch.cos(omega * time - phi)
    return z


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

        self.noise_mix = nn.Parameter(
            torch.zeros(1, self.n_nodes, 2).uniform_(-1, 1))

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
            tension_mod: torch.Tensor = None) -> torch.Tensor:

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
            noise_mix=self.noise_mix,
            gains=self.gain,
            output_gain=self.output_gain
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

        self.mix = nn.Parameter(torch.zeros(
            self.n_layers).uniform_(-0.01, 0.01))

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

        tm = None
        outputs = []
        for i, layer in enumerate(self.layers):

            tm = layer.forward(
                forces=sparse_forces,
                tension_modifier=tm,
                damp_mod=dm,
                tension_mod=tension_mod
            )
            outputs.append(tm)

        # DECISION: Does the conv reverb cause issues because it's outside the energy calculation?
        # apply conv reverb
        wet = self.verb.forward(tm, self.room_mix)

        # wet/dry reverb mix
        x = torch.stack([tm, wet], dim=-1)
        x = x * self.wet_dry_mix
        tm = torch.sum(x, dim=-1)

        if sum_output:
            tm = torch.sum(tm, dim=1, keepdim=True)

        return tm, sparse_forces

# spiking = SpikingModel(64, 64, 64, 64, 64).to(device)

def loss_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    
    # base_loss = spiking.compute_multiband_loss(a, b, hard=True)

    a = flattened_multiband_spectrogram(
        a, {'xs': (64, 16)}, smallest_band_size=512)
    b = flattened_multiband_spectrogram(
        b, {'xs': (64, 16)}, smallest_band_size=512)

    # a = stft(a, 2048, 256, pad=False)
    # b = stft(b, 2048, 256, pad=False)

    base_loss = torch.abs(a - b).sum()
    return base_loss


def random_synthetic_instrument(
        batch_size: int,
        n_nodes: int,
        filter_size: int,
        base_resonance: float,
        device,
        n_filter_taps: int = N_FILTER_TAPS) -> InstrumentDefinitionTensors:
    """
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

    noise_mix = torch.zeros(batch_size, n_nodes, 2, device=device).uniform_(-1, 1)
    gains = torch.zeros(batch_size, n_nodes, device=device).uniform_(0.5, 1.5)
    output_gain = torch.zeros(batch_size, 1, device=device).uniform_(0.5, 1.5)

    return InstrumentDefinitionTensors(
        mass=mass,
        tension=tension,
        filters=filters,
        filters_mix=filters_mix,
        force_router=force_router,
        tension_router=tension_router,
        damping=damping,
        noise_mix=noise_mix,
        gains=gains,
        output_gain=output_gain
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

    filters/filters_mix/noise_mix are left out of this loss for now: they're
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
        + torch.abs(known_rooms - est_rooms).sum()
        + torch.abs(known_wet_dry - est_wet_dry).sum()
    )

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
        loss = loss_func(batch, recon) + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff)

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

        print(i, loss.item(), sparsity.item())


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
        real_loss = loss_func(batch, recon) + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff)
        sparsity = (control_signal > 0).sum() / control_signal.numel()

        # --- synthetic example: known params/control signals -> known audio ---
        known_cs, known_tension_mod, known_damp_mod = random_synthetic_control_signals(
            batch_size=batch.shape[0], n_nodes=n_nodes, n_frames=n_frames,
            device=batch.device)
        known_params = [
            random_synthetic_instrument(
                batch_size=batch.shape[0], n_nodes=n_nodes, filter_size=filter_size,
                base_resonance=base_resonance, device=batch.device)
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
        synth_recon_loss = loss_func(synthetic_audio, synth_recon)

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
            'sparsity', sparsity.item())


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
        known_params = [
            random_synthetic_instrument(
                batch_size=batch_size, n_nodes=n_nodes, filter_size=filter_size,
                base_resonance=base_resonance, device=device)
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
        synth_recon_loss = loss_func(synthetic_audio, synth_recon)

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
            'param', param_loss.item())

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

    def model_eval(model: InstrumentAutoencoder, _, target: torch.Tensor):
        control_signal, recon = model.forward(target)

        sparsity = (control_signal > 0).sum() / control_signal.numel()
        print(sparsity.item())

        # if we are "hard" sparsifying, then don't bother with sparsity loss        
        sparsity_coeff = 0 if SPARSIFY else 1
        loss = loss_func(target, recon) + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff)
        return recon, loss

    overfit_model(
        n_samples=n_samples,
        model=analysis_model,
        loss_func=loss_func,
        model_eval=model_eval,
        collection_name='parallel',
        learning_rate=LEARNING_RATE,
        port=9998
    )


def overfit_osc(n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):

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

        rnd,  = conjure.loggers(
            ['random'],
            SupportedContentType.Audio.value,
            encode_audio,
            collection,
            IdentitySerializer(),
            IdentityDeserializer())

        return [frce, rnd]

    def training_loop_hook(
            iteration: int,
            loggers: List[conjure.Conjure],
            model: LayerController,
            optim: Optimizer):

        # params_count = count_parameters(model)
        print('N PARAMS', model.total_params)

        # if iteration == 5000:
        #     print('Changing learning rate')
        #     for g in optim.param_groups:
        #         g['lr'] = LEARNING_RATE

        # TODO: index loggers by name
        rnd_logger = loggers[-1]
        frce_logger = loggers[-2]

        with torch.no_grad():
            rnd, _ = model.random_forward(n_to_keep=4)
            rnd_logger(max_norm(rnd))
            f = model.materialize_forces(do_upsample=False)
            f = f / (f.max() + 1e-12)
            frce_logger(f)

    def model_eval(model: LayerController, _, target: torch.Tensor):

        recon, control_signal = model.forward()
        
        # if we are "hard" sparsifying, then don't bother with sparsity loss
        sparsity_coeff = 0 if SPARSIFY else 1
        loss = loss_func(target, recon) + (l0_norm(control_signal) * SPARSITY_LOSS_COEFF * sparsity_coeff)
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

    args = parser.parse_args()

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
            n_to_keep=N_TO_KEEP)
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
