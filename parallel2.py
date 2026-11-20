from dataclasses import dataclass
from typing import Tuple, Callable, Iterable

from conjure import LmdbCollection, loggers, serve_conjure
from torch import nn

import torch
from torch.nn import functional as F

from conjure.logger import encode_audio
from torch.optim import Adam
from config.dotenv import Config
from data.audioiter import AudioIterator
from modules import sparsify, interpolate_last_axis, max_norm
from modules.anticausal import AntiCausalAnalysis
from modules.multibanddict import flattened_multiband_spectrogram
from modules.normalization import unit_norm
from modules.reverb import NeuralReverb
from modules.transfer import fft_convolve
from modules.upsample import ensure_last_axis_length
from util import device
from util.overfit import overfit_model
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.nn.utils.weight_norm import weight_norm
from argparse import ArgumentParser

from util.weight_init import make_initializer

# the PyTorch port of jax_scale_attention.AttentionEncoder lives in
# parallel.py; shared rather than copied, so the two stay in step
from parallel import TransformerAnalysis
from sample_rate_dho import linear_recurrence

Solution = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

MASS_COEFF = 1
BASE_RESONANCE = 0.01
FORWARD_FORCE = False
DO_NORM = False
CHANNELS = 256
SPARSIFY = True
SOFT_SPARSE = False

# which encoder hears the audio, as in parallel.py:
#   'conv'        -- Analysis: a dilated AntiCausalAnalysis conv stack
#   'transformer' -- parallel.TransformerAnalysis: self-attention whose only
#                     normalization is ScaleAwareNorm (self-conditioned
#                     AdaLN) and QK-norm attention; see jax_scale_attention.py
# either way, the instrument latent is pooled from its per-frame output by
# the latents/latents_attn heads, and the control signals are derived from it
ENCODER_ARCH = 'conv'

TRANSFORMER_N_LAYERS = 6
TRANSFORMER_N_HEADS = 4
# True: each frame sees itself and what follows (like Analysis); False: full,
# unmasked attention
TRANSFORMER_ANTICAUSAL = True
# 'relative' (absolute level enters nowhere but the returned log level --
# unused here -- so features are invariant to a global gain) or 'absolute'
# (also conditions on each frame's absolute level, giving up that invariance)
TRANSFORMER_LEVEL_MODE = 'relative'

# if True, unit_norm the attention-pooling weights (latents_attn's output)
# over time before pooling the instrument latent, so the latent's scale
# doesn't grow or shrink with however large that raw output happens to be
USE_ATTENTION_NORM = True

# raw tension t enters the oscillator as 10 ** t (see compute_tension).
# time runs 0 -> 10 over the whole segment (2 ** 17 samples, ~5.9s), so
# t in (4, 9) puts sqrt(tension) at ~27 Hz to ~8.5 kHz.  Same range as
# parallel.py's
TENSION_EXPONENT_RANGE = (4.0, 9.0)

# Start each hypernetwork head (and the direct mode's parameters) at an
# audible instrument instead of at ~0: see InstrumentHyperNetwork.
# init_operating_point.  At init the latent is tiny, so each head outputs
# ~its bias, and init_func zeroes the biases -- mass ~0 (the drive divided
# by ~0), gains and the attack envelope ~0 (no sound), tension 10 ** 0 = 1
# (well below audible)
INIT_OPERATING_POINT = True


def compute_tension(t: torch.Tensor) -> torch.Tensor:
    return 10**t


def compute_mass(m_raw: torch.Tensor) -> torch.Tensor:
    """
    As parallel.py's: positive, and bounded away from 0.  (Was `t * m`, the
    product of two raw head outputs, which is ~1e-10 at init and can be any
    sign; the drive is divided by mass, so that overflowed.)
    """
    return 1e-5 + torch.abs(m_raw)


def compute_damping(d_raw: torch.Tensor, base_resonance: float) -> torch.Tensor:
    """per-sample decay factor in (base_resonance, 1)"""
    return base_resonance + torch.clamp(d_raw, 0, 1) * (1 - base_resonance)


def control_signal(x: torch.Tensor, do_sparsify: bool, n_to_keep: int) -> torch.Tensor:
    """
    The non-negative (optionally sparse) control signal ControlSignalCreator
    derives from its conv's output; shared with the direct mode's
    LayerController, which applies it to directly-optimized tensors
    """
    # x = torch.abs(x)
    # x = x - x.mean()
    x = torch.relu(x)
    if do_sparsify:
        x = sparsify(x, n_to_keep=n_to_keep)
    return x


@dataclass
class HyperParameters:
    n_samples: int
    n_frames: int
    control_rate: int
    attack_size: int
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
    attack: torch.Tensor
    force_router: torch.Tensor
    tension_router: torch.Tensor
    damping: torch.Tensor
    gains: torch.Tensor

    def display_shapes(self):
        print(f'''
            {self.mass.shape}
            {self.tension.shape}
            {self.filters.shape}
            {self.filters_mix.shape}
            {self.attack.shape}
            {self.force_router.shape}
            {self.tension_router.shape}
            {self.damping.shape}
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


class InstrumentHyperNetwork(nn.Module):
    def __init__(
            self,
            latent_dim: int,
            n_nodes: int,
            filter_size: int,
            attack_size: int,
            attack_control_frames: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_nodes = n_nodes
        self.router_size = self.n_nodes ** 2
        self.filter_size = filter_size
        self.attack_size = attack_size
        self.attack_control_frames = attack_control_frames

        self.to_masses = ParameterGenerator(latent_dim, n_nodes)
        self.to_tension = ParameterGenerator(latent_dim, n_nodes)
        self.damping = ParameterGenerator(latent_dim, n_nodes)

        self.gain = ParameterGenerator(latent_dim, n_nodes)

        self.filters = ParameterGenerator(latent_dim, n_nodes * filter_size)
        self.filters_mix = ParameterGenerator(latent_dim, n_nodes * 2)

        # learnable per-node "attack" envelope: shapes the noise burst that
        # drives each node, kept separate from the filters above (which
        # color the *resonant* output) and from mass/damping (which govern
        # decay), so the attack transient and the resonant character can be
        # learned independently.
        #
        # generated at a coarse, sub-audio-rate resolution
        # (attack_control_frames raw points) and smoothly upsampled to
        # attack_size later (see generate_excitation) -- a handful of
        # linearly-interpolated points cannot encode tonal/periodic content,
        # which keeps this envelope's job strictly to "attack shape", not
        # pitch.
        self.attack = ParameterGenerator(latent_dim, n_nodes * attack_control_frames)

        self.force_router = ParameterGenerator(latent_dim, self.router_size)
        self.tension_router = ParameterGenerator(latent_dim, self.router_size)

    @torch.no_grad()
    def init_operating_point(self):
        """
        Set each head's bias (its output, at init) to the instrument the
        direct mode's Layer starts from:
          - mass: |m_raw| ~ U(0, 1)
          - tension: t_raw ~ U(TENSION_EXPONENT_RANGE)
          - gains: U(0.5, 1.5)
          - attack: an envelope decaying linearly from 1 to 0
          - force_router, tension_router: identity
          - filters_mix: dry 1, filtered 0
        damping and the filters keep their zero bias (damping halfway
        between base_resonance and 1; filters are unit-normed anyway).
        """
        n = self.n_nodes
        device = self.gain.ln1.bias.device
        eye = torch.eye(n, device=device).flatten()
        self.to_masses.ln1.bias.uniform_(-1, 1)
        self.to_tension.ln1.bias.uniform_(*TENSION_EXPONENT_RANGE)
        self.gain.ln1.bias.uniform_(0.5, 1.5)
        self.attack.ln1.bias.view(n, self.attack_control_frames).copy_(
            torch.linspace(1, 0, self.attack_control_frames, device=device).expand(n, -1))
        self.force_router.ln1.bias.copy_(eye)
        self.tension_router.ln1.bias.copy_(eye)
        self.filters_mix.ln1.bias.view(n, 2).copy_(
            torch.tensor([1., 0.], device=device).expand(n, 2))

    def forward(self, latent: torch.Tensor) -> InstrumentDefinitionTensors:
        batch, latent_dim = latent.shape

        g = self.gain.forward(latent)

        m = self.to_masses(latent).view(batch, self.n_nodes, 1)
        t = self.to_tension(latent).view(batch, self.n_nodes, 1)
        d = self.damping(latent).view(batch, self.n_nodes, 1)

        filt = self.filters(latent).view(batch, self.n_nodes, self.filter_size)
        mx = self.filters_mix(latent).view(batch, self.n_nodes, 2)

        attack = self.attack(latent).view(batch, self.n_nodes, self.attack_control_frames)

        fr = self.force_router(latent).view(batch, self.n_nodes, self.n_nodes)
        tr = self.tension_router(latent).view(
            batch, self.n_nodes, self.n_nodes)

        return InstrumentDefinitionTensors(
            mass=compute_mass(m),
            tension=t,
            damping=d,
            filters=filt,
            filters_mix=mx,
            attack=attack,
            force_router=fr,
            tension_router=tr,
            gains=g
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
        return control_signal(x, self.sparsify, self.n_to_keep)


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
            base_resonance: float = BASE_RESONANCE,
            attack_size: int = 512,
            attack_control_frames: int = 4):

        super().__init__()
        self.n_nodes = n_nodes
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_frames = n_samples // control_rate
        self.n_layers = n_layers
        self.channels = channels
        self.n_nodes = n_nodes
        self.filter_size = filter_size
        self.attack_size = attack_size
        self.attack_control_frames = attack_control_frames
        self.n_to_keep = n_to_keep

        self.hyper_params = HyperParameters(
            n_samples=n_samples,
            n_frames=self.n_frames,
            control_rate=self.control_rate,
            attack_size=self.attack_size,
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

        if ENCODER_ARCH == 'transformer':
            self.analysis = TransformerAnalysis(
                n_samples, control_rate, channels,
                n_layers=TRANSFORMER_N_LAYERS,
                n_heads=TRANSFORMER_N_HEADS,
                anticausal=TRANSFORMER_ANTICAUSAL,
                level_mode=TRANSFORMER_LEVEL_MODE)
        else:
            self.analysis = Analysis(n_samples, control_rate, channels)

        self.hyper_networks = nn.ModuleList([
            InstrumentHyperNetwork(channels, n_nodes, filter_size, attack_size, attack_control_frames)
            for _ in range(n_layers)
        ])

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

        # init_func (small uniform weights, zero biases) everywhere but a
        # TransformerAnalysis, whose own init it would overwrite: its
        # ScaleAwareNorms' modulation layers are zero-initialized, so each
        # starts out as a plain RMSNorm.  (Child by child, in the same order
        # self.apply would visit them, so 'conv' initializes as before.)
        for child in self.children():
            if not isinstance(child, TransformerAnalysis):
                child.apply(init_func)

        if INIT_OPERATING_POINT:
            for hyper in self.hyper_networks:
                hyper.init_operating_point()
            with torch.no_grad():
                # rooms through a relu: U(0, 1); wet/dry through a softmax:
                # mostly dry
                self.room_mix.ln1.bias.uniform_(0, 1)
                self.wet_dry.ln1.bias.copy_(torch.tensor([1., 0.5]))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        cs = self.control(x)

        t = self.tension(x)
        d = self.damp(x)

        static = StaticTensors(
            time=self.t,
            damping=self.d,
            initial_displacement=self._id
        )

        mixed = render(
            hyper=self.hyper_params,
            static=static,
            layers=[layer.forward(latents) for layer in self.hyper_networks],
            forces=cs,
            damp_mod=d,
            tension_mod=t,
            verb=self.verb,
            room_mix=self.room_mix.forward(latents),
            wet_dry=self.wet_dry.forward(latents))

        return cs, mixed


def render(
        hyper: HyperParameters,
        static: StaticTensors,
        layers: Iterable[InstrumentDefinitionTensors],
        forces: torch.Tensor,
        damp_mod: torch.Tensor,
        tension_mod: torch.Tensor,
        verb: NeuralReverb,
        room_mix: torch.Tensor,
        wet_dry: torch.Tensor) -> torch.Tensor:
    """
    Everything after the encoder/hypernetwork: run the layers in series
    (each layer's output modulating the next one's tension), sum the nodes,
    and mix in reverb.  Shared by InstrumentAutoencoder and the direct
    mode's LayerController, so the two always synthesize identically.

    layers: each with *raw* damping (compute_damping is applied here)
    room_mix, wet_dry: raw, (batch, n_rooms) and (batch, 2)
    """
    tm = None
    for params in layers:
        params.damping = compute_damping(params.damping, hyper.base_resonance)
        tm, routed = execute_parallel_layer(
            instrument=params,
            hyper=hyper,
            static=static,
            forces=forces,
            damp_mod=damp_mod,
            tension_mod=tension_mod,
            tension_modifier=tm
        )
        if FORWARD_FORCE:
            forces = routed

    tm = torch.sum(tm, dim=1, keepdim=True)

    rooms = torch.relu(room_mix)
    mx = torch.softmax(wet_dry, dim=-1)

    wet = verb.forward(tm, rooms)
    stacked = torch.stack([tm, wet], dim=-1)
    return torch.einsum('bctm,bm->bct', stacked, mx)


class Layer(nn.Module):
    """
    One layer's instrument, as directly-optimized tensors: the same raw
    values an InstrumentHyperNetwork head would output, initialized to its
    init_operating_point (plus a little noise where that's constant, to
    break symmetry between nodes)
    """
    def __init__(self, n_nodes: int, filter_size: int, attack_control_frames: int):
        super().__init__()
        eye = torch.eye(n_nodes)[None]
        self.m_raw = nn.Parameter(torch.zeros(1, n_nodes, 1).uniform_(-1, 1))
        self.t_raw = nn.Parameter(torch.zeros(1, n_nodes, 1).uniform_(*TENSION_EXPONENT_RANGE))
        self.d_raw = nn.Parameter(torch.zeros(1, n_nodes, 1).uniform_(-1, 1))
        self.gains = nn.Parameter(torch.zeros(1, n_nodes).uniform_(0.5, 1.5))
        self.filters = nn.Parameter(torch.zeros(1, n_nodes, filter_size).uniform_(-1, 1))
        self.filters_mix = nn.Parameter(
            torch.tensor([1., 0.]).repeat(1, n_nodes, 1) + torch.zeros(1, n_nodes, 2).uniform_(-0.01, 0.01))
        self.attack = nn.Parameter(
            torch.linspace(1, 0, attack_control_frames).repeat(1, n_nodes, 1)
            + torch.zeros(1, n_nodes, attack_control_frames).uniform_(-0.01, 0.01))
        self.force_router = nn.Parameter(eye + torch.zeros(1, n_nodes, n_nodes).uniform_(-0.01, 0.01))
        self.tension_router = nn.Parameter(eye + torch.zeros(1, n_nodes, n_nodes).uniform_(-0.01, 0.01))

    def forward(self) -> InstrumentDefinitionTensors:
        return InstrumentDefinitionTensors(
            mass=compute_mass(self.m_raw),
            tension=self.t_raw,
            damping=self.d_raw,
            filters=self.filters,
            filters_mix=self.filters_mix,
            attack=self.attack,
            force_router=self.force_router,
            tension_router=self.tension_router,
            gains=self.gains)


class LayerController(nn.Module):
    """
    InstrumentAutoencoder with the encoder and hypernetworks removed: the
    instrument (one Layer per layer), its control signals (forces and the
    damping/tension modulations), and the reverb mix are all optimized
    directly, and synthesized by the same `render`.  If this fits a segment
    and the autoencoder doesn't, the trouble is in the encoder/hypernetwork,
    not the physics.

    The control signals go through control_signal, as ControlSignalCreator's
    outputs do (forces sparsified iff SPARSIFY; the modulations always, to 16)
    """
    def __init__(
            self,
            n_samples: int,
            n_nodes: int,
            control_rate: int,
            n_layers: int,
            filter_size: int,
            n_to_keep: int,
            base_resonance: float = 0.5,
            attack_size: int = 512,
            attack_control_frames: int = 4):
        super().__init__()
        self.n_samples = n_samples
        self.n_nodes = n_nodes
        self.n_frames = n_samples // control_rate
        self.n_to_keep = n_to_keep

        self.hyper_params = HyperParameters(
            n_samples=n_samples,
            n_frames=self.n_frames,
            control_rate=control_rate,
            attack_size=attack_size,
            base_resonance=base_resonance,
            mass_coeff=MASS_COEFF
        )

        self.register_buffer('d', torch.zeros(1, n_nodes, 1).fill_(1))
        self.register_buffer('_id', torch.zeros(1, n_nodes, 1).fill_(1))
        self.register_buffer('t', torch.linspace(0, 10, n_samples))

        self.layers = nn.ModuleList([
            Layer(n_nodes, filter_size, attack_control_frames) for _ in range(n_layers)])

        # raw control signals: control_signal keeps |x| - mean(|x|), so
        # roughly the larger half of the frames start active.  U(-0.01, 0.01)
        # puts the output at about the autoencoder's level at init (rms
        # ~0.02 vs ~0.03); U(-1, 1) saturates every node's tanh
        self.forces = nn.Parameter(torch.zeros(1, n_nodes, self.n_frames).uniform_(-0.01, 0.01))
        # the modulations add directly to damping and to the tension
        # exponent, so start them small
        self.damp_mod = nn.Parameter(torch.zeros(1, n_nodes, self.n_frames).uniform_(-0.01, 0.01))
        self.tension_mod = nn.Parameter(torch.zeros(1, n_nodes, self.n_frames).uniform_(-0.01, 0.01))

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate=22050, n_samples=n_samples)
        self.room_mix = nn.Parameter(torch.zeros(1, self.verb.n_rooms).uniform_(0, 1))
        self.wet_dry = nn.Parameter(torch.tensor([[1., 0.5]]))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cs = control_signal(self.forces, SPARSIFY, self.n_to_keep)
        d = control_signal(self.damp_mod, True, 16)
        t = control_signal(self.tension_mod, True, 16)

        static = StaticTensors(
            time=self.t,
            damping=self.d,
            initial_displacement=self._id
        )

        mixed = render(
            hyper=self.hyper_params,
            static=static,
            layers=[layer.forward() for layer in self.layers],
            forces=cs,
            damp_mod=d,
            tension_mod=t,
            verb=self.verb,
            room_mix=self.room_mix,
            wet_dry=self.wet_dry)

        return cs, mixed


def zero_stuff_upsample(x: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Places each control-rate sample at the start of a `factor`-sample window
    of zeros -- a "with holes" upsample -- instead of interpolating between
    control-rate samples. This keeps sparse, impulse-like control events
    sparse and impulse-like at audio rate, rather than smearing them into
    ramps the way linear interpolation would.
    """
    batch, channels, frames = x.shape
    out = torch.zeros(batch, channels, frames, factor, device=x.device, dtype=x.dtype)
    out[..., 0] = x
    return out.reshape(batch, channels, frames * factor)


def generate_excitation(
        forces: torch.Tensor,
        attack: torch.Tensor,
        control_rate: int,
        attack_size: int,
        n_samples: int) -> torch.Tensor:
    """
    k-rate sparse control signal -> zero-stuffed upsample to audio rate ->
    convolved with a learnable per-node attack envelope -> multiplied by
    independent per-node white noise. This produces an audio-rate excitation
    signal whose attack character comes entirely from the learned envelope,
    keeping the resonance/decay character (mass, damping) a separate,
    interpretable concern handled downstream by the leaky integrator.

    `attack` arrives at a coarse, sub-audio-rate resolution (a handful of
    control points per node, see InstrumentHyperNetwork) and is smoothly
    (linearly) upsampled to `attack_size` here, *before* it ever touches the
    noise. A piecewise-linear curve through a few points cannot encode
    tonal/periodic content, so this envelope can only shape amplitude over
    time, never impose pitch -- unlike generating it directly at audio-rate
    resolution, which would let the network learn an arbitrary (possibly
    tonal) kernel.
    """
    impulses = zero_stuff_upsample(forces, control_rate)
    impulses = ensure_last_axis_length(impulses, desired_size=n_samples)

    envelope = torch.abs(attack)
    envelope = interpolate_last_axis(envelope, desired_size=attack_size)
    envelope = ensure_last_axis_length(envelope, desired_size=n_samples)
    shaped = fft_convolve(impulses, envelope)

    noise = torch.zeros_like(shaped).uniform_(-1, 1)
    return shaped * noise


def execute_parallel_layer(
        hyper: HyperParameters,
        static: StaticTensors,
        instrument: InstrumentDefinitionTensors,
        forces: torch.Tensor,
        tension_modifier: torch.Tensor = None,
        damp_mod: torch.Tensor = None,
        tension_mod: torch.Tensor = None) -> [torch.Tensor, torch.Tensor]:

    forces = torch.einsum('bct,bcd->bdt', forces, instrument.force_router)

    d = instrument.damping.repeat(1, 1, hyper.n_frames)
    if damp_mod is not None:
        d = d + damp_mod

    damp = torch.abs(d)
    damp = interpolate_last_axis(damp, desired_size=hyper.n_samples)

    mass = instrument.mass * hyper.mass_coeff

    excitation = generate_excitation(
        forces, instrument.attack, hyper.control_rate, hyper.attack_size, hyper.n_samples)

    # audio-rate leaky integrator/damping via the chunked parallel scan,
    # instead of running the recurrence at control rate and interpolating
    # the result up to audio rate afterward
    energy = chunked_parallel(excitation / mass, damp)

    tension = instrument.tension

    if tension_modifier is not None:
        tension_modifier = torch.einsum(
            'bct,bcd->bdt', tension_modifier, instrument.tension_router)
        tension = instrument.tension + tension_modifier

    if tension_mod is not None:
        tm = interpolate_last_axis(tension_mod, desired_size=hyper.n_samples)
        tension = tension + tm

    x = damped_harmonic_oscillator(
        energy=energy,
        time=static.time,
        mass=mass,
        damping=static.damping,
        tension=compute_tension(tension),
        initial_displacement=static.initial_displacement,
    )

    filt = fft_convolve(x, ensure_last_axis_length(
        unit_norm(instrument.filters), desired_size=hyper.n_samples))
    x = torch.stack([x, filt], dim=-1)

    x = torch.einsum('bctm,bcm->bct', x, instrument.filters_mix)

    x = torch.tanh(x * instrument.gains[..., None])

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
        initial_displacement: torch.Tensor
) -> torch.Tensor:
    x = (damping / (2 * mass))

    omega = torch.sqrt(torch.abs(tension - (x ** 2)))

    phi = torch.atan2(
        (x * initial_displacement),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    z = a * energy * torch.cos(omega * time - phi)
    return z


def chunked_parallel(
        forces: torch.Tensor,
        damping: torch.Tensor) -> torch.Tensor:
    """
    o[i] = (f[i] + o[i-1]) * d[i], i.e. o[i] = d[i] o[i-1] + d[i] f[i], via
    sample_rate_dho.linear_recurrence.

    This used to be a chunked cumprod/cumsum scan dividing by cumprod(d)
    over 2048-sample chunks.  With d = 0.75 (overfit's base_resonance 0.5,
    sigmoid(0)) that cumprod underflows to exactly 0 within ~300 samples,
    so the *forward* pass was inf/NaN; with d = 0.975 it's ~3e-23 by the
    end of a chunk, finite going forward but overflowing in the backward
    pass.  linear_recurrence does the same scan with log-space cumsums over
    chunks short enough that exp(-S) stays bounded, and combines chunks
    without dividing, so both passes stay finite.

    d is clamped to [1e-6, 1]: a damping modulation can push it above 1
    (which would grow without bound), and log(0) has no gradient.
    """
    d = torch.clamp(damping, 1e-6, 1)
    return linear_recurrence(torch.log(d), d * forces)


def loss_func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = flattened_multiband_spectrogram(
        a, {'xs': (64, 16)}, smallest_band_size=512)
    b = flattened_multiband_spectrogram(
        b, {'xs': (64, 16)}, smallest_band_size=512)

    base_loss = torch.abs(a - b).sum()
    return base_loss


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
        filter_size=32,
        n_to_keep=n_to_keep,
        base_resonance=base_resonance).to(device)

    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True)

    collection = LmdbCollection(path='parallel2')

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

    optim = Adam(analysis_model.parameters(), lr=1e-4)

    for i, batch in enumerate(iter(stream)):
        optim.zero_grad()
        batch = batch.view(-1, 1, n_samples).to(device)
        orig_audio(batch[0, ...])

        control_signal, recon = analysis_model.forward(batch)

        recon_audio(max_norm(recon[0, ...]))

        # if we are "hard" sparsifying, then don't bother with sparsity loss
        sparsity_coeff = 0 if SPARSIFY else 1
        loss = loss_func(batch, recon) + (l0_norm(control_signal) * 100 * sparsity_coeff)

        sparsity = (control_signal > 0).sum() / control_signal.numel()

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

        print(i, loss.item(), sparsity.item())


def overfit_autoencoder(n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):
    control_rate = 512

    analysis_model = InstrumentAutoencoder(
        n_samples=n_samples,
        n_nodes=n_nodes,
        control_rate=control_rate,
        n_layers=n_layers,
        channels=CHANNELS,
        filter_size=32,
        n_to_keep=n_to_keep).to(device)

    def model_eval(model: InstrumentAutoencoder, _, target: torch.Tensor):
        control_signal, recon = model.forward(target)

        sparsity = (control_signal > 0).sum() / control_signal.numel()
        print(sparsity.item())

        # if we are "hard" sparsifying, then don't bother with sparsity loss
        sparsity_coeff = 0 if SPARSIFY else 1
        loss = loss_func(target, recon) + (l0_norm(control_signal) * 1e-3 * sparsity_coeff)
        return recon, loss

    overfit_model(
        n_samples=n_samples,
        model=analysis_model,
        loss_func=loss_func,
        model_eval=model_eval,
        collection_name='parallel2',
        learning_rate=1e-4,
        port=9998
    )


def overfit_direct(n_nodes: int, n_samples: int, n_layers: int, n_to_keep: int):
    """
    overfit_autoencoder's twin, without the encoder/hypernetwork (see
    LayerController): same segment, loss, sparsity penalty, base_resonance
    and learning rate
    """
    model = LayerController(
        n_samples=n_samples,
        n_nodes=n_nodes,
        control_rate=512,
        n_layers=n_layers,
        filter_size=32,
        n_to_keep=n_to_keep).to(device)

    def model_eval(model: LayerController, _, target: torch.Tensor):
        control_signal, recon = model.forward()

        sparsity = (control_signal > 0).sum() / control_signal.numel()
        print(sparsity.item())

        sparsity_coeff = 0 if SPARSIFY else 1
        loss = loss_func(target, recon) + (l0_norm(control_signal) * 1e-3 * sparsity_coeff)
        return recon, loss

    overfit_model(
        n_samples=n_samples,
        model=model,
        loss_func=loss_func,
        model_eval=model_eval,
        collection_name='parallel2',
        learning_rate=1e-4,
        port=9998
    )


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument(
        '--mode', choices=['train', 'overfit', 'overfit_direct'], required=True)

    args = parser.parse_args()

    if args.mode == 'train':
        train_ae(
            batch_size=2,
            n_nodes=32,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=64)
    elif args.mode == 'overfit':
        overfit_autoencoder(
            n_nodes=32,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=64)
    elif args.mode == 'overfit_direct':
        overfit_direct(
            n_nodes=32,
            n_samples=2 ** 17,
            n_layers=2,
            n_to_keep=64)
