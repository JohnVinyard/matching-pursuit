# videoexample

"""[markdown]

# TODO

- EMPHASIZE NATURAL SOUNDS
- write blog post
- longer audio training examples
- talk about compression ratio in the blog post
- n_examples must be hard-coded


# Extracting Instruments from Audio Recordings

My primary focus over the last two years has been on
[sparse representations of audio signals](https://blog.cochlea.xyz/sparse-interpretable-audio-codec-paper.html).
One fruitful approach thus far has been to "factor" audio signals into two components:

1. a control signal describing the way energy is injected into some system, e.g. acoustic instrument(s) and the rooms in
    which they are played
2. The acoustic resonances of the system itself; describing how energy injected from the control plane
    is stored and emitted over time.

As a side-effect, I've started to investigate whether this factorization can be used to extract useful artifacts from
very small datasets and even individual audio segments.

This micro-experiment asks whether it is possible to extract a playable instrument from a single audio recording of
one or more acoustic instruments performing a piece of music.  This approach should meet two criteria:

- reproduce the original recording faithfully given the instrument model and control signal
- allow a user to produce _new_ audio sequences by injecting energy into the learned system,
    constituting a new control signal

# User Interface

In this experiment, the control signal is defined as a sparse, N-dimensional vector, varying at ~20hz.  In this
experiment, I chose a familiar and natural interface, mapping hand-tracking landmarks onto the control plane.
[MediaPipe](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) proved indispensable for this first pass,
It made building the first draft UI feel pretty effortless.

That said, I'm very excited about the possibilities here, as multi-dimensional, time-varying sensor data is
all around us, streaming from smartphones, watches, WiFi signals and more.

The weights for the model are persisted, and then used by a TypeScript, WebAudio implementation of the decoder that runs
in the browser.

# The Model

I've attempted a similar approach before, using a
[single-layer recurrent neural network to model the instrument](https://blog.cochlea.xyz/ssm.html).  This yieled some
success, but the frame-based approach caused many audible-artifacts that I wasn't entirely happy with.

In this new work, the control signal is a 16-dimensional time-varying vector running at ~20hz.  This signal is first
convolved with a small set "attack envelopes", which are multiplied with uniform noise.  This is then convolved with
some number of resonances, parameterized by a set of damped harmonic oscillators.  Finally, the resonance output is
multiplied by a learnable gain, followed by a non-linearity (`tanh`).

Another important aspect of acoustic instrument performance is _deformations_, or changes from the resting state of
the physical object.  A deformation might be the turn of a tuning peg on a guitar or a change in the position of a slide
on a trombone. Deformations of the instrument itself are modeled as a time-varying mix of resonances, ultimately
affecting how energy is routed to each resonance. As a concrete example, the control signal might become non-zero,
modeling a plucked guitar string.  Then the deformation mix might oscillate between two different resonances,
modeling the change in bridge height induced by a whammy/tremolo bar.

# The Loss

We fit the model using a simple, L1 loss on the short-time fourier transforms of the target and reconstruction, as well
as a sparsity loss to encourage a control signal that is simple and depends on resonances for much of the audio content.

# The Results


TODO...

"""

# examples

"""[markdown]

Thanks for reading!

# Future Work

The approach is promising, but there are many questions left to explore:

- how to create an efficient implementation using JS and the WebAudio API such that we can use larger 
    control planes, more resonances, and more expressivity?
- the control-plane representation is overly-simplistic.  Are there better models
- all of thie seems to be driving toward physical modelling synthesis.  Can we just create a differentiable physics simulation?
- what is a good, natural, intuitive set of sensor data that is readily available using smartphones or some other 
    pervasive technology that can be mapped on the control-plane dimensions

"""

"""[markdown]

# Citation

If you'd like to cite this article, you can use the following [BibTeX block](https://bibtex.org/).

"""

# citation

from base64 import b64encode
from typing import Tuple, Callable, Union, Dict, Any

import numpy as np
import torch
from attr import dataclass
from sklearn.decomposition import DictionaryLearning
from torch import nn
from torch.optim import Adam

import conjure
from conjure import serve_conjure, SupportedContentType, NumpyDeserializer, NumpySerializer, Logger, MetaData, \
    CompositeComponent, AudioComponent, ConvInstrumentComponent, conjure_article, CitationComponent, S3Collection, \
    VideoComponent
from data import get_one_audio_segment
from modules import max_norm, interpolate_last_axis, sparsify, unit_norm, flattened_multiband_spectrogram, stft
from modules.transfer import fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import device, encode_audio, make_initializer
from util.overfit import LossFunc
import argparse

MaterializeResonances = Callable[..., torch.Tensor]

init_weights = make_initializer(0.02)

# TODO: How to package up and share these params
n_samples = 2 ** 17
resonance_window_size = 2048
step_size = resonance_window_size // 2
n_frames = n_samples // step_size

# KLUDGE: control_plane_dim and n_resonances
# must have the same value (for now)
control_plane_dim = 16
n_resonances = 16
expressivity = 2
n_to_keep = 64


# TODO: Move this into conjure
def encode_array(arr: Union[np.ndarray, torch.Tensor], serializer: NumpySerializer) -> str:
    if isinstance(arr, torch.Tensor):
        arr = arr.data.cpu().numpy()

    return b64encode(serializer.to_bytes(arr)).decode()


def decaying_noise(
        n_items: int,
        n_samples: int,
        low_exp: int,
        high_exp: int,
        device: torch.device,
        include_noise: bool = True):
    t = torch.linspace(1, 0, n_samples, device=device)
    pos = torch.zeros(n_items, device=device).uniform_(low_exp, high_exp)

    if include_noise:
        noise = torch.zeros(n_items, n_samples, device=device).uniform_(-1, 1)
        return (t[None, :] ** pos[:, None]) * noise
    else:
        return (t[None, :] ** pos[:, None])


def make_ramp(n_samples: int, ramp_length: int, device: torch.device) -> torch.Tensor:
    ramp = torch.ones(n_samples, device=device)
    ramp[:ramp_length] = torch.linspace(0, 1, ramp_length, device=device)
    return ramp


def materialize_attack_envelopes(
        low_res: torch.Tensor,
        window_size: int,
        is_fft: bool = False) -> torch.Tensor:
    if low_res.shape[-1] == window_size:
        return low_res * torch.zeros_like(low_res).uniform_(-1, 1)

    if is_fft:
        low_res = torch.view_as_complex(low_res)
        low_res = torch.fft.irfft(low_res)

    # impulse = fft_resample(low_res[None, ...], desired_size=window_size, is_lowest_band=True)[0]

    impulse = interpolate_last_axis(low_res, desired_size=window_size)

    impulse = impulse * torch.zeros_like(impulse).uniform_(-1, 1)

    ramp = make_ramp(impulse.shape[-1], ramp_length=10, device=impulse.device)
    impulse = impulse * ramp
    return impulse


def execute_layer(
        control_signal: torch.Tensor,
        attack_envelopes: torch.Tensor,
        mix: torch.Tensor,
        routing: torch.Tensor,
        res: torch.Tensor,
        deformations: torch.Tensor,
        gains: torch.Tensor,
        window_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, n_events, control_plane_dim, frames = control_signal.shape
    batch, n_events, expressivity, def_frames = deformations.shape

    # route control plane to resonance plane at control-rate
    routed = (control_signal.permute(0, 1, 3, 2) @ routing).permute(0, 1, 3, 2)

    before_upsample = routed

    _, _, n_resonances, expressivity, n_samples = res.shape

    # ensure that resonances all have unit norm so that control
    # plane energy drives overall loudness/response
    res = unit_norm(res)

    # "route" energy from control plane to resonances
    routed = routed.view(batch, n_events, n_resonances, 1, frames)

    # convolve with noise impulse
    routed = upsample_with_holes(routed, n_samples)

    impulse = materialize_attack_envelopes(attack_envelopes, window_size)
    impulse = ensure_last_axis_length(impulse, n_samples)

    impulse = impulse.view(1, 1, control_plane_dim, 1, n_samples)
    routed = fft_convolve(impulse, routed)

    cs = routed.view(1, control_plane_dim, n_samples).sum(dim=1, keepdim=True)

    # convolve control plane with all resonances
    conv = fft_convolve(routed, res)

    # interpolate between variations on each resonance
    base_deformation = torch.zeros_like(deformations)
    base_deformation[:, :, 0:1, :] = 1
    d = base_deformation + deformations

    d = torch.softmax(d, dim=-2)

    d = d.view(batch, n_events, 1, expressivity, def_frames)
    d = interpolate_last_axis(d, n_samples)

    x = d * conv
    x = torch.sum(x, dim=-2)

    mixes = mix.view(1, 1, n_resonances, 1, 1, 2)
    mixes = torch.softmax(mixes, dim=-1)
    stacked = torch.stack([routed, x.reshape(*routed.shape)], dim=-1)
    x = mixes * stacked
    x = torch.sum(x, dim=-1)

    x = x.view(1, 1, n_resonances, -1)

    summed = torch.tanh(x * torch.abs(gains.view(1, 1, n_resonances, 1)))

    summed = torch.sum(summed, dim=-2, keepdim=True)

    return summed, before_upsample, cs


@torch.jit.script
def damped_harmonic_oscillator(
        time: torch.Tensor,
        mass: torch.Tensor,
        damping: torch.Tensor,
        tension: torch.Tensor,
        initial_displacement: torch.Tensor,
        initial_velocity: float,
) -> torch.Tensor:
    x = (damping / (2 * mass))

    # if torch.isnan(x).sum() > 0:
    #     print('x first appearance of NaN')

    omega = torch.sqrt(torch.clamp(tension - (x ** 2), 1e-12, np.inf))

    # if torch.isnan(omega).sum() > 0:
    #     print('omega first appearance of NaN')

    phi = torch.atan2(
        (initial_velocity + (x * initial_displacement)),
        (initial_displacement * omega)
    )
    a = initial_displacement / torch.cos(phi)

    z = a * torch.exp(-x * time) * torch.cos(omega * time - phi)
    return z


class DampedHarmonicOscillatorBlock(nn.Module):
    def __init__(
            self,
            n_samples: int,
            n_oscillators: int,
            n_resonances: int,
            expressivity: int):
        super().__init__()
        self.n_samples = n_samples
        self.n_oscillators = n_oscillators
        self.n_resonances = n_resonances
        self.expressivity = expressivity

        self.damping = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(0.5, 1.5))

        self.mass = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-2, 2))

        self.tension = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(4, 9))

        self.initial_displacement = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity) \
                .uniform_(-1, 2))

        self.amplitudes = nn.Parameter(
            torch.zeros(n_oscillators, n_resonances, expressivity, 1) \
                .uniform_(-1, 1))

    def _materialize_resonances(self, device: torch.device):
        time = torch.linspace(0, 10, self.n_samples, device=device) \
            .view(1, 1, 1, self.n_samples)

        x = damped_harmonic_oscillator(
            time=time,
            mass=torch.sigmoid(self.mass[..., None]),
            damping=torch.sigmoid(self.damping[..., None]) * 30,
            tension=10 ** self.tension[..., None],
            initial_displacement=self.initial_displacement[..., None],
            initial_velocity=0
        )

        x = x.view(self.n_oscillators, self.n_resonances, self.expressivity, self.n_samples)
        x = x * self.amplitudes ** 2
        x = torch.sum(x, dim=0)

        ramp = make_ramp(self.n_samples, ramp_length=10, device=x.device)
        return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples) * ramp[None, None, None, None, :]

    def forward(self) -> torch.Tensor:
        return self._materialize_resonances(self.damping.device)


class ResonanceLayer(nn.Module):

    def __init__(
            self,
            n_samples: int,
            resonance_window_size: int,
            control_plane_dim: int,
            n_resonances: int,
            expressivity: int,
            base_resonance: float = 0.5):
        super().__init__()
        self.expressivity = expressivity
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.resonance_window_size = resonance_window_size
        self.n_samples = n_samples
        self.base_resonance = base_resonance

        self.attack_full_size = 2048

        resonance_coeffs = resonance_window_size // 2 + 1

        self.attack_envelopes = nn.Parameter(
            # decaying_noise(self.control_plane_dim, 256, 4, 20, device=device, include_noise=False)
            torch.zeros(self.control_plane_dim, self.attack_full_size).uniform_(-1, 1)
        )

        self.router = nn.Parameter(
            torch.zeros((self.control_plane_dim, self.n_resonances)).uniform_(-1, 1))

        # self.resonance = SampleLookupBlock(
        #     n_resonances * expressivity, n_samples, 64, randomize_phases=True, windowed=True)

        self.resonance = DampedHarmonicOscillatorBlock(
            n_samples, 16, n_resonances, expressivity
        )

        self.mix = nn.Parameter(torch.zeros(self.n_resonances, 2).uniform_(-1, 1))

        # self.resonance = LatentResonanceBlock(
        #     n_samples, n_resonances, expressivity, latent_dim=16)

        # self.resonance = FFTResonanceBlock(
        #     n_samples, resonance_window_size, n_resonances, expressivity, base_resonance)

        # self.resonance = MultibandFFTResonanceBlock(
        #     n_resonances,
        #     n_samples,
        #     expressivity,
        #     smallest_band_size=16384,
        #     base_resonance=0.01,
        #     window_size=512)

        # def init_resonance() -> torch.Tensor:
        #     # base resonance
        #     res = torch.zeros((n_resonances, resonance_coeffs, 1)).uniform_(0.01, 1)
        #     # variations or deformations of the base resonance
        #     deformation = torch.zeros((1, resonance_coeffs, expressivity)).uniform_(-0.02, 0.02)
        #     # expand into (n_resonances, n_deformations)
        #     return res + deformation
        #
        # self.resonances = nn.ParameterDict(dict(
        #     amp=init_resonance(),
        #     phase=init_resonance(),
        #     decay=init_resonance(),
        # ))

        self.gains = nn.Parameter(torch.zeros((n_resonances, 1)).uniform_(0.01, 1.1))

    def get_mixes(self):
        return self.mix

    def get_attack_envelopes(self):
        return materialize_attack_envelopes(self.attack_envelopes, self.attack_full_size)

    def get_materialized_resonance(self):
        return self.resonance.forward()

    def get_gains(self):
        return self.gains

    def get_router(self):
        return self.router

    def forward(
            self,
            control_signal: torch.Tensor,
            deformations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        res = self.resonance.forward()
        # print(res.shape)
        output, fwd, cs = execute_layer(
            control_signal,
            self.attack_envelopes,
            self.mix,
            self.router,
            res,
            deformations,
            self.gains,
            self.resonance_window_size,
        )
        return output, fwd, cs


class ResonanceStack(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_samples: int,
            resonance_window_size: int,
            control_plane_dim: int,
            n_resonances: int,
            expressivity: int,
            base_resonance: float = 0.5):
        super().__init__()

        self.expressivity = expressivity
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.resonance_window_size = resonance_window_size
        self.n_samples = n_samples
        self.base_resonance = base_resonance

        self.mix = nn.Parameter(torch.zeros(n_layers))

        self.layers = nn.ModuleList([ResonanceLayer(
            n_samples,
            resonance_window_size,
            control_plane_dim,
            n_resonances,
            expressivity,
            base_resonance
        ) for _ in range(n_layers)])

    def get_mix(self, layer: int):
        layer = self.layers[layer]
        return layer.get_mixes()

    def get_materialized_resonance(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_materialized_resonance()

    def get_gains(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_gains()

    def get_router(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_router()

    def get_attack_envelopes(self, layer: int) -> torch.Tensor:
        layer = self.layers[layer]
        return layer.get_attack_envelopes()

    def forward(self, control_signal: torch.Tensor, deformations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_events, cpd, frames = control_signal.shape

        outputs = []
        cs = control_signal
        fcs = None

        for layer in self.layers:
            output, cs, fcs = layer(cs, deformations)
            outputs.append(output)

        final = torch.stack(outputs, dim=-1)
        mx = torch.softmax(self.mix, dim=-1)

        final = final @ mx[:, None]
        return final.view(batch_size, n_events, self.n_samples), fcs


class OverfitResonanceStack(nn.Module):

    def __init__(
            self,
            n_layers: int,
            n_samples: int,
            resonance_window_size: int,
            control_plane_dim: int,
            n_resonances: int,
            expressivity: int,
            n_frames: int,
            base_resonance: float = 0.5,
            n_to_keep: int = 1024):
        super().__init__()
        self.expressivity = expressivity
        self.n_resonances = n_resonances
        self.control_plane_dim = control_plane_dim
        self.resonance_window_size = resonance_window_size
        self.n_samples = n_samples
        self.base_resonance = base_resonance
        self.n_frames = n_frames
        self.n_to_keep = n_to_keep

        control_plane = torch.zeros(
            (1, 1, control_plane_dim, n_frames)) \
            .uniform_(-0.01, 0.01)

        self.control_plane = nn.Parameter(control_plane)

        deformations = torch.zeros(
            (1, 1, expressivity, n_frames)).uniform_(-0.01, 0.01)
        self.deformations = nn.Parameter(deformations)

        self.network = ResonanceStack(
            n_layers=n_layers,
            n_samples=n_samples,
            resonance_window_size=resonance_window_size,
            control_plane_dim=control_plane_dim,
            n_resonances=n_resonances,
            expressivity=expressivity,
            base_resonance=base_resonance
        )

        self.apply(init_weights)

    @property
    def flattened_deformations(self):
        return self.deformations.view(self.expressivity, self.n_frames)

    def _get_mapping(self, n_components: int) -> np.ndarray:
        cs = self.control_signal.data.cpu().numpy() \
            .reshape(self.control_plane_dim, self.n_frames).T
        pca = DictionaryLearning(n_components=n_components)
        pca.fit(cs)
        # this will be of shape (n_components, control_plane_dim)
        return pca.components_

    def get_hand_tracking_mapping(self) -> np.ndarray:
        mapping = self._get_mapping(n_components=21 * 3)
        print('PCA Weight Shape', mapping.shape)
        return mapping

    def get_mixes(self, layer: int) -> torch.Tensor:
        return self.network.get_mix(layer)

    def get_materialized_resonance(self, layer: int) -> torch.Tensor:
        return self.network.get_materialized_resonance(layer)

    def get_gains(self, layer: int) -> torch.Tensor:
        return self.network.get_gains(layer)

    def get_router(self, layer: int) -> torch.Tensor:
        return self.network.get_router(layer)

    def get_attack_envelopes(self, layer: int) -> torch.Tensor:
        return self.network.get_attack_envelopes(layer)

    def _process_control_plane(
            self,
            cp: torch.Tensor,
            n_to_keep: int = None,
            do_sparsify: bool = False) -> torch.Tensor:
        cp = cp.view(1, self.control_plane_dim, self.n_frames)

        if do_sparsify:
            cp = sparsify(cp, n_to_keep=n_to_keep or self.n_to_keep)

        cp = cp.view(1, 1, self.control_plane_dim, self.n_frames)
        cp = torch.relu(cp)
        return cp

    @property
    def control_signal(self):
        cp = self.control_plane
        cp = self._process_control_plane(cp)
        return cp

    @property
    def active_elements(self) -> torch.Tensor:
        return (self.control_signal > 0).sum()

    @property
    def sparsity(self) -> torch.Tensor:
        return self.active_elements / self.control_signal.numel()

    def random(self, use_learned_deformations: bool = False):
        # print(self.control_plane.min().item(), self.control_plane.max().item())
        rcp = torch \
            .zeros_like(self.control_plane) \
            .uniform_(
            self.control_plane.min().item(),
            self.control_plane.max().item())

        rcp = self._process_control_plane(rcp, n_to_keep=16, do_sparsify=True)
        x = self.forward(
            rcp, self.deformations if use_learned_deformations else torch.zeros_like(self.deformations))
        return x

    def forward(
            self,
            cp: torch.Tensor = None,
            deformations: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cp = cp if cp is not None else self.control_signal
        deformations = deformations if deformations is not None else self.deformations
        x, cs = self.network.forward(cp, deformations)
        return x, cs, cp

    def compression_ratio(self, n_samples: int):
        # thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Note: This approach should work regardless of whether we have a "hard" number
        # of non-zero control-plan elements to keep, or if we have a sparsity penalty
        # that encourages a sparse control plane
        return (self.deformations.numel() + count_parameters(self.network) + (self.active_elements * 3)) / n_samples

    def produce_web_audio_params(self) -> dict:
        hand = self.get_hand_tracking_mapping().T
        assert hand.shape == (self.control_plane_dim, 21 * 3)

        router = self.get_router(0)
        assert router.shape == (self.control_plane_dim, self.n_resonances)

        gains = self.get_gains(0).view(-1)
        assert gains.shape == (self.n_resonances,)

        resonances = self.get_materialized_resonance(0).reshape(-1, self.n_samples)
        assert resonances.shape == (self.n_resonances * self.expressivity, self.n_samples)

        attacks = self.get_attack_envelopes(0)

        mixes = self.get_mixes(0)

        print(mixes.data.cpu().numpy())
        print(torch.softmax(mixes, dim=-1).data.cpu().numpy())

        serializer = NumpySerializer()

        params = dict(
            gains=encode_array(gains, serializer),
            router=encode_array(router, serializer),
            resonances=encode_array(resonances, serializer),
            hand=encode_array(hand, serializer),
            attacks=encode_array(attacks, serializer),
            mix=encode_array(mixes, serializer)
        )

        return params


def generate_param_dict(
        key: str,
        logger: Logger,
        model: OverfitResonanceStack) -> [dict, MetaData]:
    params = model.produce_web_audio_params()
    _, meta = logger.log_json(key, params)
    print('WEIGHTS URI', meta.public_uri.geturl())
    return params, meta


@dataclass
class OverfitModelResult:
    model: OverfitResonanceStack
    target: torch.Tensor
    recon: torch.Tensor
    rand: torch.Tensor


def transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(x, {'xs': (64, 16)})


def compute_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = transform(x)
    y = transform(y)
    return torch.abs(x - y).sum()


def produce_overfit_model(
        n_samples: int,
        resonance_window_size: int,
        control_plane_dim: int,
        n_resonances: int,
        expressivity: int,
        n_to_keep: int,
        n_iterations: int,
        loss_func: LossFunc) -> OverfitModelResult:
    """
    Overfit a resonance model on a single audio segment and return
    artifacts, including the trained model itself
    """

    step_size = resonance_window_size // 2
    n_frames = n_samples // step_size

    target = get_one_audio_segment(n_samples)
    model = OverfitResonanceStack(
        n_layers=1,
        n_samples=n_samples,
        resonance_window_size=resonance_window_size,
        control_plane_dim=control_plane_dim,
        n_resonances=n_resonances,
        expressivity=expressivity,
        base_resonance=0.01,
        n_frames=n_frames,
        n_to_keep=n_to_keep
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)

    for i in range(n_iterations):
        optimizer.zero_grad()
        recon, fcs, cp = model.forward()
        loss = loss_func(target, recon)
        print(i, loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rnd = model.random(use_learned_deformations=True)[0]

    return OverfitModelResult(
        target=max_norm(target),
        recon=max_norm(recon),
        rand=max_norm(rnd),
        model=model
    )


def produce_audio_component(
        logger: conjure.Logger,
        key: str,
        audio: Union[torch.Tensor, np.ndarray]) -> AudioComponent:
    _, meta = logger.log_sound(key, audio)
    component = AudioComponent(meta.public_uri, height=200)
    return component


def produce_content_section(
        logger: conjure.Logger,
        n_samples: int,
        resonance_window_size: int,
        control_plane_dim: int,
        n_resonances: int,
        expressivity: int,
        n_to_keep: int,
        n_iterations: int,
        loss_func: LossFunc,
        example_number: int = 1
) -> CompositeComponent:
    result = produce_overfit_model(
        n_samples,
        resonance_window_size,
        control_plane_dim,
        n_resonances,
        expressivity,
        n_to_keep,
        n_iterations,
        loss_func
    )

    orig_component = produce_audio_component(logger, 'orig', result.target)
    recon_component = produce_audio_component(logger, 'recon', result.recon)
    rand_component = produce_audio_component(logger, 'rand', result.rand)

    _, meta = generate_param_dict('model', logger, result.model)
    instr_component = ConvInstrumentComponent(meta.public_uri)

    return CompositeComponent(
        header=f'# Example {example_number}',
        orig=f'## Original Audio',
        orig_component=orig_component,
        recon=f'## Reconstruction',
        recon_component=recon_component,
        rand=f'## Random Audio',
        rand_component=rand_component,
        conv=f'## Hand-Controlled Instrument',
        conv_component=instr_component,
    )


# def produce_content_sections(
#         n_examples: int,
#         logger: conjure.Logger,
#         n_samples: int,
#         resonance_window_size: int,
#         control_plane_dim: int,
#         n_resonances: int,
#         expressivity: int,
#         n_to_keep: int,
#         n_iterations: int,
#         loss_func: LossFunc,
#     ) -> CompositeComponent:
#
#     elements = {n: produce_content_section(
#         logger,
#         n_samples,
#         resonance_window_size,
#         control_plane_dim,
#         n_resonances,
#         expressivity,
#         n_to_keep,
#         n_iterations,
#         loss_func,
#         n + 1
#     ) for n in range(n_examples)}
#
#     return CompositeComponent(**elements)



def conv_instrument_dict(
        n_examples: int,
        logger: conjure.Logger,
        n_samples: int,
        resonance_window_size: int,
        control_plane_dim: int,
        n_resonances: int,
        expressivity: int,
        n_to_keep: int,
        n_iterations: int,
        loss_func: LossFunc) -> Dict[str, Any]:
    examples = {
        f'example_{i + 1}': produce_content_section(
            logger,
            n_samples,
            resonance_window_size,
            control_plane_dim,
            n_resonances,
            expressivity,
            n_to_keep,
            n_iterations,
            loss_func)
        for i in range(n_examples)}

    return dict(
        videoexample=VideoComponent(
            src='https://state-space-model-demo-3.s3.us-east-1.amazonaws.com/rnn-instr-demo.mp4#t=14.5',
            width=500,
            height=500,
            start_time=1.4
        ),
        examples=examples,
        citation=CitationComponent(
            tag='johnvinyardresonancemodel',
            author='Vinyard, John',
            url='https://blog.cochlea.xyz/resonance-model.html',
            header='Factoring Audio Into Playable Resonance Models',
            year='2025',
        )
    )


def generate_article(n_iteraations: int, n_examples: int):
    collection = S3Collection('resonancemodel', is_public=True, cors_enabled=True)
    logger = conjure.Logger(collection)

    content = conv_instrument_dict(
        logger=logger,
        n_samples=n_samples,
        resonance_window_size=resonance_window_size,
        control_plane_dim=control_plane_dim,
        n_resonances=n_resonances,
        expressivity=expressivity,
        n_to_keep=n_to_keep,
        n_iterations=n_iteraations,
        loss_func=compute_loss,
        n_examples=n_examples
    )

    conjure_article(
        __file__,
        'html',
        title='Conv Instrument',
        web_components_version='0.0.93',
        **content
    )


def overfit_model():
    """
    function to support live training and monitoring
    """

    target = get_one_audio_segment(n_samples)
    model = OverfitResonanceStack(
        n_layers=1,
        n_samples=n_samples,
        resonance_window_size=resonance_window_size,
        control_plane_dim=control_plane_dim,
        n_resonances=n_resonances,
        expressivity=expressivity,
        base_resonance=0.01,
        n_frames=n_frames,
        n_to_keep=n_to_keep
    ).to(device)

    optimizer = Adam(model.parameters(), lr=1e-3)
    collection = conjure.LmdbCollection(path='resonancemodel')

    remote_collection = conjure.S3Collection('resonancemodel', is_public=True, cors_enabled=True)
    remote_logger = conjure.Logger(remote_collection)

    t, r, rand, res, att = conjure.loggers(
        ['target', 'recon', 'random', 'resonance', 'att'],
        'audio/wav',
        encode_audio,
        collection,
        store_history=True)

    def to_numpy(x: torch.Tensor) -> np.ndarray:
        return x.data.cpu().numpy()

    c, deformations, routing = conjure.loggers(
        ['control', 'deformations', 'routing'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer(),
        store_history=True)

    serve_conjure(
        [t, r, c, rand, res, deformations, routing, att],
        port=9999,
        n_workers=1,
        web_components_version='0.0.93')

    t(max_norm(target))

    def train():
        iteration = 0

        while True:
            optimizer.zero_grad()
            recon, fcs, cp = model.forward()

            r(max_norm(recon))
            c(model.control_signal[0, 0])

            x = stft(target, 2048, 256, pad=True)
            y = stft(recon, 2048, 256, pad=True)

            loss = torch.abs(x - y).sum() + torch.abs(cp).sum()

            loss.backward()
            optimizer.step()

            print(
                iteration,
                loss.item(),
                model.compression_ratio(n_samples),
                model.active_elements.item(),
                model.sparsity.item())

            deformations(model.flattened_deformations)
            routing(torch.abs(model.get_router(0)))

            fcs = max_norm(fcs)
            att(fcs)

            with torch.no_grad():
                rand(max_norm(model.random(use_learned_deformations=True)[0]))
                rz = model.get_materialized_resonance(0).view(-1, n_samples)
                res(max_norm(rz[np.random.randint(0, n_resonances * expressivity - 1)]))

            iteration += 1

            if iteration > 0 and iteration % 10000 == 0:
                print('Serializing')
                generate_param_dict('resonancemodelparams', remote_logger, model)
                input('Continue?')

    train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'article'], default='train')
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--examples', type=int, default=2)
    args = parser.parse_args()

    if args.mode == 'train':
        overfit_model()
    elif args.mode == 'article':
        generate_article(n_examples=args.examples, n_iteraations=args.iterations)
    else:
        raise ValueError(f'Unknown mode {args.mode}')
