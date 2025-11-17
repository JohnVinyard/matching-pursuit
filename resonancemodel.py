"""[markdown]

In this micro-experiment, I ask whether it's possible to factor short audio segments into two distinct parts:  first a
control-signal, an abstract representation of the actions taken by a human performer, and second, an "instrument",
representing the resonances of the physical object(s) being played, and the room in which the performance occurs.

The weights, or learned parameters of the instrument artifact can be used to produce _new_ musical performances,
while the two factors together constitute a (lossy) compressive and interpretable representation of the original signal.

To make this more concrete, here's a video of me using a learned instrument to produce a new audio segment via a
hand-tracking interface.
"""

# videoexample


"""[markdown]

If you're excited to just listen to reconstructions and play with the "instrument" artifacts, feel free to 
[jump ahead](#Examples) to the audio examples!


# Extracting Instruments from Audio Recordings

My primary focus over the last two years has been on finding
[sparse representations of audio signals](https://blog.cochlea.xyz/sparse-interpretable-audio-codec-paper.html).
One promising approach has been to build models that factor audio signals into two components:

1. a control signal, describing the way energy is _injected_ into some system;  the work of a human performer.
2. The acoustic resonances of the system itself; describing how energy injected from the control signal
    is stored by the system and emitted over time.

Human musicians _aren't_ engaged in wiggling physical objects many hundreds or thousands of times per-second, but they
do learn to skillfully manipulate these properties to create engaging music.  This mental model naturally leads to 
sparser, score-like representations of musical audio.

I've also begun to investigate whether this approach can extract useful artifacts from very small datasets, and in 
this case, even _individual audio segments_.

# The Model

Instead of compiling a massive dataset and worshipping at the altar of the 
["Bitter Lesson"](https://en.wikipedia.org/wiki/Bitter_lesson), I instead cram this experiment full of inductive 
biases, imposing my own, rudimentary mental model of a musical instrument as a strong form of regularization.  
My hypothesis is that we can learn something _useful_ from even a few seconds of audio by insisting that it fits a
simple model of physics and resonance.

I've attempted a similar experiment before, using a
[single-layer recurrent neural network to model the instrument](https://blog.cochlea.xyz/ssm.html).  This yielded some 
limited success, but the frame-based approach caused many audible artifacts that I wasn't happy with.

In this new experiment, the control signal is a 16-dimensional time-varying vector running at ~20hz.  This signal is first
convolved with a small set of "attack envelopes", which are multiplied with uniform (white) noise.  These attacks are then 
routed to and convolved with some number of resonances, parameterized by a set of 
[damped harmonic oscillators](https://phys.libretexts.org/Bookshelves/University_Physics/University_Physics_(OpenStax)/Book%3A_University_Physics_I_-_Mechanics_Sound_Oscillations_and_Waves_(OpenStax)/15%3A_Oscillations/15.06%3A_Damped_Oscillations).  
Finally, the resonance output is multiplied by a learnable gain, followed by a non-linearity (`tanh`) to simulate subtle 
distortions.

We also model _deformations_, or changes from the resting state of
the physical object.  A deformation  in the real-world might be the turn of a tuning peg on a guitar or a change in the 
position of a slide on a trombone. Deformations are modeled as a time-varying change in the weights that determine 
how energy is routed to different resonances. As a concrete example, the control signal might become non-zero,
modeling a transient pluck of a  guitar string.  Then the deformation mix might oscillate between two different resonances,
modeling the change in bridge height induced by a whammy/tremolo bar.

**Importantly**, assuming we've chosen a model with sufficient capacity to describe the resonances, the control signal
scales with the length of the audio segment, but the model of the resonances does not;  it is _constant_.

# The Loss

We fit the model using a simple, L1 loss on a multi-resolution short-time fourier transform of the target and reconstruction.  
I have a strong intuition that a more perceptually-informed loss would work better, and require even _less_ overall 
model capacity, but that's an experiment for another day!

We encourage sparsity in the control signal with an addition L1 loss on control signal magnitudes, pushing most entries
to zero.  We hope that the model will rely heavily on resonances in the object, rather than a high-energy, busy control
signal.

For the examples on this page, we train the model for 10,000 iterations using the Adam optimizer.


# User Interface

Our control signal is defined as a sparse, N-dimensional vector, varying at ~20hz.  For this work, 
I chose a familiar and natural interface, the human hand.  We project hand-tracking landmarks onto the control 
signal's input space.  I should note that [MediaPipe](https://mediapipe.readthedocs.io/en/latest/solutions/hands.html) 
proved indispensable for this first pass, making this part of the implementation straightforward and quick.

I'm certain that the current interface is not ideal, and I'm very excited about alternative possibilities.
We're literally _swimming_ in multi-dimensional, time-varying sensor data, streaming from smartphones, watches, 
WiFi signals and much more!  

After overfitting a single audio segment of ~12 seconds, weights for the model are persisted, and then used by a 
[WebAudio](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API) implementation of the decoder that runs 
in the browser.

To ensure that the browser-based implementation can perform in realtime, the model we learn is _tiny_, with a control plane
dimension of 16, a resonance count of 16, and "expressivity" of 2, the latter defining how many alternative resonances 
make up out deformation space.

# The Results

## Reconstruction Quality

The reconstruction quality leaves much to be desired, but there's no doubt that the model captures many key aspects of 
the sound.  What's more, the "instrument" artifact allows us to play in the resonance space of the original recording, 
producing the same tones and timbres in novel arrangements. 

## Model Size and Compression Ratio

Comparing the model size, including the control signal and deformations, which vary over time, we end up with a 
representation about 14% the size of the original WAV audio.  Clearly, this compression is lossy, and there's 
much room for improvement.

## Size of Web Audio Parameters

We fully "materialize" both the attack envelopes and resonances to our full sample rate (22050hz, in this case) to avoid
implementing the interpolation and damped-harmonic oscillator code in TypeScript/JavaScript.  This increases the size 
of the stored model significantly and could be replaced with a one-time upsampling operation prior to decoding in 
future versions, to save disk space.

"""

"""[markdown]

# Examples

"""

# examples

"""[markdown]

Thanks for reading!

# Future Work

The approach is promising, but there are many questions left to explore:

1. how to create an efficient implementation using JavaScript and the WebAudio API such that we can use larger 
    control planes, more resonances, and more expressivity/deform-ability?
1. the control-plane representation is overly-simplistic.  Are there better models?
1. This work _seems_ to be driving toward physical modelling synthesis.  Can we just create a differentiable physics simulation?
1. what is a good, natural, intuitive set of sensor data that is readily available using smartphones or some other 
    pervasive technology that can be mapped on the control-plane dimensions in natural and fun ways?

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
    VideoComponent, ImageComponent
from data import get_one_audio_segment
from modules import max_norm, interpolate_last_axis, sparsify, unit_norm, stft
from modules.transfer import fft_convolve
from modules.upsample import upsample_with_holes, ensure_last_axis_length
from util import device, encode_audio, make_initializer
import argparse


LossFunc = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]

MaterializeResonances = Callable[..., torch.Tensor]

init_weights = make_initializer(0.02)

# TODO: How to package up and share these params
n_samples = 2 ** 17
resonance_window_size = 256
step_size = resonance_window_size // 2
n_frames = n_samples // step_size

# KLUDGE: control_plane_dim and n_resonances
# must have the same value (for now)
control_plane_dim = 16
n_resonances = 16
expressivity = 2
n_to_keep = 128
do_sparsify = False
sparsity_coefficient = 0.5
n_oscillators = 64

attack_full_size = 2048
attack_n_frames = 256
deformation_frame_size = 128

web_components_version = '0.0.101'

use_learned_deformations_for_random = False


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


# def make_ramp(n_samples: int, ramp_length: int, device: torch.device) -> torch.Tensor:
#     ramp = torch.ones(n_samples, device=device)
#     ramp[:ramp_length] = torch.linspace(0, 1, ramp_length, device=device)
#     return ramp


def materialize_attack_envelopes(
        low_res: torch.Tensor,
        window_size: int,
        is_fft: bool = False,
        add_noise: bool = True) -> torch.Tensor:

    if low_res.shape[-1] == window_size:
        return low_res * torch.zeros_like(low_res).uniform_(-1, 1)

    if is_fft:
        low_res = torch.view_as_complex(low_res)
        low_res = torch.fft.irfft(low_res)

    # impulse = fft_resample(low_res[None, ...], desired_size=window_size, is_lowest_band=True)[0]

    impulse = interpolate_last_axis(low_res, desired_size=window_size)

    if add_noise:
        impulse = impulse * torch.zeros_like(impulse).uniform_(-1, 1)

    # ramp = make_ramp(impulse.shape[-1], ramp_length=10, device=impulse.device)
    # impulse = impulse * ramp
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
    stacked = torch.stack([
        routed,
        x.reshape(*routed.shape)
    ], dim=-1)
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

        # ramp = make_ramp(self.n_samples, ramp_length=10, device=x.device)
        return x.view(1, 1, self.n_resonances, self.expressivity, self.n_samples) #* ramp[None, None, None, None, :]

    def forward(self) -> torch.Tensor:
        x = self._materialize_resonances(self.damping.device)
        x = unit_norm(x)
        return x


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

        self.attack_full_size = attack_full_size

        resonance_coeffs = resonance_window_size // 2 + 1

        self.attack_envelopes = nn.Parameter(
            # decaying_noise(self.control_plane_dim, 256, 4, 20, device=device, include_noise=False)
            torch.zeros(self.control_plane_dim, attack_n_frames).uniform_(-1, 1)
        )

        self.router = nn.Parameter(
            torch.zeros((self.control_plane_dim, self.n_resonances)).uniform_(-1, 1))


        self.resonance = DampedHarmonicOscillatorBlock(
            n_samples, n_oscillators, n_resonances, expressivity
        )

        self.mix = nn.Parameter(torch.zeros(self.n_resonances, 2).uniform_(-1, 1))


        self.gains = nn.Parameter(torch.zeros((n_resonances, 1)).uniform_(0.01, 1.1))

    def get_mixes(self):
        return self.mix

    def get_attack_envelopes(self):
        # The web audio component adds random noise each time, instead of "baking" a single
        # uniform sampling into the attack envelopes
        return materialize_attack_envelopes(self.attack_envelopes, self.attack_full_size, add_noise=False)

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
            self.attack_full_size,
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

        # self.resonance_frame_size = 1024

        self.deformation_frame_size = deformation_frame_size
        self.deformation_frames = n_samples // self.deformation_frame_size


        control_plane = torch.zeros(
            (1, 1, control_plane_dim, n_frames)) \
            .uniform_(-0.01, 0.01)

        self.control_plane = nn.Parameter(control_plane)

        deformations = torch.zeros(
            (1, 1, expressivity, self.deformation_frames)).uniform_(-0.01, 0.01)
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
        return self.deformations.view(self.expressivity, self.deformation_frames)

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
            do_sparsify: bool = do_sparsify) -> torch.Tensor:
        cp = cp.view(1, self.control_plane_dim, self.n_frames)

        # I _think_ this allows gradients to flow back to all elements, rather than
        # just the top-k
        cp = cp + cp.mean()

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

        router = self.get_router(0).T
        assert router.shape == (self.control_plane_dim, self.n_resonances)

        gains = self.get_gains(0).view(-1)
        assert gains.shape == (self.n_resonances,)

        resonances = self.get_materialized_resonance(0).reshape(-1, self.n_samples)
        assert resonances.shape == (self.n_resonances * self.expressivity, self.n_samples)

        attacks = self.get_attack_envelopes(0)

        mixes = self.get_mixes(0)


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
    control_signal: torch.Tensor


def transform(x: torch.Tensor) -> torch.Tensor:
    return stft(x, 2048, 256, pad=True)


def compute_loss(
        x: torch.Tensor,
        y: torch.Tensor,
        cp: torch.Tensor,
        sparsity_loss: float = sparsity_coefficient) -> torch.Tensor:
    x = transform(x)
    y = transform(y)
    return torch.abs(x - y).sum() + (torch.abs(cp).sum() * sparsity_loss)


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
    target = max_norm(target)

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
        loss = loss_func(target, recon, cp)
        print(i, loss.item())
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        rnd = model.random(use_learned_deformations=use_learned_deformations_for_random)[0]

    return OverfitModelResult(
        target=max_norm(target),
        recon=max_norm(recon),
        rand=max_norm(rnd),
        model=model,
        control_signal=cp
    )


def produce_audio_component(
        logger: conjure.Logger,
        key: str,
        audio: Union[torch.Tensor, np.ndarray]) -> AudioComponent:
    _, meta = logger.log_sound(key, audio)
    component = AudioComponent(meta.public_uri, height=200)
    return component

def produce_control_signal_component(logger: conjure.Logger, key: str, control_signal: torch.Tensor) -> ImageComponent:
    _, meta = logger.log_matrix_with_cmap(key, control_signal.view(control_plane_dim, n_frames), cmap='gray')
    component = ImageComponent(meta.public_uri, height=200)
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
    cs_component = produce_control_signal_component(logger, 'cs', result.control_signal)

    _, meta = generate_param_dict('model', logger, result.model)

    print('INSTRUMENT URI', meta.public_uri)
    instr_component = ConvInstrumentComponent(meta.public_uri)

    return CompositeComponent(
        header=f'# Example {example_number}',
        orig=f'## Original Audio',
        orig_component=orig_component,
        recon=f'## Reconstruction',
        recon_component=recon_component,
        control_signal_header=f'## Sparse Control Signal',
        control_signal=cs_component,
        rand=f'## Random Audio',
        rand_component=rand_component,
        conv=f'## Hand-Controlled Instrument',
        conv_component=instr_component,
    )


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
            loss_func,
            example_number=i + 1)
        for i in range(n_examples)}

    return dict(
        videoexample=VideoComponent(
            src='https://resonancemodel.s3.us-east-1.amazonaws.com/resonancemodel.mp4',
            width=500,
            height=500,
            start_time=1.4
        ),
        examples=CompositeComponent(**examples),
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
        title='Extracting Playable Instrument Models from Short Audio Examples',
        web_components_version=web_components_version,
        **content
    )


def overfit_model():
    """
    function to support live training and monitoring
    """

    target = get_one_audio_segment(n_samples)
    target = max_norm(target)

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
        web_components_version=web_components_version)

    t(max_norm(target))

    def train():
        iteration = 0

        while True:
            optimizer.zero_grad()
            recon, fcs, cp = model.forward()

            r(max_norm(recon))
            c(model.control_signal[0, 0])


            loss = compute_loss(target, recon, cp)

            loss.backward()
            optimizer.step()

            print(
                iteration,
                loss.item(),
                model.compression_ratio(n_samples).item(),
                model.active_elements.item(),
                model.sparsity.item())

            deformations(model.flattened_deformations)
            routing(torch.abs(model.get_router(0)))

            fcs = max_norm(fcs)
            att(fcs)

            with torch.no_grad():
                rand(max_norm(model.random(use_learned_deformations=use_learned_deformations_for_random)[0]))
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
