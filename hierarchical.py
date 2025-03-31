"""[markdown]

# Audio Splatting

## Gaussian Splatting

In [Gaussian Splatting](https://en.wikipedia.org/wiki/Gaussian_splatting), a large number of three-dimensional
gaussians are randomly initialized and then fit via backpropagation to several two-dimensional views of a
three-dimensional scene/environment.


## Application to Audio

We draw inspiration from this field of research and apply a similar process to audio "atoms" used to compose a
reconstruction of acoustic instruments in a physical space.

In this work, we use a (roughly) physics-inspired event generator, similar to the one used in
[Toward a Sparse Interpretable Audio Codec](https://blog.cochlea.xyz/sparse-interpretable-audio-codec-paper.html) to
_overfit_ to a single audio segment, drawn from the [MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA).

## Training Process

We randomly initialize 64 event vectors and then iteratively minimize a differentiable, perceptually-inspired loss
function for 3,000 iterations.

Finally, we randomly perturb the learned/overfit event vectors to begin to get a sense for some of the ways we might
manipulate and edit the sparse representation.

## A Sparse, Interpretable Representation

The sparse, event-based representation shows promise for interpretability and manipulability, event without a [trained
encoder network](https://blog.cochlea.xyz/sparse-interpretable-audio-codec-paper.html#Encoder).

## Future Work
The current work deals only with mono audio at 22050hz, but it's possible to imagine extending to stereo, or even
multi-microphone situations where a three-dimensional sound "field" needs to be approximated.


A previous version of this article can be found [here](https://blog.cochlea.xyz/gamma-audio-splat.html).

All code for this experiment can be found
[here](https://github.com/JohnVinyard/matching-pursuit/blob/main/hierarchical.py).

"""
from spiking import AutocorrelationLoss, SpikingModel

"""[markdown]

## Cite this Work

If you'd like to cite this article, you can use the following [BibTeX block](https://bibtex.org/).

"""

# citation

"""[markdown]

## Example 1

"""

# example_1


"""[markdown]

## Example 2

"""

# example_2




import argparse
from itertools import count
from typing import Tuple, Dict, List

import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, loggers, serve_conjure, SupportedContentType, NumpySerializer, NumpyDeserializer, \
    conjure_article, CompositeComponent, S3Collection, AudioComponent, AudioTimelineComponent, CitationComponent
from conjure.logger import encode_audio, Logger
from data import get_one_audio_segment
from modules import unit_norm, flattened_multiband_spectrogram, sparse_softmax, max_norm, amplitude_envelope, \
    iterative_loss, stft
from modules.eventgenerators.splat import SplattingEventGenerator
from modules.infoloss import CorrelationLoss
from modules.multiheadtransform import MultiHeadTransform
from util import device, make_initializer
from sklearn.manifold import TSNE


initializer = make_initializer(0.05)


article_title = 'Audio Splatting With Physics-Inspired Event Generators'

class OverfitHierarchicalEvents(nn.Module):
    def __init__(
            self,
            n_samples: int,
            samplerate: int,
            n_events: int,
            context_dim: int):
        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.n_events = n_events
        self.context_dim = context_dim

        self.n_frames = n_samples // 256

        event_levels = int(np.log2(n_events))
        total_levels = int(np.log2(n_samples))

        self.event_levels = event_levels

        self.event_generator = SplattingEventGenerator(
            n_samples=n_samples,
            samplerate=samplerate,
            n_resonance_octaves=16,
            n_frames=n_samples // 256,
            hard_reverb_choice=False,
            hierarchical_scheduler=True,
            wavetable_resonance=False,
        )
        self.transform = MultiHeadTransform(
            self.context_dim, hidden_channels=128, shapes=self.event_generator.shape_spec, n_layers=1)

        self.event_time_dim = int(np.log2(self.n_samples))

        rng = 0.1

        self.event_vectors = nn.Parameter(torch.zeros(1, 2, self.context_dim).uniform_(-rng, rng))
        self.hierarchical_event_vectors = nn.ParameterDict(
            {str(i): torch.zeros(1, 2, self.context_dim).uniform_(-rng, rng) for i in range(event_levels - 1)})

        self.times = nn.Parameter(
            torch.zeros(1, 2, total_levels, 2).uniform_(-rng, rng))
        self.hierarchical_time_vectors = nn.ParameterDict(
            {str(i): torch.zeros(1, (2 ** (i + 2)), total_levels, 2).uniform_(-rng, rng) for i in
             range(event_levels - 1)})

        self.apply(initializer)

    @property
    def normalized_atoms(self):
        return unit_norm(self.atoms, dim=-1)

    def _forward(
            self,
            events: torch.Tensor,
            times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(self.event_levels - 1):
            # scale = 1 / (i + 1)
            scale = 1

            # TODO: consider bringing back scaling as we approach the leaves of the tree
            events = \
                events.view(1, -1, 1, self.context_dim) \
                + (self.hierarchical_event_vectors[str(i)].view(1, 1, 2, self.context_dim) * scale)
            events = events.view(1, -1, self.context_dim)

            # TODO: Consider masking lower bits as we approach the leaves of the tree, so that
            # new levels can only _refine_, and not completely move entire branches
            batch, n_events, n_bits, _ = times.shape
            times = times.view(batch, n_events, 1, n_bits, 2).repeat(1, 1, 2, 1, 1).view(batch, n_events * 2, n_bits, 2)
            times = times + (self.hierarchical_time_vectors[str(i)] * scale)

        event_vectors = events

        params = self.transform.forward(events)
        print('TIMES', times.shape)
        events = self.event_generator.forward(**params, times=times)
        return events, event_vectors, times

    def perturbed(self):
        events = self.event_vectors.clone()
        times = self.times.clone()

        perturbation = torch.zeros_like(events).uniform_(-0.5, 0.5)
        return self._forward(events + perturbation, times)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        events = self.event_vectors.clone()
        times = self.times.clone()

        return self._forward(events, times)


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)
    # return stft(x, 2048, 256, pad=True)

def reconstruction_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    target_spec = loss_transform(target)
    recon_spec = loss_transform(recon)
    loss = torch.abs(target_spec - recon_spec).sum()
    return loss


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def overfit():
    n_samples = 2 ** 16
    samplerate = 22050
    n_events = 32
    event_dim = 16

    # Begin: this would be a nice little helper to wrap up
    collection = LmdbCollection(path='hierarchical')
    collection.destroy()
    collection = LmdbCollection(path='hierarchical')

    recon_audio, orig_audio, perturbed_audio = loggers(
        ['recon', 'orig', 'perturbed'],
        'audio/wav',
        encode_audio,
        collection)

    eventvectors, eventtimes = loggers(
        ['eventvectors', 'eventtimes'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer()
    )

    audio = get_one_audio_segment(n_samples, samplerate, device='cpu')
    target = audio.view(1, 1, n_samples).to(device)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
        perturbed_audio,
        eventvectors,
        eventtimes,
    ], port=9999, n_workers=1)
    # end proposed helper function

    model = OverfitHierarchicalEvents(
        n_samples, samplerate, n_events, context_dim=event_dim).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    loss_model = CorrelationLoss(n_elements=512).to(device)

    # loss_model = AutocorrelationLoss(64, 64).to(device)
    # loss_model = SpikingModel(64, 64, 64, memory_size=16).to(device)
    loss_model = CorrelationLoss(512).to(device)

    for i in count():
        optim.zero_grad()
        recon, vectors, times = model.forward()

        times = sparse_softmax(times, dim=-1)
        weights = torch.from_numpy(np.array([0, 1])).to(device).float()

        eventvectors(max_norm(vectors[0]))
        t = times[0] @ weights
        eventtimes((t > 0).float())

        recon_summed = torch.sum(recon, dim=1, keepdim=True)
        recon_audio(max_norm(recon_summed))

        perturbed, _, _ = model.perturbed()
        perturbed_summed = torch.sum(perturbed, dim=1, keepdim=True)
        perturbed_summed = max_norm(perturbed_summed)
        perturbed_audio(perturbed_summed)

        # loss = loss_model.compute_multiband_loss(target, recon)
        loss = loss_model.multiband_noise_loss(target, recon_summed, 64, 16)

        # loss = iterative_loss(target, recon, loss_transform, ratio_loss=False)

        # loss = reconstruction_loss(target, recon_summed)
        # t = loss_model.forward(target)
        # r = loss_model.forward(recon_summed)
        # loss = torch.abs(t - r).sum()
        # loss = loss_model.compute_multiband_loss(target, recon_summed, 64, 16)
        # loss = loss_model.multiband_noise_loss(target, recon_summed, 128, 32)

        loss.backward()
        optim.step()
        print(i, loss.item())

def process_events2(
        logger: Logger,
        events: torch.Tensor,
        vectors: torch.Tensor,
        times: torch.Tensor,
        total_seconds: float,
        n_events:int,
        context_dim: int) -> Tuple[List[Dict], Dict]:

    # compute amplitude envelopes
    envelopes = amplitude_envelope(events, 128).data.cpu().numpy().reshape((n_events, -1))

    norms = torch.norm(events, dim=-1).reshape((-1))
    max_norm = torch.max(norms)
    opacities = norms / (max_norm + 1e-12)

    # compute event positions/times, in seconds
    times = [float(x) for x in times.reshape((-1,))]

    # normalize event vectors and map onto the y dimension
    normalized = vectors.data.cpu().numpy().reshape((-1, context_dim))
    normalized = normalized - normalized.min(axis=0, keepdims=True)
    normalized = normalized / (normalized.max(axis=0, keepdims=True) + 1e-8)
    tsne = TSNE(n_components=1)
    points = tsne.fit_transform(normalized)
    points = points - points.min()
    points = points / (points.max() + 1e-8)

    # create a random projection to map colors
    proj = np.random.uniform(0, 1, (context_dim, 3))
    colors = normalized @ proj
    colors -= colors.min()
    colors /= (colors.max() + 1e-8)
    colors *= 255
    colors = colors.astype(np.uint8)

    colors = [f'rgba({c[0]}, {c[1]}, {c[2]}, {opacities[i]})' for i, c in enumerate(colors)]

    evts = {f'event{i}': events[:, i: i + 1, :] for i in range(events.shape[1])}

    #
    #
    event_components = {}
    # for k, v in events.items():
    #     _, e = logger.log_sound(k, v)
    #     scatterplot_srcs.append(e.public_uri)
    #     event_components[k] = AudioComponent(e.public_uri, height=15, controls=False)

    scatterplot_srcs = []

    for k, v in evts.items():
        _, e = logger.log_sound(k, v)
        scatterplot_srcs.append(e.public_uri)
        event_components[k] = AudioComponent(e.public_uri, height=15, controls=False)


    return [{
        'eventTime': times[i],
        'offset': times[i],
        'y': float(points[i]),
        'color': colors[i],
        'audioUrl': scatterplot_srcs[i].geturl(),
        'eventEnvelope': envelopes[i].tolist(),
        'eventDuration': total_seconds,
    } for i in range(n_events)], event_components

    # t = np.array(times) / total_seconds
    # points = np.concatenate([points.reshape((-1, 1)), t.reshape((-1, 1))], axis=-1)
    #
    # return points, times, colors


def reconstruction_section(logger: Logger, samplerate: int, context_dim: int, n_iterations: int = 1000) -> CompositeComponent:
    n_samples = 2 ** 16
    samplerate = 22050
    n_events = 64
    event_dim = 16

    total_seconds = n_samples / samplerate

    audio = get_one_audio_segment(n_samples, samplerate, device='cpu')
    target = audio.view(1, 1, n_samples).to(device)

    model = OverfitHierarchicalEvents(
        n_samples, samplerate, n_events, context_dim=event_dim).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    loss_model = CorrelationLoss(n_elements=512).to(device)

    for i in range(n_iterations):
        optim.zero_grad()
        recon, vectors, times = model.forward()
        recon_summed = torch.sum(recon, dim=1, keepdim=True)

        # loss = loss_model.multiband_noise_loss(target, recon_summed, 128, 32)
        # loss = iterative_loss(target, recon, loss_transform)

        t = loss_transform(target)
        r = loss_transform(recon_summed)
        loss = torch.abs(t - r).sum()

        loss.backward()
        optim.step()
        print(i, loss.item())

    recon, vectors, times = model.forward()

    perturbed, _, _ = model.perturbed()
    perturbed = torch.sum(perturbed, dim=1, keepdim=True)


    recon_summed = torch.sum(recon, dim=1, keepdim=True)
    recon_summed = max_norm(recon_summed)

    # first, ensure only one element is activated
    times = sparse_softmax(times, dim=-1)

    # project to [0, 1] space
    weights = torch.from_numpy(np.array([0, 1])).to(device).float()
    t = times[0] @ weights
    t = t.data.cpu().numpy()

    time_levels = int(np.log2(n_samples))

    # create coefficients for each entry in the binary vector
    time_coeffs = np.zeros((time_levels, 1))
    time_coeffs[:] = 2
    exponents = np.linspace(0, time_levels - 1, time_levels)
    print('EXPONENTS', exponents, exponents.shape)
    time_coeffs = time_coeffs ** exponents
    print(time_coeffs)

    sample_times = t @ time_coeffs[..., None]
    print('TIME IN SAMPLES', sample_times.min(), sample_times.max())
    times_in_seconds = sample_times / samplerate
    print('TIMES IN SECONDS', times_in_seconds.min(), times_in_seconds.max())

    _, orig_audio = logger.log_sound('orig', target)
    orig_component = AudioComponent(orig_audio.public_uri, height=100)

    _, recon_audio = logger.log_sound('recon', recon_summed)
    recon_component = AudioComponent(recon_audio.public_uri, height=100)

    _, p_audio = logger.log_sound('perturbed', perturbed)
    p_component = AudioComponent(p_audio.public_uri, height=100)

    events, event_components = process_events2(logger, recon, vectors, times_in_seconds, total_seconds, n_events, context_dim)

    timeline = AudioTimelineComponent(duration=total_seconds, width=1000, height=500, events=events)

    return CompositeComponent(
        orig='Original Audio',
        orig_audio=orig_component,
        recon='Reconstruction',
        recon_audio=recon_component,
        perturbed='Perturbed Audio',
        perturbed_audio=p_component,
        timeline='Timeline',
        timeline_component=timeline
    )


def demo_page_dict() -> Dict[str, any]:
    remote = S3Collection('audio-splatting', is_public=True, cors_enabled=True)
    logger = Logger(remote)

    n_iterations = 2000
    samplerate = 22050
    context_dim = 16

    example_1 = reconstruction_section(logger, samplerate, context_dim, n_iterations)
    example_2 = reconstruction_section(logger, samplerate, context_dim, n_iterations)
    # example_3 = reconstruction_section(logger, samplerate, context_dim, n_iterations)
    # example_4 = reconstruction_section(logger, samplerate, context_dim, n_iterations)

    citation = CitationComponent(
        tag='johnvinyardaudiosplatting',
        author='Vinyard, John',
        url='https://blog.cochlea.xyz/audio-splatting.html',
        header=article_title,
        year='2025',
    )

    return dict(
        example_1=example_1,
        example_2=example_2,
        # example_3=example_3,
        # example_4=example_4,
        citation=citation
    )


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(
        __file__,
        'html',
        title=article_title,
        **display)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'demo'])

    args = parser.parse_args()

    if args.mode == 'train':
        overfit()
    else:
        generate_demo_page()
