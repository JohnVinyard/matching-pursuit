"""[markdown]


# Sparse Interpretable Audio Codec

# Introduction

# Previous Work

# Model

# Training Procedure

This article covers the continuation of work I've been pursuing in the area of sparse, interpretable audio models.  Our
goal is to decompose recordings of acoustic instruments (orchestral music from the
[MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) dataset) into constituent "events", which are encoded
as low-dimensional vectors carrying information about attack envelopes and physical resonances of both the instrument being
played and the room in which the performance occurs.

Some previous iterations of this work:


- [Iterative Decomposition Model V4](https://blog.cochlea.xyz/v4blogpost.html)
- [Gaussian/Gamma Splatting for Audio](https://blog.cochlea.xyz/gamma-audio-splat.html)


**In this newest version, we introduce a streaming algorithm so that audio segments of arbitrary lengths can be decomposed into
constituent events.**

All training and model code can be
[found here](https://github.com/JohnVinyard/matching-pursuit/blob/main/iterativedecomposition.py).


# Future Work

## Better Perceptual Audio Losses

Recent experiments use a greedy, per-event loss which maximizes the energy removed from the signal at each step, as well
as a learned, adversarial loss.  Reconstruction quality will likely benefit from a more perceptually-aligned loss and a
larger, more diverse dataset.


## Model Size, Training Time and Dataset

Firstly, this model is relatively small, weighing in at ~26M parameters (~117 MB on disk) and has only been trained for
around 24 hours, so it seems there is a lot of space to increase the model size, dataset size and training time to
further improve.  The reconstruction quality of the examples on this page is not amazing, certainly not good enough
even for a lossy audio codec, but the structure the model extracts seems like it could be used for many interesting
applications.  The training data should be expanded beyond the MusicNet dataset.

## Different Event Generator Variants

The decoder side of the model is very interesting, and all sorts of physical modelling-like approaches could yield
better, more realistic, and sparser renderings of the audio.

"""

"""[markdown]

# Cite this Article

If you'd like to cite this article, you can use the following [BibTeX block](https://bibtex.org/).

"""

# citation


"""[markdown]

# Event Scatterplot

Here is a scatterplot mapping events from four different audio segments onto a 2D plane using t-SNE.  
Each 32-dimensional event vector encodes information about attack, resonance, and room impulse response.

You can play around with a larger scatterplot of events sampled from MusicNet reconstructions [here](https://blog.cochlea.xyz/scatterv7.html).

"""

# large_scatterplot

"""[markdown]

# Streaming Algorithm for Arbitrary-Length Audio Segments

In this latest iteration of the work, we introduce a "streaming" algorithm so that we can decompose audio segments of
arbitrary lengths.

"""

"""[markdown]

## Original (Streaming) 

"""

# streaming.orig

"""[markdown]

## Reconstruction (Streaming) 

"""

# streaming.recon

"""[markdown]

# Examples

"""

"""[markdown]

## Example 1

"""

"""[markdown]

### Original Audio

"""
# example_1.orig_audio

"""[markdown]

### Reconstruction

"""

# example_1.recon_audio

"""[markdown]

### Randomized

"""

"""[markdown]

Here, we generate random event vectors with the original event times.
"""

# example_1.random_events

"""[markdown]

Here we use the original event vectors, but generate random times.

"""

# example_1.random_times


"""[markdown]

### Event Vectors

 
"""

# example_1.latents



"""[markdown]

### Event Scatterplot

Events clustered using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

"""

# example_1.scatterplot

"""[markdown]

### Individual Audio Events

"""

# example_1.event0
# example_1.event1
# example_1.event2
# example_1.event3
# example_1.event4
# example_1.event5
# example_1.event6
# example_1.event7
# example_1.event8
# example_1.event9
# example_1.event10
# example_1.event11
# example_1.event12
# example_1.event13
# example_1.event14
# example_1.event15
# example_1.event16
# example_1.event17
# example_1.event18
# example_1.event19
# example_1.event20
# example_1.event21
# example_1.event22
# example_1.event23
# example_1.event24
# example_1.event25
# example_1.event26
# example_1.event27
# example_1.event28
# example_1.event29
# example_1.event30
# example_1.event31


"""[markdown]

## Example 2

"""

"""[markdown]

### Original Audio

"""
# example_2.orig_audio

"""[markdown]

### Reconstruction

"""

# example_2.recon_audio

"""[markdown]

### Randomized

"""

"""[markdown]

Here, we generate random event vectors with the original event times.
"""

# example_2.random_events

"""[markdown]

Here we use the original event vectors, but generate random times.

"""

# example_2.random_times




"""[markdown]

### Event Vectors



"""

# example_2.latents


"""[markdown]

### Event Scatterplot

Events clustered using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

"""



# example_2.scatterplot

"""[markdown]

### Individual Audio Events

"""

# example_2.event0
# example_2.event1
# example_2.event2
# example_2.event3
# example_2.event4
# example_2.event5
# example_2.event6
# example_2.event7
# example_2.event8
# example_2.event9
# example_2.event10
# example_2.event11
# example_2.event12
# example_2.event13
# example_2.event14
# example_2.event15
# example_2.event16
# example_2.event17
# example_2.event18
# example_2.event19
# example_2.event20
# example_2.event21
# example_2.event22
# example_2.event23
# example_2.event24
# example_2.event25
# example_2.event26
# example_2.event27
# example_2.event28
# example_2.event29
# example_2.event30
# example_2.event31


"""[markdown]

## Example 3

"""

"""[markdown]

### Original Audio

"""
# example_3.orig_audio

"""[markdown]

### Reconstruction

"""

# example_3.recon_audio

"""[markdown]

### Randomized

"""

"""[markdown]

Here, we generate random event vectors with the original event times.
"""

# example_3.random_events

"""[markdown]

Here we use the original event vectors, but generate random times.

"""

# example_3.random_times



"""[markdown]

### Event Vectors



"""

# example_3.latents


"""[markdown]

### Event Scatterplot

Events clustered using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

"""

# example_3.scatterplot

"""[markdown]

### Individual Audio Events

"""

# example_3.event0
# example_3.event1
# example_3.event2
# example_3.event3
# example_3.event4
# example_3.event5
# example_3.event6
# example_3.event7
# example_3.event8
# example_3.event9
# example_3.event10
# example_3.event11
# example_3.event12
# example_3.event13
# example_3.event14
# example_3.event15
# example_3.event16
# example_3.event17
# example_3.event18
# example_3.event19
# example_3.event20
# example_3.event21
# example_3.event22
# example_3.event23
# example_3.event24
# example_3.event25
# example_3.event26
# example_3.event27
# example_3.event28
# example_3.event29
# example_3.event30
# example_3.event31


"""[markdown]

## Example 4

"""

"""[markdown]

### Original Audio

"""
# example_4.orig_audio

"""[markdown]

### Reconstruction

"""

# example_4.recon_audio

"""[markdown]

### Randomized

"""

"""[markdown]

Here, we generate random event vectors with the original event times.
"""

# example_4.random_events

"""[markdown]

Here we use the original event vectors, but generate random times.

"""

# example_4.random_times


"""[markdown]

### Event Vectors



"""

# example_4.latents


"""[markdown]

### Event Scatterplot

Events clustered using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

"""

# example_4.scatterplot

"""[markdown]

### Individual Audio Events

"""

# example_4.event0
# example_4.event1
# example_4.event2
# example_4.event3
# example_4.event4
# example_4.event5
# example_4.event6
# example_4.event7
# example_4.event8
# example_4.event9
# example_4.event10
# example_4.event11
# example_4.event12
# example_4.event13
# example_4.event14
# example_4.event15
# example_4.event16
# example_4.event17
# example_4.event18
# example_4.event19
# example_4.event20
# example_4.event21
# example_4.event22
# example_4.event23
# example_4.event24
# example_4.event25
# example_4.event26
# example_4.event27
# example_4.event28
# example_4.event29
# example_4.event30
# example_4.event31


"""[markdown]

## Example 5

"""

"""[markdown]

### Original Audio

"""
# example_5.orig_audio

"""[markdown]

### Reconstruction

"""

# example_5.recon_audio

"""[markdown]

### Randomized

"""

"""[markdown]

Here, we generate random event vectors with the original event times.
"""

# example_5.random_events

"""[markdown]

Here we use the original event vectors, but generate random times.

"""

# example_5.random_times



"""[markdown]

### Event Vectors



"""

# example_5.latents


"""[markdown]

### Event Scatterplot

Events clustered using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

"""

# example_5.scatterplot

"""[markdown]

### Individual Audio Events

"""

# example_5.event0
# example_5.event1
# example_5.event2
# example_5.event3
# example_5.event4
# example_5.event5
# example_5.event6
# example_5.event7
# example_5.event8
# example_5.event9
# example_5.event10
# example_5.event11
# example_5.event12
# example_5.event13
# example_5.event14
# example_5.event15
# example_5.event16
# example_5.event17
# example_5.event18
# example_5.event19
# example_5.event20
# example_5.event21
# example_5.event22
# example_5.event23
# example_5.event24
# example_5.event25
# example_5.event26
# example_5.event27
# example_5.event28
# example_5.event29
# example_5.event30
# example_5.event31


n_samples = 2 ** 17
samples_per_event = 2048

# this is cut in half since we'll mask out the second half of encoder activations
n_events = (n_samples // samples_per_event) // 2
context_dim = 32

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256

n_frames = n_samples // transform_step_size


from argparse import ArgumentParser
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.manifold import TSNE
from torch import nn

from conjure import S3Collection, \
    conjure_article, CitationComponent, AudioComponent, ImageComponent, \
    CompositeComponent, Logger, ScatterPlotComponent
from data import get_one_audio_segment, AudioIterator
from iterativedecomposition import Model as IterativeDecompositionModel
from modules.eventgenerators.overfitresonance import OverfitResonanceModel
from modules import max_norm, sparse_softmax


remote_collection_name = 'iterative-decomposition-v3'


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_events(
        vectors: torch.Tensor,
        times: torch.Tensor,
        total_seconds: float) -> Tuple:
    positions = torch.argmax(times, dim=-1, keepdim=True) / times.shape[-1]
    times = [float(x) for x in (positions * total_seconds).view(-1).data.cpu().numpy()]

    normalized = vectors.data.cpu().numpy().reshape((-1, context_dim))
    normalized = normalized - normalized.min(axis=0, keepdims=True)
    normalized = normalized / (normalized.max(axis=0, keepdims=True) + 1e-8)
    tsne = TSNE(n_components=2)
    points = tsne.fit_transform(normalized)
    proj = np.random.uniform(0, 1, (2, 3))
    colors = points @ proj
    colors -= colors.min()
    colors /= (colors.max() + 1e-8)
    colors *= 255
    colors = colors.astype(np.uint8)
    colors = [f'rgb({c[0]} {c[1]} {c[2]})' for c in colors]

    return points, times, colors


def load_model(wavetable_device: str = 'cpu') -> nn.Module:

    hidden_channels = 512

    model = IterativeDecompositionModel(
        in_channels=1024,
        hidden_channels=hidden_channels,
        resonance_model=OverfitResonanceModel(
            n_noise_filters=64,
            noise_expressivity=4,
            noise_filter_samples=128,
            noise_deformations=32,
            instr_expressivity=4,
            n_events=1,
            n_resonances=4096,
            n_envelopes=64,
            n_decays=64,
            n_deformations=64,
            n_samples=n_samples,
            n_frames=n_frames,
            samplerate=samplerate,
            hidden_channels=hidden_channels,
            wavetable_device=wavetable_device,
            fine_positioning=True,
            fft_resonance=True
        ))

    with open('iterativedecomposition9.dat', 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    print('Total parameters', count_parameters(model))
    print('Encoder parameters', count_parameters(model.encoder))
    print('Decoder parameters', count_parameters(model.resonance))

    return model


def scatterplot_section(logger: Logger) -> ScatterPlotComponent:
    model = load_model()
    ai = AudioIterator(
        batch_size=4,
        n_samples=n_samples,
        samplerate=22050,
        normalize=True,
        as_torch=True)

    batch = next(iter(ai))
    batch = batch.view(-1, 1, n_samples).to('cpu')
    events, vectors, times = model.iterative(batch)

    total_seconds = n_samples / samplerate

    points, times, colors = process_events(vectors, times, total_seconds)

    events = events.view(-1, n_samples)

    events = {f'event{i}': events[i: i + 1, :] for i in range(events.shape[0])}

    scatterplot_srcs = []

    event_components = {}
    for k, v in events.items():
        _, e = logger.log_sound(k, v)
        scatterplot_srcs.append(e.public_uri)
        event_components[k] = AudioComponent(e.public_uri, height=35, controls=False)

    scatterplot_component = ScatterPlotComponent(
        scatterplot_srcs,
        width=500,
        height=500,
        radius=0.3,
        points=points,
        times=times,
        colors=colors, )

    return scatterplot_component


def generate_multiple_events(
        model: nn.Module,
        vectors: torch.Tensor,
        times: torch.Tensor) -> torch.Tensor:

    generation_result = torch.cat(
        [model.generate(vectors[:, i:i + 1, :], times[:, i:i + 1, :]) for i in range(n_events)], dim=1)

    generation_result = torch.sum(generation_result, dim=1, keepdim=True)
    generation_result = max_norm(generation_result)
    return generation_result

def generate(
        model: nn.Module,
        vectors: torch.Tensor,
        times: torch.Tensor,
        randomize_events: bool,
        randomize_times: bool) -> torch.Tensor:

    batch, n_events, _ = vectors.shape

    if randomize_events:
        vectors = torch.zeros_like(vectors).uniform_(vectors.min().item(), vectors.max().item())

    if randomize_times:
        times = torch.zeros_like(times).uniform_(-1, 1)
        times = sparse_softmax(times, dim=-1, normalize=True) * times

    generation_result = generate_multiple_events(model, vectors, times)
    return generation_result


def streaming_section(logger: Logger) -> CompositeComponent:
    model = load_model()
    samples = get_one_audio_segment(n_samples * 4, samplerate, device='cpu').view(1, 1, -1)

    with torch.no_grad():
        recon = model.streaming(samples)
        recon = max_norm(recon)

    _, orig = logger.log_sound(key='streamingorig', audio=samples)
    orig = AudioComponent(orig.public_uri, height=100, controls=True, scale=4)

    _, recon = logger.log_sound(key='streamingrecon', audio=recon)
    recon = AudioComponent(recon.public_uri, height=100, controls=True, scale=4)

    return CompositeComponent(
        orig=orig,
        recon=recon,
    )


def reconstruction_section(logger: Logger) -> CompositeComponent:
    model = load_model()

    # get a random audio segment
    samples = get_one_audio_segment(n_samples, samplerate, device='cpu').view(1, 1, n_samples)
    events, vectors, times = model.iterative(samples)

    # generate audio with the same times, but randomized event vectors
    randomized_events = generate(model, vectors, times, randomize_events=True, randomize_times=False)
    _, random_events = logger.log_sound('randomizedevents', randomized_events)
    random_events_component = AudioComponent(random_events.public_uri, height=100, controls=True)

    # generate audio with the same events, but randomized times
    randomized_times = generate(model, vectors, times, randomize_events=False, randomize_times=True)
    _, random_times = logger.log_sound('randomizedtimes', randomized_times)
    random_times_component = AudioComponent(random_times.public_uri, height=100, controls=True)

    total_seconds = n_samples / samplerate

    points, times, colors = process_events(vectors, times, total_seconds)

    # sum together all events
    summed = torch.sum(events, dim=1, keepdim=True)
    summed = max_norm(summed)

    _, original = logger.log_sound(f'original', samples)
    _, reconstruction = logger.log_sound(f'reconstruction', summed)

    orig_audio_component = AudioComponent(original.public_uri, height=100)
    recon_audio_component = AudioComponent(reconstruction.public_uri, height=100)

    events = {f'event{i}': events[:, i: i + 1, :] for i in range(events.shape[1])}

    scatterplot_srcs = []

    event_components = {}
    for k, v in events.items():
        _, e = logger.log_sound(k, v)
        scatterplot_srcs.append(e.public_uri)
        event_components[k] = AudioComponent(e.public_uri, height=25, controls=False)

    scatterplot_component = ScatterPlotComponent(
        scatterplot_srcs,
        width=300,
        height=300,
        radius=0.04,
        points=points,
        times=times,
        colors=colors, )

    _, event_vectors = logger.log_matrix_with_cmap('latents', vectors[0].T, cmap='hot')
    latents = ImageComponent(event_vectors.public_uri, height=200, title='latent event vectors', full_width=False)


    composite = CompositeComponent(
        orig_audio=orig_audio_component,
        recon_audio=recon_audio_component,
        latents=latents,
        scatterplot=scatterplot_component,
        random_events=random_events_component,
        random_times=random_times_component,
        **event_components
    )

    return composite


"""[markdown]

# Notes

This blog post is generated from a
[Python script](https://github.com/JohnVinyard/matching-pursuit/blob/main/v3blogpost.py) using
[conjure](https://github.com/JohnVinyard/conjure).

[^1]:  While the STFT (short-time fourier transform) doesn't capture _everything_ of perceptual import, it does a fairly
good job, better than the "raw", time-domain audio signal, at least.  In the time domain, we get into trouble when we
begin to try to represent and remove the noisier parts of the signal;  here the statistics and relationships between
different auditory bandpass filters become more important than the precise amplitude values.  

"""


def demo_page_dict() -> Dict[str, any]:
    print(f'Generating article...')

    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    logger = Logger(remote)

    print('Creating large scatterplot')
    large_scatterplot = scatterplot_section(logger)

    print('Creating streaming section')
    streaming = streaming_section(logger)

    print('Creating reconstruction examples')
    example_1 = reconstruction_section(logger)
    example_2 = reconstruction_section(logger)
    example_3 = reconstruction_section(logger)
    example_4 = reconstruction_section(logger)
    example_5 = reconstruction_section(logger)

    citation = CitationComponent(
        tag='johnvinyarditerativedecompositionv3',
        author='Vinyard, John',
        url='https://blog.cochlea.xyz/iterative-decomposition-v7.html',
        header='Iterative Decomposition V7',
        year='2024',
    )

    return dict(
        large_scatterplot=large_scatterplot,
        streaming=streaming,
        example_1=example_1,
        example_2=example_2,
        example_3=example_3,
        example_4=example_4,
        example_5=example_5,
        citation=citation
    )


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(
        __file__,
        'html',
        title='Iterative Decomposition Model V7',
        **display)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--clear', action='store_true')
    parser.add_argument('--list', action='store_true')

    args = parser.parse_args()

    if args.list:
        remote = S3Collection(
            remote_collection_name, is_public=True, cors_enabled=True)
        print(remote)
        print('Listing stored keys')
        for key in remote.iter_prefix(start_key=b'', prefix=b''):
            print(key)

    if args.clear:
        remote = S3Collection(
            remote_collection_name, is_public=True, cors_enabled=True)
        remote.destroy(prefix=b'')

    generate_demo_page()
