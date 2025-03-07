"""[markdown]

This scatterplot represents musical events from a number of short (~3 second) segments of classical pieces drawn from the
[MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA).  The segments were generated by a neural network
that iteratively/incrementally decomposes the audio into a sparse set of events and times-of-occurrence.

Each event is represented by a 32-dimensional vector, which describes the attack envelope and the resonance of both
the instrument and the room in which the performance occurs.  Events are projected into a 2D space via t-SNE, and
colors are chosen via a random projection into 3D color space.

You can read more about the model architecture and training procedure, and listen to reconstructions
[here](https://blog.cochlea.xyz/v4blogpost.html).

Click or tap to play individual events.
"""

# large_scatterplot



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
    conjure_article, AudioComponent, Logger, ScatterPlotComponent
from data import AudioIterator
from iterativedecomposition import Model as IterativeDecompositionModel
from modules.eventgenerators.overfitresonance import OverfitResonanceModel
from util import count_parameters


remote_collection_name = 'iterative-decomposition-scatterplot'


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def process_events(
        vectors: torch.Tensor,
        times: torch.Tensor,
        total_seconds: float
) -> Tuple:
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
            fine_positioning=False,
            fft_resonance=True
        ))


    with open('iterativedecomposition7.dat', 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    print('Total parameters', count_parameters(model))
    print('Encoder parameters', count_parameters(model.encoder))
    print('Decoder parameters', count_parameters(model.resonance))

    return model


def scatterplot_section(logger: Logger, total_segments: int) -> ScatterPlotComponent:
    model = load_model()
    ai = AudioIterator(
        batch_size=1,
        n_samples=n_samples,
        samplerate=22050,
        normalize=True,
        as_torch=True)

    all_vectors = []
    all_events = []
    all_times = []

    for i in range(total_segments):
        batch = next(iter(ai))
        print(f'processing segment {i}')
        batch = batch.view(-1, 1, n_samples).to('cpu')
        with torch.no_grad():
            events, vectors, times = model.iterative(batch)
        all_times.append(times)
        all_events.append(events)
        all_vectors.append(vectors)

    vectors = torch.cat(all_vectors, dim=1)
    events = torch.cat(all_events, dim=1)
    times = torch.cat(all_times, dim=1)

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
        width=1500,
        height=1500,
        radius=0.5,
        points=points,
        times=times,
        colors=colors, )

    return scatterplot_component


def demo_page_dict() -> Dict[str, any]:
    print(f'Generating article...')

    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    logger = Logger(remote)

    print('Creating large scatterplot')
    large_scatterplot = scatterplot_section(logger, total_segments=32)


    return dict(
        large_scatterplot=large_scatterplot,
    )


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(
        __file__,
        'html',
        title='Iterative Decomposition Events Scatterplot',
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
