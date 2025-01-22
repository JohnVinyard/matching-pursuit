"""[markdown]

# Streaming

## Original (Streaming)

"""

# streaming.orig

"""[markdown]

## Reconstruction (Streaming) 

"""

# streaming.recon

"""[markdown]

## Events

"""

# streaming.scatter


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
from typing import Dict

import torch
from torch import nn

from conjure import S3Collection, \
    conjure_article, CitationComponent, AudioComponent, \
    CompositeComponent, Logger, ScatterPlotComponent
from data import get_one_audio_segment
from iterativedecomposition import Model as IterativeDecompositionModel
from modules.eventgenerators.overfitresonance import OverfitResonanceModel
import numpy as np
from typing import Tuple
from sklearn.manifold import TSNE
from modules import max_norm
from util import device


remote_collection_name = 'streaming-report-v1'


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def process_events(
        vectors: torch.Tensor,
        times: torch.Tensor,
        total_seconds: float) -> Tuple:

    # compute each event time in seconds
    positions = torch.argmax(times, dim=-1, keepdim=True) / times.shape[-1]
    times = [float(x) for x in (positions * total_seconds).view(-1).data.cpu().numpy()]

    # normalize event vectors
    normalized = vectors.data.cpu().numpy().reshape((-1, context_dim))
    normalized = normalized - normalized.min(axis=0, keepdims=True)
    normalized = normalized / (normalized.max(axis=0, keepdims=True) + 1e-8)

    # map normalized event vectors into  single dimension
    tsne = TSNE(n_components=2)
    points = tsne.fit_transform(normalized)


    proj = np.random.uniform(0, 1, (2, 3))
    colors = points @ proj
    colors -= colors.min()
    colors /= (colors.max() + 1e-8)
    colors *= 255
    colors = colors.astype(np.uint8)
    colors = [f'rgb({c[0]} {c[1]} {c[2]})' for c in colors]

    # t = np.array(times) / total_seconds
    # points = np.concatenate([points.reshape((-1, 1)), t.reshape((-1, 1))], axis=-1)[:, ::-1]

    return points, times, colors


def load_model(wavetable_device: str = 'gpu') -> nn.Module:
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
            wavetable_device=device,
            fine_positioning=True,
            fft_resonance=True
        )).to(device)

    with open('iterativedecomposition8.dat', 'rb') as f:
        model.load_state_dict(torch.load(f))

    # model = model.eval()

    print('Total parameters', count_parameters(model))
    print('Encoder parameters', count_parameters(model.encoder))
    print('Decoder parameters', count_parameters(model.resonance))

    return model


# def scatterplot_section(logger: Logger, events, points, times, colors) -> ScatterPlotComponent:
#     events = events.view(-1, n_samples)
#
#     events = {f'event{i}': events[i: i + 1, :] for i in range(events.shape[0])}
#
#     scatterplot_srcs = []
#
#     event_components = {}
#     for k, v in events.items():
#         _, e = logger.log_sound(k, v)
#         scatterplot_srcs.append(e.public_uri)
#         event_components[k] = AudioComponent(e.public_uri, height=35, controls=False)
#
#     scatterplot_component = ScatterPlotComponent(
#         scatterplot_srcs,
#         width=500,
#         height=500,
#         radius=0.3,
#         points=points,
#         times=times,
#         colors=colors, )
#
#     return scatterplot_component


def streaming_section(logger: Logger) -> CompositeComponent:
    model = load_model()
    samples = get_one_audio_segment(n_samples * 8, samplerate).view(1, 1, -1).to(device)
    # samples = max_norm(samples)

    with torch.no_grad():
        recon, event_vectors, times, events = model.streaming(samples, return_event_vectors=True)
        # recon = model.streaming(samples, return_event_vectors=False)
        recon = max_norm(recon)

    # points, times, colors = process_events(event_vectors, times, n_seconds)

    # scatter = scatterplot_section(logger, events, points, times, colors)

    _, orig = logger.log_sound(key='streamingorig', audio=samples)
    orig = AudioComponent(orig.public_uri, height=100, controls=True, scale=4, samples=1024)

    _, recon = logger.log_sound(key='streamingrecon', audio=recon)
    recon = AudioComponent(recon.public_uri, height=100, controls=True, scale=4, samples=1024)

    # TODO: audio timeline version

    return CompositeComponent(
        orig=orig,
        recon=recon,
        # scatter=scatter
    )


"""[markdown]

# Notes

This blog post is generated from a
[Python script](https://github.com/JohnVinyard/matching-pursuit/blob/main/v3blogpost.py) using
[conjure](https://github.com/JohnVinyard/conjure).

"""


def demo_page_dict() -> Dict[str, any]:
    print(f'Generating article...')

    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    logger = Logger(remote)

    print('Creating streaming section')
    streaming = streaming_section(logger)

    citation = CitationComponent(
        tag='johnvinyarditerativedecompositionv3',
        author='Vinyard, John',
        url='https://blog.cochlea.xyz/iterative-decomposition-v7.html',
        header='Iterative Decomposition V7',
        year='2024',
    )

    return dict(
        streaming=streaming,
        citation=citation
    )


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(
        __file__,
        'html',
        title='Streaming Iterative Decomposition Model',
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
