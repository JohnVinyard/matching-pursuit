"""[markdown]

Hello

"""

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

### Event Vectors
 
"""

# example_1.latents

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


from typing import Dict, Union

from modules.eventgenerators.overfitresonance import OverfitResonanceModel


# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 16
samples_per_event = 2048
n_events = n_samples // samples_per_event

context_dim = 16

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256

n_frames = n_samples // transform_step_size



import numpy as np
import torch

from data import get_one_audio_segment, get_audio_segment

from conjure import logger, LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer, S3Collection, \
    conjure_article, CitationComponent, numpy_conjure, AudioComponent, pickle_conjure, ImageComponent, \
    CompositeComponent, Logger
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from iterativedecomposition import Model as IterativeDecompositionModel


remote_collection_name = 'iterative-decomposition-v3'


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()

# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reconstruction_section(logger: Logger) -> CompositeComponent:
    hidden_channels = 512

    model = IterativeDecompositionModel(
        in_channels=1024,
        hidden_channels=hidden_channels,
        resonance_model=OverfitResonanceModel(
                n_noise_filters=32,
                noise_expressivity=8,
                noise_filter_samples=128,
                noise_deformations=16,
                instr_expressivity=8,
                n_events=1,
                n_resonances=2048,
                n_envelopes=128,
                n_decays=32,
                n_deformations=32,
                n_samples=n_samples,
                n_frames=n_frames,
                samplerate=samplerate,
                hidden_channels=hidden_channels,
                wavetable_device='cpu'
            ))

    with open('iterativedecomposition.dat', 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))
        # model = model.to(device)

    print('Total parameters', count_parameters(model))
    print('Encoder parameters', count_parameters(model.encoder))
    print('Decoder parameters', count_parameters(model.resonance))

    # get a random audio segment
    samples = get_one_audio_segment(n_samples, samplerate, device='cpu').view(1, 1, n_samples)
    events, vectors, times = model.iterative(samples)
    print(events.shape)

    # sum together all events
    summed = torch.sum(events, dim=1, keepdim=True)

    _, original = logger.log_sound(f'original', samples)
    _, reconstruction = logger.log_sound(f'reconstruction', summed)

    orig_audio_component = AudioComponent(original.public_uri, height=200)
    recon_audio_component = AudioComponent(reconstruction.public_uri, height=200)


    events = {f'event{i}': events[:, i: i + 1, :] for i in range(events.shape[1])}

    event_components = {}
    for k, v in events.items():
        _, e = logger.log_sound(k, v)
        event_components[k] = AudioComponent(e.public_uri, height=50, controls=False)

    _, event_vectors = logger.log_matrix_with_cmap('latents', vectors[0], cmap='hot')
    latents = ImageComponent(event_vectors.public_uri, height=200, title='latent event vectors')

    composite = CompositeComponent(
        orig_audio=orig_audio_component,
        recon_audio=recon_audio_component,
        latents=latents,
        **event_components
    )
    return composite

"""[markdown]

Thanks for reading!

"""


def demo_page_dict() -> Dict[str, any]:
    print(f'Generating article...')


    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    logger = Logger(remote)

    example_1 = reconstruction_section(logger)

    @numpy_conjure(remote)
    def fetch_audio(url: str, start_sample: int) -> np.ndarray:
        return get_audio_segment(
            url,
            target_samplerate=samplerate,
            start_sample=start_sample,
            duration_samples=n_samples)

    # def encode(arr: np.ndarray) -> bytes:
    #     return encode_audio(arr)
    #
    # def display_matrix(arr: Union[torch.Tensor, np.ndarray], cmap: str = 'gray') -> bytes:
    #     if arr.ndim > 2:
    #         raise ValueError('Only two-dimensional arrays are supported')
    #
    #     if isinstance(arr, torch.Tensor):
    #         arr = arr.data.cpu().numpy()
    #
    #     arr = arr * -1
    #
    #     bio = BytesIO()
    #     plt.matshow(arr, cmap=cmap)
    #     plt.axis('off')
    #     plt.margins(0, 0)
    #     plt.savefig(bio, pad_inches=0, bbox_inches='tight')
    #     plt.clf()
    #     bio.seek(0)
    #     return bio.read()

    # define loggers
    # audio_logger = logger(
    #     'audio', 'audio/wav', encode, remote)
    #
    # matrix_logger = logger(
    #     'matrix', 'image/png', display_matrix, remote)

    return dict(
        example_1=example_1,
    )


def generate_demo_page():
    display = demo_page_dict()
    conjure_article(
        __file__,
        'html',
        title='Learning "Playable" State-Space Models from Audio',
        **display)





"""[markdown]

# Conclusion


## Future Work


# Cite this Article

If you'd like to cite this article, you can use the following [BibTeX block](https://bibtex.org/).

"""

# citation


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
