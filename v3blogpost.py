"""[markdown]

# Iterative Decomposition V3 Model

This article covers the continuation of work I've been pursuing in the area of sparse, interpretable audio models.

Some previous iterations of this work:
    - [Iterative Decomposition Model V2](https://blog.cochlea.xyz/siam.html)
    - [Gaussian/Gamma Splatting for Audio](https://blog.cochlea.xyz/gamma-audio-splat.html)


All training and model code can be
[found here](https://github.com/JohnVinyard/matching-pursuit/blob/main/iterativedecomposition.py).

This blog post is generated from a
[Python script](https://github.com/JohnVinyard/matching-pursuit/blob/main/v3blogpost.py) using
[conjure](https://github.com/JohnVinyard/conjure).

Our goal is to decompose a musical audio signal into a small number of "events", roughly analogous to a musical score,
but carrying information about the resonant characteristics of the instrument being played, and the room it is being
played in.  Each event is represented by a low-dimensional (16, in this case) vector and a time at which the event
occurs in the "score".

We seek to achieve this goal by iteratively guessing at the next-most informative event, removing it from the original
signal, and repeating the process, until no informative/interesting signal is remaining.

This is very similar to the [matching pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) algorithm, where we
repeatedly convolve an audio signal with a dictionary of audio atoms, picking the most highly-correlated atom at each
step, removing it, and the repeating the process until the norm of the original signal reaches some acceptable
threshold.

## The Algorithm

In this work, we replace the convolution of a large dictionary of audio "atoms" with analysis via a "deep" neural
network, which uses an STFT transform followed by
[cascading dilated convolutions](https://github.com/JohnVinyard/matching-pursuit/blob/main/iterativedecomposition.py#L86)
to efficiently analyze relatively long audio segments.  At each iteration, it proposes and **event vector** and a
**time-of-occurrence**.  This event is then rendered by an "event generator" network, "scheduled", by convolving the
rendered event with a dirac function (unit-valued spike) at the desired time.  It is subtracted from the original audio
signal, and the process is repeated.  During training, this process runs for a fixed number of steps, but it's possible
to imagine a modification whereby some other stopping condition is observed to improve efficiency.

The decoder makes some simple physics-based assumptions about the underlying signal, and uses convolutions with long
kernels to model the transfer functions of the instruments and the rooms in which they are performed.

## The Training Process

We train the model on ~three-second segments of audio from the
[MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA), which represents approximately 33 hours of
public-domain classical music.  We optimize the model via gradient descent using the following training objectives:

1. An iterative reconstruction loss, which asks the model to maximize the energy it removes from the signal at each step
2. A sparsity loss, which asks the model to minimize the l1 norm of the event vectors, ideally leading to a sparse (few event) solution
3. An adversarial loss, which masks about 50% of events and asks a discriminator network (trained in parallel) to
   judge them;  this is intended to encourage the events to be independent and stand on their own as believable, musical
   events.


## Improvements over the Previous Model

While the previous model only operated on around 1.5 seconds of audio, this model doubles that window, ultimately
driving toward a fully streaming algorithm that can handle signals of arbitrary length.  It also makes progress toward
a much simpler decoder, which generates each event as a linear combination of lookup tables for the following elements:

- a noisy impulse, or injection of energy into the system
- a number resonances, built by combining sine, sawtooch, triangle and square waves
- an interpolation between the resonances, representing the deformation of the system/instrument being played
   (e.g, the bending of a violin string as vibrato)
- a pre-baked room impulse response, which is, in fact, just another transfer function, this time for the entire room
   or space in which the piece is played

# Future Work


## Model Size, Training Time and Dataset

Firstly, this model is relatively small, weighing in at 18M parameters (82 MB on disk) and has only been trained for
around 24 hours, so it seems there is a lot of space to increase the model size, dataset size and training time to
further improve.  The reconstruction quality of the examples on this page is not amazing, certainly not good enough
even for a lossy audio codec, but the structure the model extracts seems like it could be used for many interesting
applications.

## Streaming and/or Arbitrary Lengths

Ultimately, the model should be able to handle audio segments of arbitrary lengths, adhering to some event "budget" to
find the sparsest-possible explanation of the audio segment.

## A Better Sparsity Loss

Some of the examples lead me to believe that my current sparsity loss is too aggressive;  the model sometimes prefers
to leave events out entirely rather than get the "win" of reducing overall signal energy.  Using the l1 norm penalty
seems like a sledgehammer, and a more nuanced loss would probably do better.

## Different Event Generator Variants

The decoder side of the model is very interesting, and all sorts of physical modelling-like approaches could yield
better, more realistic, and sparser renderings of the audio.

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

### Event Vectors

"""

# example_2.latents

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

### Event Vectors

"""

# example_3.latents

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

### Event Vectors

"""

# example_4.latents

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

### Event Vectors

"""

# example_5.latents

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


from typing import Dict

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
from conjure import S3Collection, \
    conjure_article, CitationComponent, numpy_conjure, AudioComponent, ImageComponent, \
    CompositeComponent, Logger
from argparse import ArgumentParser
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
        event_components[k] = AudioComponent(e.public_uri, height=35, controls=False)

    _, event_vectors = logger.log_matrix('latents', vectors[0].T)
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
    example_2 = reconstruction_section(logger)
    example_3 = reconstruction_section(logger)
    example_4 = reconstruction_section(logger)
    example_5 = reconstruction_section(logger)

    @numpy_conjure(remote)
    def fetch_audio(url: str, start_sample: int) -> np.ndarray:
        return get_audio_segment(
            url,
            target_samplerate=samplerate,
            start_sample=start_sample,
            duration_samples=n_samples)

    citation = CitationComponent(
        tag='johnvinyarditerativedecompositionv3',
        author='Vinyard, John',
        url='https://blog.cochlea.xyz/iterative-decomposition-v3.html',
        header='Iterative Decomposition V3',
        year='2024'
    )

    return dict(
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
