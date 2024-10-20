"""[markdown]

This work attempts to reproduce a short segment of "natural" (i.e., produced by acoustic
instruments or physical objects in the world) audio by decomposing it into two distinct pieces:

1. A state-space model simulating the resonances of the system
2. a sparse control signal, representing energy injected into the system.

The control signal can be thought of as roughly corresponding to a musical score, and the state-space model
can be thought of as the dynamics/resonances of the musical instrument and the room in which it was played.

It's notable that in this experiment (unlike
[my other recent work](https://blog.cochlea.xyz/siam.html)), **there is no learned "encoder"**.  We simply "overfit"
parameters to a single audio sample, by minimizing a combination of [reconstruction and sparsity losses](#Sparsity Loss).

As a sneak-peek, here's a novel sound created by feeding a random, sparse control signal into
a state-space model "extracted" from an audio segment from Beethoven's "Piano Sonata No 15 in D major".

Feel free to [jump ahead](#Examples) if you're curious to hear all the audio examples first!

"""

# example_1.random_component

"""[markdown]

First, we'll set up high-level parameters for the experiment

"""

# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 18

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

# the size of each, half-lapped audio "frame"
window_size = 512

# the dimensionality of the control plane or control signal
control_plane_dim = 32

# the dimensionality of the state vector, or hidden state
state_dim = 128

"""[markdown]

# The Model

State-Space models look a lot like 
[RNNs (recurrent neural networks)](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks) 
in that they are auto-regressive and have a hidden/inner state vector that represents
something like the "memory" of the model.  In this example, I tend to think of the
hidden state as the stored energy of the resonant object.  A human
musician has injected energy into the system by striking, plucking, or dragging a bow and the instrument stores that 
energy and "leaks" it out in ways that are (hopefully) pleasing to the ear.

## Formula

Formally, state space models take the following form (in pseudocode)

First, we initialize the state/hidden vector

`state_vector = zeros(state_dim)`

Then, we transform the input and the _previous hidden state_ into a _new_ hidden state.  This is where the 
"auto-regressive" or recursive nature of the model comes into view;  notice that `state-vector` is on both sides of the 
equation.  **There's a feedback look happening here**, which is a hallmark of 
[waveguide synthesis](https://www.osar.fr/notes/waveguides/) and other physical modelling synthesis.

`state_vector = (state_vector * state_matrix) + (input * input_matrix)`

Finally, we map the hidden state and the input into a new output

 `output_vector = (state_vector * output_matrix) + (input * direct_matrix)`

This process is repeated until we have no more inputs to process.  The `direct_matrix` is a mapping from
inputs directly to the output vector, rather like a "skip connection" in other neural network architectures.

As long as we have something like conservation of energy happening (not enforced explicitly), it's easy to see how
the exponential decay we observe in resonant objects emerges from our model. 


## The Experiment

We'll build a [PyTorch](https://pytorch.org/) model that will learn the four matrices described 
above, along with a sparse control signal, by "overfitting" the model to a single segment of ~12 seconds of audio drawn 
from my favorite source for acoustic musical signals, the 
[MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) dataset.  For the final example, we'll try fitting
a different kind of "natural" acoustic signal, human speech, just for funsies!

Even though we're _overfitting_ a single audio signal, the sparsity term serves 
as a 
[regularizer](https://www.reddit.com/r/learnmachinelearning/comments/w7yrog/what_regularization_does_to_a_machine_learning/) 
that still forces the model to generalize in some way.  Our working theory is that the control signal must be _sparse_, 
which places certain constraints on the type of matrices the model must learn to accurately reproduce the audio.  If I
strike a piano key, the sound does not die away immediately and I do not have to continue to "drive" the sound by
continually injecting energy;  the strings and the body of the piano continue to resonate for quite some time.  

While it hasn't showed up in the code we've seen so far, but we'll be using 
[`conjure`](https://github.com/JohnVinyard/conjure) to monitor the training process while iterating on the code, and 
eventually to generate this article once things have settled.

We'll start with some boring imports.

"""

from io import BytesIO
from typing import Dict, Union

import numpy as np
import torch
from torch import nn
from itertools import count

from data import get_one_audio_segment, get_audio_segment
from modules import max_norm, flattened_multiband_spectrogram
from modules.overlap_add import overlap_add
from torch.optim import Adam

from util import device, encode_audio

from conjure import logger, LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer, S3Collection, \
    conjure_article, CitationComponent, numpy_conjure, AudioComponent, pickle_conjure, ImageComponent, \
    CompositeComponent
from torch.nn.utils.clip_grad import clip_grad_value_
from argparse import ArgumentParser
from matplotlib import pyplot as plt

remote_collection_name = 'state-space-model-demo'

"""[markdown]

# The `SSM` Class

Now, for the good stuff!  We'll define our simple State-Space Model as an 
[`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)-derived class with four parameters 
corresponding to each of the four matrices.

Note that there is a slight deviation from the canonical SSM in that we have a fifth matrix, which projects from our
"control plane" for the instrument into the dimension of a single audio frame.

"""


class SSM(nn.Module):
    """
    A state-space model-like module, with one additional matrix, used to project the control
    signal into the shape of each audio frame.

    The final output is produced by overlap-adding the windows/frames of audio into a single
    1D signal.
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int):
        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        # state matrix mapping previous state vector to next state vector
        self.state_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        # matrix mapping audio frame to hidden/state vector dimension
        self.input_matrix = nn.Parameter(
            torch.zeros(input_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        # matrix mapping hidden/state vector to audio frame dimension
        self.output_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, input_dim).uniform_(-0.01, 0.01)
        )

        # skip-connection-like matrix mapping input audio frame to next
        # output audio frame
        self.direct_matrix = nn.Parameter(
            torch.zeros(input_dim, input_dim).uniform_(-0.01, 0.01)
        )



    def forward(self, control: torch.Tensor) -> torch.Tensor:
        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)

        proj = control @ self.proj
        assert proj.shape == (batch, frames, self.input_dim)

        results = []
        state_vec = torch.zeros(batch, self.state_matrix_dim, device=control.device)

        for i in range(frames):
            inp = proj[:, i, :]
            state_vec = (state_vec @ self.state_matrix) + (inp @ self.input_matrix)
            output = (state_vec @ self.output_matrix) + (inp @ self.direct_matrix)
            results.append(output.view(batch, 1, self.input_dim))

        result = torch.cat(results, dim=1)
        result = result[:, None, :, :]

        result = overlap_add(result)
        return result[..., :frames * (self.input_dim // 2)]


"""[markdown]

# The `OverfitControlPlane` Class

This model encapsulates an `SSM` instance, and also has a parameter for the sparse "control plane" that will serve
as the input energy for our resonant model.  I think of this as a time-series of vectors that describe the different
ways that energy can be injected into the model, e.g., you might have individual dimensions representing different
keys on a piano, or strings on a cello.  

I don't expect the control signals learned here to be quite _that_ clear-cut
and interpretable, but you might notice that the random audio samples produced using the learned models
do seem to disentangle some characteristics of the instruments being played!

"""


class OverfitControlPlane(nn.Module):
    """
    Encapsulates parameters for control signal and state-space model
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int, n_samples: int):
        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim)
        self.n_samples = n_samples
        self.n_frames = int(n_samples / (input_dim // 2))

        self.control = nn.Parameter(
            torch.zeros(1, control_plane_dim, self.n_frames).uniform_(-0.01, 0.01))

    @property
    def control_signal_display(self) -> np.ndarray:
        return self.control_signal.data.cpu().numpy().reshape((-1, self.n_frames))

    @property
    def control_signal(self) -> torch.Tensor:
        return torch.relu(self.control)

    def random(self, p=0.001):
        """
        Produces a random, sparse control signal, emulating short, transient bursts
        of energy into the system modelled by the `SSM`
        """
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=p)
        audio = self.forward(sig=cp)
        return max_norm(audio)

    def forward(self, sig=None):
        """
        Inject energy defined by `sig` (or by the `control` parameters encapsulated by this class)
        into the system modelled by `SSM`
        """
        return self.ssm.forward(sig if sig is not None else self.control_signal)


"""[markdown]
# The Training Process

To train the `OverfitControlPlane` model, we randomly initialize parameters for `SSM` and the learned
control signal, and minimize a loss that consists of a reconstruction term and a sparsity term via gradient
descent.  For this experiment, we're using the [`Adam`](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html)
optimizer with a learning rate of `1e-2`.

"""

"""[markdown]

## Reconstruction Loss

The first loss term is a simple reconstruction loss, consisting of the l1 norm of the difference between
two multi-samplerate and multi-resolution spectrograms. 

"""


def transform(x: torch.Tensor):
    """
    Decompose audio into sub-bands of varying sample rate, and compute spectrogram with
    varying time-frequency tradeoffs on each band.
    """
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the difference between the `recon` and `target`
    representations
    """
    fake_spec = transform(recon)
    real_spec = transform(target)
    return torch.abs(fake_spec - real_spec).sum()


"""[markdown]

## Sparsity Loss

Ideally, we want the model to resonate, or store and "leak" energy slowly in the way that an
acoustic instrument might.  This means that the control signal is not dense and continually "driving" the instrument, 
but injecting energy infrequently in ways that encourage the natural resonances of the physical object.  

I'm not fully satisfied with this approach. e.g. it tends to pull away from what might be a nice, 
natural control signal for a violin or other bowed instrument.  In my mind, this might look like a sub-20hz sawtooth 
wave that would "drive" the string, continually catching and releasing as the bow drags across the string.

For now, the sparsity term _does_ seem to encourage models that resonate, but my intuition is that
there is a better, more nuanced approach that could handle bowed string instruments and wind instruments, 
in addition to percussive instruments, where this approach really seems to shine.
"""


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the control signal
    """
    return torch.abs(c).sum() * 1e-5


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def construct_experiment_model(state_dict: Union[None, dict] = None) -> OverfitControlPlane:
    """
    Construct a randomly initialized `OverfitControlPlane` instance, ready for training/overfitting
    """
    model = OverfitControlPlane(
        control_plane_dim=control_plane_dim,
        input_dim=window_size,
        state_matrix_dim=state_dim,
        n_samples=n_samples
    )
    model = model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


"""[markdown]

# Examples

Finally, some trained models to listen to!  Each example consists of the following:

1. the original audio signal from the MusicNet dataset
1. the sparse control signal for the reconstruction
1. the reconstructed audio, produced using the sparse control signal and the learned state-space model
1. a novel, random audio signal produced using the learned state-space model and a random control signal

Just for fun, we attempt to learn the fourth example from a speech signal from the 
[LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

"""

# example_1

# example_2

# example_3

# example_4

"""[markdown]

# Code For Generating this Article

What follows is the code used to train the model and produce the article you're reading.  It uses 
the [`conjure`](https://github.com/JohnVinyard/conjure) Python library, a tool I've been writing 
that helps to persist and display images, audio and other code artifacts that are interleaved throughout
this post.

"""


def demo_page_dict(n_iterations: int = 100) -> Dict[str, any]:
    print(f'Generating article, training models for {n_iterations} iterations')

    remote = S3Collection(
        remote_collection_name, is_public=True, cors_enabled=True)

    @numpy_conjure(remote)
    def fetch_audio(url: str, start_sample: int) -> np.ndarray:
        return get_audio_segment(
            url,
            target_samplerate=samplerate,
            start_sample=start_sample,
            duration_samples=n_samples)

    def train_model_for_segment(
            target: torch.Tensor,
            iterations: int):

        while True:
            model = construct_experiment_model()
            optim = Adam(model.parameters(), lr=1e-2)

            for iteration in range(iterations):
                optim.zero_grad()
                recon = model.forward()
                loss = reconstruction_loss(recon, target) + sparsity_loss(model.control_signal)
                non_zero = (model.control_signal > 0).sum()
                sparsity = (non_zero / model.control_signal.numel()).item()

                if torch.isnan(loss).any():
                    print(f'detected NaN at iteration {iteration}')
                    break

                loss.backward()
                clip_grad_value_(model.parameters(), 0.5)
                optim.step()
                print(iteration, loss.item(), sparsity)

            if iteration < n_iterations - 1:
                print('NaN detected, starting anew')
                continue

            return model.state_dict()

    def encode(arr: np.ndarray) -> bytes:
        return encode_audio(arr)

    def display_matrix(arr: Union[torch.Tensor, np.ndarray], cmap: str = 'gray') -> bytes:
        if arr.ndim > 2:
            raise ValueError('Only two-dimensional arrays are supported')

        if isinstance(arr, torch.Tensor):
            arr = arr.data.cpu().numpy()

        arr = arr * -1

        bio = BytesIO()
        plt.matshow(arr, cmap=cmap)
        plt.axis('off')
        plt.margins(0, 0)
        plt.savefig(bio, pad_inches=0, bbox_inches='tight')
        plt.clf()
        bio.seek(0)
        return bio.read()

    # define loggers
    audio_logger = logger(
        'audio', 'audio/wav', encode, remote)

    matrix_logger = logger(
        'matrix', 'image/png', display_matrix, remote)

    @pickle_conjure(remote)
    def train_model_for_segment_and_produce_artifacts(
            url: str,
            start_sample: int,
            n_iterations: int):

        print(f'Generating example for {url} with start sample {start_sample}')

        audio_array = fetch_audio(url, start_sample)
        audio_tensor = torch.from_numpy(audio_array).to(device).view(1, 1, n_samples)
        audio_tensor = max_norm(audio_tensor)
        state_dict = train_model_for_segment(audio_tensor, n_iterations)
        hydrated = construct_experiment_model(state_dict)

        with torch.no_grad():
            recon = hydrated.forward()
            random = hydrated.random()

        _, orig_audio = audio_logger.result_and_meta(audio_array)
        _, recon_audio = audio_logger.result_and_meta(recon)
        _, random_audio = audio_logger.result_and_meta(random)
        _, control_plane = matrix_logger.result_and_meta(hydrated.control_signal_display)

        result = dict(
            orig=orig_audio,
            recon=recon_audio,
            control_plane=control_plane,
            random=random_audio,
        )
        return result

    def train_model_and_produce_components(
            url: str,
            start_sample: int,
            n_iterations: int):

        """
        Produce artifacts/media for a single example section
        """

        result_dict = train_model_for_segment_and_produce_artifacts(
            url, start_sample, n_iterations)

        orig = AudioComponent(result_dict['orig'].public_uri, height=200, samples=512)
        recon = AudioComponent(result_dict['recon'].public_uri, height=200, samples=512)
        random = AudioComponent(result_dict['random'].public_uri, height=200, samples=512)
        control = ImageComponent(result_dict['control_plane'].public_uri, height=200)

        return dict(orig=orig, recon=recon,control=control, random=random)

    def train_model_and_produce_content_section(
            url: str,
            start_sample: int,
            n_iterations: int,
            number: int) -> CompositeComponent:

        """
        Produce a single "Examples" section for the post
        """

        component_dict = train_model_and_produce_components(url, start_sample, n_iterations)
        composite = CompositeComponent(
            header=f'## Example {number}',
            orig_header='### Original Audio',
            orig_text=f'A random {n_seconds:.2f} seconds segment of audio drawn from the MusicNet dataset',
            orig_component=component_dict['orig'],
            recon_header='### Reconstruction',
            recon_text=f'Reconstruction of the original audio after overfitting the model for {n_iterations} iterations',
            recon_component=component_dict['recon'],
            random_header='### Random Audio',
            random_text=f'Signal produced by a random, sparse control signal after overfitting the model for {n_iterations} iterations',
            random_component=component_dict['random'],
            control_header='### Control Signal',
            control_text=f'Sparse control signal for the original audio after overfitting the model for {n_iterations} iterations',
            control_component=component_dict['control']
        )
        return composite

    example_1 = train_model_and_produce_content_section(
        'https://music-net.s3.amazonaws.com/2358',
        start_sample=2 ** 16,
        n_iterations=n_iterations,
        number=1
    )

    example_2 = train_model_and_produce_content_section(
        'https://music-net.s3.amazonaws.com/2296',
        start_sample=2 ** 18,
        n_iterations=n_iterations,
        number=2
    )

    example_3 = train_model_and_produce_content_section(
        'https://music-net.s3.amazonaws.com/2391',
        start_sample=2 ** 18,
        n_iterations=n_iterations,
        number=3
    )

    example_4 = train_model_and_produce_content_section(
        'https://lj-speech.s3.amazonaws.com/LJ019-0120.wav',
        start_sample=0,
        n_iterations=n_iterations,
        number=4
    )

    citation = CitationComponent(
        tag='johnvinyardstatespacemodels',
        author='Vinyard, John',
        url='https://blog.cochlea.xyz/ssm.html',
        header='State Space Modelling for Sparse Decomposition of Audio',
        year='2024'
    )

    return dict(
        example_1=example_1,
        example_2=example_2,
        example_3=example_3,
        example_4=example_4,
        citation=citation,
    )


def generate_demo_page(iterations: int = 500):
    display = demo_page_dict(n_iterations=iterations)
    conjure_article(
        __file__,
        'html',
        title='Learning "Playable" State-Space Models from Audio',
        **display)


"""[markdown]

# Training Code

As I developed this model, I used the following code to pick a random audio segment, overfit a model, and monitor
its progress during training.

"""


def train_and_monitor():
    target = get_one_audio_segment(n_samples=n_samples, samplerate=samplerate)
    collection = LmdbCollection(path='ssm')

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random'],
        'audio/wav',
        encode_audio,
        collection)

    envelopes, state_space = loggers(
        ['envelopes', 'statespace'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
        envelopes,
        random_audio,
        state_space
    ], port=9999, n_workers=1)

    def train(target: torch.Tensor):
        model = construct_experiment_model()

        optim = Adam(model.parameters(), lr=1e-2)

        for iteration in count():
            optim.zero_grad()
            recon = model.forward()
            recon_audio(max_norm(recon))
            loss = reconstruction_loss(recon, target) + sparsity_loss(model.control_signal)

            non_zero = (model.control_signal > 0).sum()
            sparsity = (non_zero / model.control_signal.numel()).item()

            state_space(model.ssm.state_matrix)
            envelopes(model.control_signal.view(control_plane_dim, -1))
            loss.backward()

            clip_grad_value_(model.parameters(), 0.5)

            optim.step()
            print(iteration, loss.item(), sparsity)

            with torch.no_grad():
                rnd = model.random()
                random_audio(rnd)

    train(target)


"""[markdown]

# Conclusion

Thanks for reading this far!  

I'm excited about the results of this experiment, but am not totally pleased
with the frame-based approach, which leads to very noticeable artifacts in the reconstructions.  It **runs counter to
one of my guiding principles** as I try to design a sparse, interpretable, and easy-to-manipulate audio codec, which
is that there is no place for arbitrary, fixed-size "frames".  Ideally, we represent audio as a sparse set of events
or sound sources that are sample-rate independent, i.e., more like a function or operator, and less like a rasterized
representation.

I'm just beginning to learn more about state-space models and was excited when I learned from Albert Gu in his 
excellent talk ["Efficiently Modeling Long Sequences with Structured State Spaces"](https://youtu.be/luCBXCErkCs?si=rRSQ7af3X6cZRivW&t=1776) 
that there are ways to transform state-space models, which strongly resemble 
[IIR filters](https://en.wikipedia.org/wiki/Infinite_impulse_response), into their 
[FIR](https://en.wikipedia.org/wiki/Finite_impulse_response) counterpart, convolutions, which I've depended on 
heavily to model resonance in other recent work.

I'm looking forward to following this thread and beginning to find where the two different approaches converge! 

## Future Work

1. Instead of (or in addition to) a sparsity loss, could we build in more physics-informed losses, such as conservation
   of energy, i.e., overall energy can never _increase_ unless it comes from the control signal?
2. Could we use 
   [`scipy.signal.StateSpace`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html) to
   derive a continuous-time formulation of the model?
3. How would a model like this work as an event generator in my [sparse, interpretable audio model from other 
   experiments?](https://blog.cochlea.xyz/machine-learning/2024/02/29/siam.html)
4. Could we treat an entire, multi-instrument song as a single, large state-space model, learning a compressed 
   representation of the audio _and_ a "playable" instrument, all at the same time?


# Cite this Article

If you'd like to cite this article, you can use the following [BibTeX block](https://bibtex.org/).

"""

# citation


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--prefix', type=str, required=False, default='')
    args = parser.parse_args()

    if args.mode == 'train':
        train_and_monitor()
    elif args.mode == 'demo':
        generate_demo_page(args.iterations)
    elif args.mode == 'list':
        remote = S3Collection(
            remote_collection_name, is_public=True, cors_enabled=True)
        print('Listing stored keys')
        for key in remote.iter_prefix(start_key=args.prefix):
            print(key)
    elif args.mode == 'clear':
        remote = S3Collection(
            remote_collection_name, is_public=True, cors_enabled=True)
        remote.destroy(prefix=args.prefix)
    else:
        raise ValueError('Please provide one of train, demo, or clear')
