"""[markdown]

This work attempts to reproduce a short segment of "natural" (i.e., produced by acoustic
instruments or physical objects in the world) audio by decomposing it into two distinct pieces:

1. A single-layer RNN simulating the resonances of the system
2. a sparse control signal, representing energy injected into the system.

The control signal can be thought of as roughly corresponding to a musical score, and the RNN
can be thought of as the dynamics/resonances of the musical instrument and the room in which it was played.

It's notable that in this experiment (unlike
[my other recent work](https://blog.cochlea.xyz/sparse-interpretable-audio-codec-paper.html)), **there is no learned "encoder"**.  We simply "overfit"
parameters to a single audio segment, by minimizing a combination of [reconstruction and sparsity losses](#Sparsity Loss).

As a sneak-peek, here's a novel sound created by feeding a random, sparse control signal into
a state-space model "extracted" from an audio segment.

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
window_size = 128

# the dimensionality of the control plane or control signal
control_plane_dim = 64

# the dimensionality of the state vector, or hidden state for the RNN
state_dim = 128

# the number of (batch, control_plane_dim, frames) elements allowed to be non-zero
n_active_sites = 256


"""[markdown]


## The Experiment

We'll build a [PyTorch](https://pytorch.org/) model that will learn a system's resonances, along with a sparse control 
signal, by "overfitting" the model to a single segment of ~12 seconds of audio drawn 
from my favorite source for acoustic musical signals, the 
[MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) dataset.

Even though we're _overfitting_ a single audio signal, the sparsity term serves 
as a 
[regularizer](https://www.reddit.com/r/learnmachinelearning/comments/w7yrog/what_regularization_does_to_a_machine_learning/) 
that still forces the model to generalize in some way.  Our working theory is that the control signal must be _sparse_, 
which places certain constraints on the type of matrices the model must learn to accurately reproduce the audio.  If I
strike a piano key, the sound does not die away immediately and I do not have to continue to "drive" the sound by
continually injecting energy;  the strings and the body of the piano continue to resonate for quite some time. 

In this experiment, we use the l0 norm for the sparsity loss, and a straight-through estimator so that it remains
roughly differentiable.
 

While it hasn't showed up in the code we've seen so far, but we'll be using 
[`conjure`](https://github.com/JohnVinyard/conjure) to monitor the training process while iterating on the code, and 
eventually to generate this article once things have settled.

We'll start with some boring imports.

"""

from typing import Dict, Union

import numpy as np
import torch
from torch import nn
from itertools import count

from data import get_one_audio_segment
from modules import max_norm, flattened_multiband_spectrogram, sparsify
from torch.optim import Adam

from util import device, encode_audio, make_initializer

from conjure import logger, LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer, S3Collection, \
    conjure_article, CitationComponent, AudioComponent, ImageComponent, \
    CompositeComponent, Logger
from torch.nn.utils.clip_grad import clip_grad_value_
from argparse import ArgumentParser
from modules.infoloss import CorrelationLoss
from torch.nn import functional as F

remote_collection_name = 'state-space-model-demo-2'

"""[markdown]

# The `SSM` Class

Now, for the good stuff!  We'll define our simple State-Space Model as an 
[`nn.Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)-derived class with four parameters 
corresponding to each of the four matrices.

Note that there is a slight deviation from the canonical SSM in that we have a fifth matrix, which projects from our
"control plane" for the instrument into the dimension of a single audio frame.

"""

init_weights = make_initializer(0.05)


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

        self.net = nn.RNN(
            input_size=input_dim,
            hidden_size=state_matrix_dim,
            num_layers=1,
            nonlinearity='tanh',
            bias=False,
            batch_first=True)

        print(self.net)

        # matrix mapping control signal to audio frame dimension
        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        ) // 8

        self.out_proj = nn.Linear(state_dim, window_size, bias=False)


        self.apply(init_weights)


    def forward(self, control: torch.Tensor) -> torch.Tensor:
        """
        (batch, control_plane, time) -> (batch, window_size, time)
        """

        batch, cpd, frames = control.shape
        assert cpd == self.control_plane_dim

        control = control.permute(0, 2, 1)

        # try to ensure that the input signal only includes low-frequency info
        proj = control @ self.proj
        # proj = F.interpolate(proj, size=self.input_dim, mode='linear')
        # proj = proj * torch.zeros_like(proj).uniform_(-1, 1)
        # proj = proj * torch.hann_window(self.input_dim, device=proj.device)

        assert proj.shape == (batch, frames, self.input_dim)
        result, hidden = self.net.forward(proj)

        result = self.out_proj(result)

        result = result.view(batch, 1, -1)
        result = torch.sin(result)
        return result



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

# thanks to https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class OverfitControlPlane(nn.Module):
    """
    Encapsulates parameters for control signal and state-space model
    """

    def __init__(
            self,
            control_plane_dim: int,
            input_dim: int,
            state_matrix_dim: int,
            n_samples: int):
        
        super().__init__()
        self.ssm = SSM(control_plane_dim, input_dim, state_matrix_dim)
        self.n_samples = n_samples
        self.n_frames = n_samples // input_dim

        self.control = nn.Parameter(
            torch.zeros(1, control_plane_dim, self.n_frames).uniform_(0, 1))

    @property
    def control_signal_display(self) -> np.ndarray:
        return self.control_signal.data.cpu().numpy().reshape((-1, self.n_frames))

    @property
    def control_signal(self) -> torch.Tensor:
        s = sparsify(self.control, n_to_keep=n_active_sites)
        return torch.relu(s)

    # TODO: This should depend on the time-dimension alone
    def random(self, p=0.0001):
        """
        Produces a random, sparse control signal, emulating short, transient bursts
        of energy into the system modelled by the `SSM`
        """
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=p)
        audio = self.forward(sig=cp)
        return max_norm(audio)

    def rolled_control_plane(self):
        indices = torch.randperm(control_plane_dim)
        cp = self.control_signal[:, indices, :]
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
    return torch.abs(c).sum() * 1e-2


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

"""

# example_1

# example_2

# example_3

# example_4

# example_5


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

    def train_model_for_segment(
            target: torch.Tensor,
            iterations: int):

        loss_model = CorrelationLoss().to(device)

        while True:
            model = construct_experiment_model()
            optim = Adam(model.parameters(), lr=1e-3)

            for iteration in range(iterations):
                optim.zero_grad()
                recon = model.forward()
                loss = \
                    loss_model.multiband_noise_loss(target, recon, window_size=32, step=16) \
                    + reconstruction_loss(recon, target)

                if torch.isnan(loss).any():
                    print(f'detected NaN at iteration {iteration}')
                    break

                loss.backward()
                clip_grad_value_(model.parameters(), 0.5)
                optim.step()
                print(iteration, loss.item())

            if iteration < n_iterations - 1:
                print('NaN detected, starting anew')
                continue

            # total SSM parameters
            model_param_count = count_parameters(model.ssm)
            # non-zero control plane parameters
            non_zero = torch.sum(model.control_signal > 0)
            total_params = model_param_count + non_zero
            compression_ratio = total_params / n_samples

            print('COMPRESSION RATIO', compression_ratio * 100)
            break


        return model.state_dict()

    def encode(arr: np.ndarray) -> bytes:
        return encode_audio(arr)

    conj_logger = Logger(remote)

    # define loggers
    audio_logger = logger(
        'audio', 'audio/wav', encode, remote)

    def train_model_for_segment_and_produce_artifacts(n_iterations: int):


        audio_tensor = get_one_audio_segment(n_samples).view(1, 1, n_samples)
        audio_tensor = max_norm(audio_tensor)
        state_dict = train_model_for_segment(audio_tensor, n_iterations)
        hydrated = construct_experiment_model(state_dict)

        with torch.no_grad():
            recon = hydrated.forward()
            random = hydrated.random()
            rolled = hydrated.rolled_control_plane()

        _, orig_audio = conj_logger.log_sound('orig', audio_tensor)
        _, recon_audio = audio_logger.result_and_meta(recon)
        _, random_audio = audio_logger.result_and_meta(random)
        _, rolled_audio = audio_logger.result_and_meta(rolled)
        _, control_plane = conj_logger.log_matrix_with_cmap('controlplane', hydrated.control_signal[0], cmap='hot')

        result = dict(
            orig=orig_audio,
            recon=recon_audio,
            control_plane=control_plane,
            random=random_audio,
            rolled=rolled_audio
        )
        return result

    def train_model_and_produce_components(n_iterations: int):

        """
        Produce artifacts/media for a single example section
        """

        result_dict = train_model_for_segment_and_produce_artifacts(n_iterations)

        orig = AudioComponent(result_dict['orig'].public_uri, height=200, samples=512)
        recon = AudioComponent(result_dict['recon'].public_uri, height=200, samples=512)
        random = AudioComponent(result_dict['random'].public_uri, height=200, samples=512)
        rolled = AudioComponent(result_dict['rolled'].public_uri, height=200, samples=512)
        control = ImageComponent(result_dict['control_plane'].public_uri, height=200)

        return dict(
            orig=orig,
            recon=recon,
            control=control,
            random=random,
            rolled=rolled)


    def train_model_and_produce_content_section(n_iterations: int, number: int) -> CompositeComponent:

        """
        Produce a single "Examples" section for the post
        """

        component_dict = train_model_and_produce_components(n_iterations)
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

            rolled_header='### Random Permutation Along Control Plane Dimension',
            rolled_text=f'',
            rolled_component=component_dict['rolled'],

            control_header='### Control Signal',
            control_text=f'Sparse control signal for the original audio after overfitting the model for {n_iterations} iterations',
            control_component=component_dict['control'],
        )
        return composite

    example_1 = train_model_and_produce_content_section(
        n_iterations=n_iterations,
        number=1
    )

    example_2 = train_model_and_produce_content_section(
        n_iterations=n_iterations,
        number=2
    )

    example_3 = train_model_and_produce_content_section(
        n_iterations=n_iterations,
        number=3
    )

    example_4 = train_model_and_produce_content_section(
        n_iterations=n_iterations,
        number=4
    )

    example_5 = train_model_and_produce_content_section(
        n_iterations=n_iterations,
        number=5
    )

    citation = CitationComponent(
        tag='johnvinyardstatespacemodels',
        author='Vinyard, John',
        url='https://blog.cochlea.xyz/ssm.html',
        header='RNN Resonance Modelling for Sparse Decomposition of Audio',
        year='2024'
    )

    return dict(
        example_1=example_1,
        example_2=example_2,
        example_3=example_3,
        example_4=example_4,
        example_5=example_5,
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

    envelopes, = loggers(
        ['envelopes'],
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
    ], port=9999, n_workers=1)

    loss_model = CorrelationLoss().to(device)

    def train(target: torch.Tensor):
        model = construct_experiment_model()

        optim = Adam(model.parameters(), lr=1e-3)

        for iteration in count():
            optim.zero_grad()
            recon = model.forward()
            recon_audio(max_norm(recon))
            loss = \
                loss_model.multiband_noise_loss(target, recon, window_size=32, step=16) \
                + reconstruction_loss(recon, target)

            envelopes(model.control_signal.view(control_plane_dim, -1))
            loss.backward()

            clip_grad_value_(model.parameters(), 0.5)

            optim.step()
            print(iteration, loss.item())

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
   experiments?](https://blog.cochlea.xyz/sparse-interpretable-audio-codec-paper.html)
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
