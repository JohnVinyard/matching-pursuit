"""[markdown]

This work takes a slightly different approach to the problem of decomposing
audio into a sparse and interpretable representation.  It models the physical
system that produced a segment of "natural"
(i.e., produced by acoustic instruments) musical audio as a state-space-model and
attempts to find a sparse control signal for the system.  The control system
can be thought of as the energy injected into the system by a human musician,
corresponding roughly to a score, and the state-space model can be thought of
as the dynamics and resonances of the musical instrument and room in which it
was played.

If you're curious to jump ahead, [listen to examples first](#Examples)!

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

State-Space models look a whole lot like RNNs (recurrent neural networks) in that
they are auto-regressive and have a hidden/inner state vector that represents
something like the "memory" of the model.  In this example, I conceptualize the
hidden state or state vector as the stored energy of the resonant object.  A human
musician has injected energy into the system by striking, plucking, or dragging a bow
across a string and the instrument will store that energy and "leak" it out in pleasing
ways.

## Formula

Formally, state space models take the follwing form (in pseudocode)

First, we initialize the state/hidden vector

`state_vector = zeros(state_dim)`

Then, we transform the input and the _previous hidden state_ into a _new_ hidden state.

`state_vector = (state_vector * state_matrix) + (input * input_matrix)`

Finally, we map the hidden state and the input into a new output

 `output_vector = (state_vector * output_matrix) + (input * direct_matrix)`

This process is repeated until we have no more inputs to process.

## This Experiment

In this experiment, we'll build a model PyTorch that will learn the four matrices described above along with
a sparse control signal by "overfitting" the model to a single segment of {n_seconds:.2} seconds of audio from
my goto for acoustic musical signals, the [MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA) dataset.

It hasn't showed up in the code quite yet, but we'll be using [`conjure`](https://github.com/JohnVinyard/conjure) to 
monitor the training process while iterating on the code, and to generate this article once things have settled.

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

Now, for the good stuff!  We'll define our simple State-Space Model as an `nn.Module`-derived class
with four parameters corresponding to each of the four matrices.

"""


class SSM(nn.Module):
    """
    A state-space model
    """

    def __init__(self, control_plane_dim: int, input_dim: int, state_matrix_dim: int):
        super().__init__()
        self.state_matrix_dim = state_matrix_dim
        self.input_dim = input_dim
        self.control_plane_dim = control_plane_dim

        self.proj = nn.Parameter(
            torch.zeros(control_plane_dim, input_dim).uniform_(-0.01, 0.01)
        )

        self.state_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        self.input_matrix = nn.Parameter(
            torch.zeros(input_dim, state_matrix_dim).uniform_(-0.01, 0.01))

        self.output_matrix = nn.Parameter(
            torch.zeros(state_matrix_dim, input_dim).uniform_(-0.01, 0.01)
        )
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

This model wraps up an `SSM` instance, and also has a parameter for the sparse "control plane" that will serve
as the input energy for our resonant model.  I think of this as a time-series of vector that describe the different
ways that energy can be injected into the model, e.g., you might have individual dimensions representing different
keys on a piano, or strings on a cello.  

I don't expect the control signals learned here to be quite that clear-cut
and interpretable, but you might notice that the random audio samples produced using the learned models
do seem to disentangle some characteristics of the instruments being played!

"""

class OverfitControlPlane(nn.Module):
    """
    Includes parameter for control plane or control signal
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

    def random(self):
        cp = torch.zeros_like(self.control, device=self.control.device).bernoulli_(p=0.001)
        audio = self.forward(sig=cp)
        return max_norm(audio)

    def forward(self, sig=None):
        return self.ssm.forward(sig if sig is not None else self.control_signal)




"""[markdown]
# The training process

Here is an [inline link](https://www.example.com).  How does it look?

> Here is a block quote

"""

"""[markdown]

# Sparsity Loss

blah

## More about the sparsity loss 

"""




def transform(x: torch.Tensor):
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)


def perceptual_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    fake_spec = transform(recon)
    real_spec = transform(target)
    return torch.abs(fake_spec - real_spec).sum()


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    return torch.abs(c).sum() * 1e-5


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def construct_experiment_model(state_dict: Union[None, dict] = None) -> OverfitControlPlane:
    """
    Construct a randomly initialize `OverfitControlPlane` model, ready for training/overfitting
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

"""

# example_1

# example_2

# example_3

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
                loss = perceptual_loss(recon, target) + sparsity_loss(model.control_signal)
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
            random=random_audio,
            control_plane=control_plane
        )
        return result

    def train_model_and_produce_components(
            url: str,
            start_sample: int,
            n_iterations: int):

        result_dict = train_model_for_segment_and_produce_artifacts(
            url, start_sample, n_iterations)

        orig = AudioComponent(result_dict['orig'].public_uri, height=200, samples=512)
        recon = AudioComponent(result_dict['recon'].public_uri, height=200, samples=512)
        random = AudioComponent(result_dict['random'].public_uri, height=200, samples=512)
        control = ImageComponent(result_dict['control_plane'].public_uri, height=200)

        return dict(orig=orig, recon=recon, random=random, control=control)

    def train_model_and_produce_content_section(
            url: str,
            start_sample: int,
            n_iterations: int,
            number: int) -> CompositeComponent:

        component_dict = train_model_and_produce_components(url, start_sample, n_iterations)
        composite = CompositeComponent(
            f'## Example {number}',
            '### Original Audio',
            f'A random {n_seconds:.2f} seconds segment of audio drawn from the MusicNet dataset',
            component_dict['orig'],
            '### Reconstruction',
            f'Reconstruction of the original audio after overfitting the model for {n_iterations} iterations',
            component_dict['recon'],
            '### Random Audio',
            f'Signal produced by a random, sparse control signal after overfitting the model for {n_iterations} iterations',
            component_dict['random'],
            '### Control Signal',
            f'Sparse control signal for the original audio after overfitting the model for {n_iterations} iterations',
            component_dict['control']
        )
        return composite

    example_1 = train_model_and_produce_content_section(
        'https://music-net.s3.amazonaws.com/2358',
        start_sample=2**16,
        n_iterations=n_iterations,
        number=1
    )

    example_2 = train_model_and_produce_content_section(
        'https://music-net.s3.amazonaws.com/2296',
        start_sample=2**18,
        n_iterations=n_iterations,
        number=2
    )

    example_3 = train_model_and_produce_content_section(
        'https://music-net.s3.amazonaws.com/2391',
        start_sample=2**18,
        n_iterations=n_iterations,
        number=3
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
        citation=citation,
    )


def generate_demo_page(iterations: int = 500):
    display = demo_page_dict(n_iterations=iterations)
    conjure_article(
        __file__,
        'html',
        title='Learning Playable State-Space Models from Audio',
        **display)



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
            loss = perceptual_loss(recon, target) + sparsity_loss(model.control_signal)

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

Thanks for reading!

# Cite this Article

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
