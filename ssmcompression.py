
# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 18

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

# the size of each, half-lapped audio "frame"
window_size = 512

# the dimensionality of the control plane or control signal
control_plane_dim = 64

# the dimensionality of the state vector, or hidden state
state_dim = 64


from argparse import ArgumentParser
from itertools import count
from typing import Union

import torch
from torch.nn.utils.clip_grad import clip_grad_value_
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, SupportedContentType, loggers, \
    NumpySerializer, NumpyDeserializer
from data import get_one_audio_segment
from modules import max_norm, flattened_multiband_spectrogram, OverfitControlPlane, SSM, stft
from util import device, encode_audio


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

# def transform(x: torch.Tensor) -> torch.Tensor:
#     return stft(x, ws=2048, step=256, pad=True)


def reconstruction_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the difference between the `recon` and `target`
    representations
    """
    fake_spec = transform(recon)
    real_spec = transform(target)
    return torch.abs(fake_spec - real_spec).sum()


def sparsity_loss(c: torch.Tensor) -> torch.Tensor:
    """
    Compute the l1 norm of the control signal
    """
    return torch.abs(c).sum() * 1e-4


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
        n_samples=n_samples,
        windowed=False

    )
    model = model.to(device)

    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model


def compute_compression_ratio(
        n_samples: int,
        model: SSM,
        control_plane: torch.Tensor,
        sparsity: float) -> float:

    total_original_params = n_samples

    model_params = model.parameter_count
    nonzero = control_plane.numel() * sparsity
    total_model_params = model_params + nonzero
    ratio = total_model_params / total_original_params
    return ratio

def train_and_monitor(pattern: str):
    print(pattern)

    print(f'training on {n_seconds} of audio')

    target = get_one_audio_segment(
        n_samples=n_samples, samplerate=samplerate, pattern=pattern)

    collection = LmdbCollection(path='ssmcompression')

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
            ratio = compute_compression_ratio(
                n_samples, model.ssm, model.control_signal, sparsity)

            state_space(model.ssm.state_matrix)
            envelopes(model.control_signal.view(control_plane_dim, -1))
            loss.backward()

            clip_grad_value_(model.parameters(), 0.1)

            optim.step()
            print(iteration, loss.item(), sparsity, ratio)

            if iteration % 25 == 0:
                with torch.no_grad():
                    rnd = model.random()
                    random_audio(rnd)

    train(target)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--pattern', type=str, required=False, default='*1755.wav')
    args = parser.parse_args()
    train_and_monitor(args.pattern)

