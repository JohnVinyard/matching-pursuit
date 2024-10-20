"""[markdown]

In this article, we explore an audio transformation which is perceptually-inspired and maintains fine-grained
audio information while remaining invariant to imperceptible phase shifts.

TODO: relationship to the Auditory Image Model and to Mallat's scattering transform

https://www.acousticscale.org/wiki/index.php/AIM2006_Documentation

https://www.di.ens.fr/data/scattering/

https://www.di.ens.fr/data/scattering/audio/

"""

from modules import gammatone_filter_bank, max_norm, stft, rectified_filter_bank
from modules.aim import auditory_image_model
from modules.overfitraw import OverfitRawAudio
from util import device
from torch.optim import Adam
from typing import Callable
from conjure import S3Collection, Logger, numpy_conjure, conjure_article, AudioComponent, ImageComponent
from data import get_audio_segment
import numpy as np
import torch
from argparse import ArgumentParser
from torch.nn import functional as F


"""[markdown]

For this experiment, we'll just be using a single piece of source audio from the 
[MusicNet dataset](https://zenodo.org/records/5120004#.Yhxr0-jMJBA).

We'll explore what different audio transformations "hear" by overfitting raw audio samples to minimize
the loss between the transform of the original audio and the transform of the raw audio samples.
"""

# source


collection = S3Collection(
    'phase-invariant-feature',
    is_public=True,
    cors_enabled=True)

logger = Logger(collection)

samplerate = 22050
n_samples = 2 ** 17


@numpy_conjure(collection)
def fetch_audio(url: str, start_sample: int) -> np.ndarray:
    return get_audio_segment(
        url,
        target_samplerate=samplerate,
        start_sample=start_sample,
        duration_samples=n_samples)


AudioTransform = Callable[[torch.Tensor], torch.Tensor]


"""[markdown]
# The Magnitude Spectrogram

First, we explore the most commonly-used feature for audio loss functions
"""


def spectrogram(
        audio: np.ndarray,
        window_size: int = 2048,
        step_size: int = 256,
        mag_only: bool = False,
        normalize: bool = False) -> np.ndarray:

    n_coeffs = window_size // 2 + 1

    audio = torch.from_numpy(audio).view(1, 1, audio.shape[-1])
    spec = stft(audio, ws=window_size, step=step_size, pad=True)
    spec = spec.data.cpu().numpy()
    spec = spec.reshape((-1, n_coeffs)).T

    if mag_only:
        mag = np.abs(spec)
        if normalize:
            mag /= mag.std(axis=1, keepdims=True)
        return mag

    return spec


# mag_spec


"""[markdown]

## Reconstruction from the Magnitude Spectrogram

"""

def reconstruct_with_transform(
    target: np.ndarray,
    iterations: int,
    transform: AudioTransform,
) -> np.ndarray:

    target = torch.from_numpy(target).float().to(device).view(1, 1, target.shape[-1])
    target = max_norm(target)
    real_repr = transform(target)
    model = OverfitRawAudio((1, 1, target.shape[-1]), normalize=False).to(device)
    optim = Adam(model.parameters(), lr=1e-2)

    for i in range(iterations):
        optim.zero_grad()
        recon = model.forward(None)
        fake_repr = transform(recon)
        loss = F.mse_loss(fake_repr, real_repr)
        loss.backward()
        optim.step()
        print(i, loss.item())

    final = model.forward(None)
    final = final.data.cpu().numpy()
    return final


def reconstruct_from_mag_spectrogram(
        target: np.ndarray,
        iterations: int,
        window_size: int,
        step_size: int) -> np.ndarray:

    def transform(signal: torch.Tensor) -> torch.Tensor:
        return stft(signal, ws=window_size, step=step_size, pad=True)

    result = reconstruct_with_transform(target, iterations, transform)
    return result


def reconstruct_from_aim(
        target: np.ndarray,
        iterations: int,
        filter_bank: torch.Tensor,
        window_size: int,
        step_size: int):

    def transform(signal: torch.Tensor) -> torch.Tensor:
        return auditory_image_model(signal, filter_bank, window_size, step_size)

    result = reconstruct_with_transform(target, iterations, transform)
    return result


# mag_spec_recon

"""[markdown]

## Reconstruction with Longer Windows and Shorter Step Size

"""

# better_display

"""[markdown]

## Reconstruction with AIM-like feature


"""

# aim

# spec_display


def check_sparse(audio_example: np.ndarray, filter_bank: torch.Tensor) -> np.ndarray:
    audio_example = torch.from_numpy(audio_example).float().to(device).view(1, 1, audio_example.shape[-1])

    n_filters = 128
    window_size = 256
    aim_step_size = 64

    spec = rectified_filter_bank(audio_example, filter_bank)

    spec = spec.data.cpu().numpy()
    spec = spec.reshape((n_filters, -1))[:, :2048]
    return spec


def generate_page_dict(iterations: int = 1000) -> dict:

    # source audio ===================================
    # fetch the source audio
    audio_example = fetch_audio(
        'https://music-net.s3.amazonaws.com/2112',
        start_sample=samplerate * 30)
    audio_example /= (audio_example.max() + 1e-8)
    # encode it as a wav file
    encoded_audio, audio_meta = logger.log_sound('source-audio', audio_example)
    # create a component for display
    audio_display = AudioComponent(audio_meta.public_uri, height=200)

    # mag spectrogram ====================================
    # compute the magnitude spectrogram
    spec = spectrogram(audio_example, mag_only=True)
    # encode as an image
    img_data, spec_meta = logger.log_matrix_with_cmap(
        'magnitude-spectrogram',
        np.flipud(spec),
        cmap='hot')
    # create a component to display the spec
    spec_display = ImageComponent(spec_meta.public_uri, height=400)

    # mag spectrogram reconstruction ==========================
    mag_spec_recon = reconstruct_from_mag_spectrogram(
        audio_example, iterations, window_size=512, step_size=256)
    encoded_mag_spec_recon, encoded_recon_meta = logger.log_sound('mag-spec-recon', mag_spec_recon)
    mag_spec_recon_display = AudioComponent(encoded_recon_meta.public_uri, height=400)

    # better recon
    better = reconstruct_from_mag_spectrogram(audio_example, iterations, window_size=2048, step_size=256)
    encoded_better, better_meta = logger.log_sound('mag-spec-recon-better', better)
    better_display = AudioComponent(better_meta.public_uri, height=400)

    # aim recon ====================================================

    n_filters = 128
    window_size = 256
    aim_step_size = 64

    fb = gammatone_filter_bank(
        n_filters=n_filters, size=256, device=device, band_spacing='geometric')
    with_aim = reconstruct_from_aim(
        audio_example, iterations, fb, window_size=window_size, step_size=aim_step_size)
    encoded_aim, aim_meta = logger.log_sound('aim-recon', with_aim)
    aim_display = AudioComponent(aim_meta.public_uri, height=400)


    # sparse
    spec = check_sparse(audio_example, fb)
    encoded_spec, spec_meta = logger.log_matrix_with_cmap('spec', spec, cmap='hot')
    spec_display = ImageComponent(spec_meta.public_uri, height=400)

    # auditory image model ========================================
    # ta = torch.from_numpy(audio_example).float().to(device).view(1, 1, -1)
    # aim = auditory_image_model(ta, fb, aim_window_size=window_size, aim_step_size=aim_step_size)
    # batch, channels, time, periodicity = aim.shape
    #
    #
    # aim = aim.view(channels, time, periodicity).permute(1, 2, 0)
    # aim = aim / (aim.max() + 1e-8)
    # print('creating aim movie')
    # aim_movie, aim_meta = logger.log_movie('aim', aim)
    # print('done creating movie')
    # aim_movie = ImageComponent(aim_meta.public_uri, height=400)


    return dict(
        source=audio_display,
        mag_spec=spec_display,
        mag_spec_recon=mag_spec_recon_display,
        better_display=better_display,
        aim=aim_display,
        spec_display=spec_display
        # aim_movie=aim_movie
    )

def generate_article(iterations: int = 5000):
    page_components = generate_page_dict(iterations=iterations)
    conjure_article(
        __file__,
        'html',
        title="Phase Invariant Feature",
        max_depth=1,
        **page_components)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--clear',
        action='store_true',
        required=False,
        default=False)

    args = parser.parse_args()
    if args.clear:
        collection.destroy()
    else:
        generate_article(iterations=10)
