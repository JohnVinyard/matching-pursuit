"""[markdown]

In this article, we explore an audio transformation which is perceptually-inspired and maintains fine-grained
audio information while remaining invariant to imperceptible phase shifts.

TODO: relationship to the Auditory Image Model and to Mallat's scattering transform

https://www.acousticscale.org/wiki/index.php/AIM2006_Documentation

https://www.di.ens.fr/data/scattering/

https://www.di.ens.fr/data/scattering/audio/

"""
from modules import gammatone_filter_bank
from modules.aim import auditory_image_model
from modules.overfitraw import OverfitRawAudio
from util import device
from torch.optim import Adam
from typing import Tuple, Callable
from conjure import S3Collection, Logger, numpy_conjure, conjure_article, pickle_conjure, AudioComponent, ImageComponent
from data import get_audio_segment
import numpy as np
import torch
from argparse import ArgumentParser
from scipy.signal import stft
from matplotlib import cm
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

@numpy_conjure(collection)
def overfit_audio_samples(
        orig: torch.Tensor,
        transform: AudioTransform,
        n_iterations: int = 1000) -> np.ndarray:

    model = OverfitRawAudio((1, 1, n_samples), normalize=True).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    orig = orig.view(1, 1, n_samples)
    orig_transform = transform(orig)

    for i in range(n_iterations):
        optim.zero_grad()
        recon = model.forward(None)
        recon_transform = transform(recon)
        loss = F.mse_loss(recon_transform, orig_transform)
        loss.backward()
        optim.step()
        print(i, loss.item())

    return model.as_numpy_array()


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

    _, _, spec = stft(audio, nperseg=window_size, noverlap=window_size - step_size)
    coeffs, frames = spec.shape

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

"""[markdown]

## An Aside: Phase Manipulations

"""

"""[markdown]

# The RainbowGram

We visualize a feature that also displays frame-to-frame phase deltas
"""

def to_magnitude_and_phase_delta(
        audio: np.ndarray,
        window_size: int = 2048,
        step_size: int = 256,
        normalize: bool = False) -> Tuple[np.ndarray, np.ndarray]:

    spec = spectrogram(audio, window_size, step_size, normalize=normalize)

    mag = np.abs(spec)

    phase = np.angle(spec)
    phase_delta = np.gradient(phase, axis=-1)
    phase = phase_delta % (2 * np.pi)

    return mag, phase


def from_magnitude_and_phase_delta(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    accum = np.cumsum(phase, axis=-1)
    accum = (accum * np.pi) % (2 * np.pi) - np.pi
    spec = mag * torch.exp(1j * accum)
    return spec


def unit_normalize(x: np.ndarray) -> np.ndarray:
    return x / (x.max() + 1e-8)


def rainbowgram(audio: np.ndarray):
    mag, phase = to_magnitude_and_phase_delta(audio, normalize=True)
    mag = unit_normalize(mag)
    phase = unit_normalize(phase)
    rg = cm.rainbow(phase)[..., :3]
    rg *= mag[..., None]
    rg = unit_normalize(rg)
    return rg

# rainbowgram


"""[markdown]

## Reconstruction from the RainbowGram

"""


"""[markdown]
# Perceptual Results of Phase Transformations

"""

"""[markdown]

## Fully Randomized Phase

"""

"""[markdown]

## Per-Channel Random Phase Shifts

"""

"""[markdown]
# The Phase-Invariant Feature and its Benefits
 
"""

# aim

def generate_page_dict() -> dict:

    # source audio ===================================
    # fetch the source audio
    audio_example = fetch_audio(
        'https://music-net.s3.amazonaws.com/2391',
        start_sample=samplerate * 30)
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

    # rainbowgram =============================================
    rg = rainbowgram(audio_example)
    img_data, rg_meta = logger.log_matrix('rainbowgram', rg[::-1, :, :])
    rainbowgram_display = ImageComponent(rg_meta.public_uri, height=400)

    # auditory image model ========================================
    n_filters = 128
    window_size = 256
    fb = gammatone_filter_bank(n_filters=n_filters, size=256, device=device, band_spacing='geometric')

    ta = torch.from_numpy(audio_example).float().to(device).view(1, 1, -1)
    aim = auditory_image_model(ta, fb, aim_window_size=window_size, aim_step_size=64)
    batch, channels, time, periodicity = aim.shape


    aim = aim.view(channels, time, periodicity).permute(1, 2, 0)
    aim = aim / (aim.max() + 1e-8)
    aim_movie, aim_meta = logger.log_movie('aim', aim)
    aim_display = ImageComponent(aim_meta.public_uri, height=400)


    return dict(
        source=audio_display,
        mag_spec=spec_display,
        rainbowgram=rainbowgram_display,
        aim=aim_display)

def generate_article():
    page_components = generate_page_dict()
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
        generate_article()
