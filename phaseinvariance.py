"""[markdown]

In this article, we explore an audio transformation which is perceptually-inspired and maintains fine-grained
audio information while remaining invariant to imperceptible phase shifts.

TODO: relationship to the Auditory Image Model and to Mallat's scattering transform

https://www.acousticscale.org/wiki/index.php/AIM2006_Documentation

https://www.di.ens.fr/data/scattering/

https://www.di.ens.fr/data/scattering/audio/

"""
from typing import Tuple

from conjure import S3Collection, Logger, numpy_conjure, conjure_article
from data import get_audio_segment
import numpy as np
import torch
from argparse import ArgumentParser
from scipy.signal import stft

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


"""[markdown]
# The Magnitude Spectrogram

First, we explore the most commonly-used feature for audio loss functions
"""


def spectrogram(
        audio: np.ndarray,
        window_size: int = 2048,
        step_size: int = 256,
        mag_only: bool = False) -> np.ndarray:

    _, _, spec = stft(audio, nperseg=window_size, noverlap=window_size - step_size)
    coeffs, frames = spec.shape

    if mag_only:
        return np.abs(spec)

    return spec


"""[markdown]

## Reconstruction from the Magnitude Spectrogram

"""

"""[markdown]

# The RainbowGram

We visualize a feature that also displays frame-to-frame phase deltas
"""

def to_magnitude_and_phase_delta(
        audio: np.ndarray,
        window_size: int = 2048,
        step_size: int = 256) -> Tuple[np.ndarray, np.ndarray]:

    spec = spectrogram(audio, window_size, step_size)

    mag = np.abs(spec)
    phase = np.angle(spec)
    phase_delta = np.diff(phase, axis=-1)
    phase = phase_delta % (2 * np.pi)

    return mag, phase


def from_magnitude_and_phase_delta(mag: np.ndarray, phase: np.ndarray) -> np.ndarray:
    accum = np.cumsum(phase, axis=-1)
    accum = (accum * np.pi) % (2 * np.pi) - np.pi
    spec = mag * torch.exp(1j * accum)
    return spec


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

def generate_page_dict() -> dict:

    return dict()

def generate_article():
    conjure_article(
        __file__, 'html', title="Phase Invariant Feature", max_depth=1)

if __name__ == '__main__':
    parser = ArgumentParser()
    args = parser.parse_args()
    generate_article()
