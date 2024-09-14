"""[markdown]

# TODOS
- convenience function for the `BytesIO` pattern
- fix zounds issues
"""

from functools import reduce
from typing import Any, Dict, Tuple
from conjure import numpy_conjure, audio_conjure, bytes_conjure, S3Collection, SupportedContentType, tensor_movie, \
    ImageComponent, AudioComponent, CitationComponent
import requests
from io import BytesIO
from librosa import load
import numpy as np
# from conjurearticle import conjure_article, AudioComponent, CitationComponent, ImageComponent
from soundfile import SoundFile
from scipy.signal import gammatone, stft
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
from enum import Enum

collection = S3Collection(
    bucket='conjure-test', 
    is_public=True, 
    cors_enabled=True)


"""[markdown]

# Focusing on Perceptually Relevant Features of Audio 

> "Here's a single quote"


- this
- is
- a
- list

We'll start with some audio

"""

# class ExtendedContentType(Enum):
#     GIF = 'image/gif'


def gammatone_filter_bank(n_filters: int, size: int) -> np.ndarray:
    bank = np.zeros((n_filters, size))
    
    frequencies = np.linspace(
        20, 
        11000, 
        num=n_filters)
    
    for i, freq in enumerate(frequencies):
        b, a = gammatone(
            freq=freq, 
            ftype='fir', 
            order=4, 
            numtaps=size, 
            fs=22050)
        
        bank[i] = b
    
    bank = bank / np.abs(bank).max(axis=-1, keepdims=True)
    return bank

@bytes_conjure(collection, content_type=SupportedContentType.Image, read_hook=lambda x: x)
def time_frequency_plot(spec: np.ndarray) -> bytes:
    bio = BytesIO()
    plt.matshow(spec[::-1, :])
    plt.savefig(bio)
    plt.clf()
    bio.seek(0)
    return bio.read()

@bytes_conjure(collection, content_type=SupportedContentType.Image, read_hook=lambda x: x)
def gammatone_plot(filterbank: np.ndarray) -> bytes:
    bio = BytesIO()
    for i, filt in enumerate(filterbank):
        plt.plot(filt * 10 + (i * 20), color='black')
    
    plt.savefig(bio)
    plt.clf()
    bio.seek(0)
    return bio.read()


@bytes_conjure(collection, content_type='image/gif')
def tensor_movie_gif(x: np.ndarray):
    a, b, c = x.shape
    
    x = x.reshape((a, b*c))
    x /= x.max(axis=-1, keepdims=True)
    x = x.reshape((a, b, c))
    
    x = np.log(1e-3 + x)
    
    b = tensor_movie(x)
    return b


# audio

# spec

@numpy_conjure(collection, read_hook=lambda x: f'Reading audio {x} from cache')
def get_audio_segment(
        url: str, 
        target_samplerate: int, 
        start_sample: int, 
        duration_samples: int):
    
    resp = requests.get(url)
    bio = BytesIO(resp.content)
    bio.seek(0)
    
    samples, _ = load(bio, sr=target_samplerate, mono=True)
    
    segment = samples[start_sample: start_sample + duration_samples]
    
    diff = duration_samples - segment.shape[0]
    if diff > 0:
        segment = np.pad(segment, [(0, diff)])
    
    return segment.astype(np.float32)

@audio_conjure(collection)
def encode_audio(samples: np.ndarray):
    bio = BytesIO()
    with SoundFile(
            bio, 
            mode='w', 
            samplerate=22050, 
            channels=1, 
            format='WAV', 
            subtype='PCM_16') as sf:
        
        sf.write(samples)
    bio.seek(0)
    
    return bio.read()

def fft_convolve(*args, norm=None) -> torch.Tensor:

    n_samples = args[0].shape[-1]

    # pad to avoid wraparound artifacts
    padded = [F.pad(x, (0, x.shape[-1])) for x in args]
    
    specs = [torch.fft.rfft(x, dim=-1, norm=norm) for x in padded]
    spec = reduce(lambda accum, current: accum * current, specs[1:], specs[0])
    final = torch.fft.irfft(spec, dim=-1, norm=norm)

    # remove padding
    return final[..., :n_samples]


def n_fft_coeffs(x: int):
    return x // 2 + 1

def convolve(signal: np.ndarray, filters: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    
    n_samples = signal.shape[-1]
    
    signal = signal.reshape((-1, n_samples))
    
    n_filters, filter_size = filters.shape
    padded = np.pad(filters, [(0, 0), (0, n_samples - filter_size)])
    
    sig = torch.from_numpy(signal)
    filt = torch.from_numpy(padded)
    
    spec = fft_convolve(sig ,filt)
    
    # half-wave rectification
    spec = torch.relu(spec)
    rectified = spec.view(1, n_filters, n_samples)
    aim_window_size = 128
    rectified = rectified.unfold(-1, aim_window_size, aim_window_size // 2)
    aim = torch.abs(torch.fft.rfft(rectified, dim=-1)) # (batch, channels, time, periodicity)
    aim = aim.view(n_filters, -1, n_fft_coeffs(aim_window_size)) # (channels, time, periodicity)
    aim = aim.data.cpu().numpy()
    aim = aim.transpose((1, 0, 2))[:, ::-1, :]
    
    spec = spec.view(1, -1, signal.shape[-1])
    spec = F.max_pool1d(spec, kernel_size=256, stride=128)
    
    pooled = spec.data.cpu().numpy().reshape((n_filters, -1))
    return pooled, aim


"""[markdown]

## Gammatone Filter Bank

You can see a handful of filters here, and how they vary in their time/frequency
tradeoff.

"""

# zoomed_gammatone

"""[markdown]

Here, you can see the full filterbank.

"""

# full_gammatone

"""[markdown]

And finally, the spectrogram that results from convolving the filter bank with our signal.

Note that we've pooled over sliding windows, since the original result of the convolution 
will have the same time dimension as the original signal.
"""

# gammatone_spec

"""[markdown]

Here's move of the auditory image model-inspired data

"""

# aim_movie

def main() -> Dict[str, Any]:
    
    n_samples = 2 ** 16
    
    audio = get_audio_segment(
        url='https://music-net.s3.amazonaws.com/1728',
        target_samplerate=22050,
        start_sample=128,
        duration_samples=n_samples
    )
    
    _, _, spec = stft(audio)
    spec = np.abs(spec)
    
    _, tf = time_frequency_plot.result_and_meta(spec)
    _, audio_metadata = encode_audio.result_and_meta(audio)
    
    n_filters = 128
    filter_size = 128


    # I also want to avoid this:  just running a function and grabbing the
    # metadata
    # I also had to define this gammatone plot function
    # which doesn't do much of anything

    # logger.log(filters,
    fb = gammatone_filter_bank(n_filters, filter_size)
    _, gt_full_meta = gammatone_plot.result_and_meta(fb)

    _, gt_zoomed_meta = gammatone_plot.result_and_meta(fb[16:32])
    
    gammatone_spec, aim = convolve(audio, fb)
    
    _, aim_movie = tensor_movie_gif.result_and_meta(aim)
    
    _, gammatone_spec = time_frequency_plot.result_and_meta(gammatone_spec)
    
    
    spec = ImageComponent(
        src=tf.public_uri.geturl(),
        height=400
    )
    
    aim_movie = ImageComponent(
        src=aim_movie.public_uri.geturl(),
        height=400
    )
    
    gammatone_spec = ImageComponent(
        src=gammatone_spec.public_uri.geturl(),
        height=400
    )
    
    full_gammatone = ImageComponent(
        src=gt_full_meta.public_uri.geturl(), 
        height=400
    )
    
    zoomed_gammatone = ImageComponent(
        src=gt_zoomed_meta.public_uri.geturl(),
        height=400
    )
    
    audio = AudioComponent(
        src=audio_metadata.public_uri.geturl(), 
        height=100, 
        scale=1, 
        controls=True
    )
    
    citation = CitationComponent(
        tag='blah',
        author='Vinyard, John',
        url='https://example.com',
        header='blah',
        year='2024'
    )
    
    return dict(
        audio=audio,
        citation=citation,
        full_gammatone=full_gammatone,
        zoomed_gammatone=zoomed_gammatone,
        spec=spec,
        gammatone_spec=gammatone_spec,
        aim_movie=aim_movie
    )

"""[markdown]

# Citations

If you'd like to cite this article, please use the following:
"""

# citation


if __name__ == '__main__':
    display = main()
    conjure_article(__file__, 'html', **display)