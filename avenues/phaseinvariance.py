"""[markdown]

# TODOS
- table of contents
- remove empty code blocks
- Figure out conjure content types
- convenience function for the `BytesIO` pattern
- create s3 folder per experiment automatically with correct policies 
- how can this work for both experiment time (monitoring) and writing time?
"""

from typing import Any, Dict
from conjure import numpy_conjure, audio_conjure, bytes_conjure, S3Collection
import requests
from io import BytesIO
from librosa import load
import numpy as np
from conjurearticle import conjure_article, AudioComponent, CitationComponent, ImageComponent
from soundfile import SoundFile
from scipy.signal import gammatone, stft
from matplotlib import pyplot as plt
from enum import Enum

collection = S3Collection(
    bucket='conjure-test', 
    is_public=True, 
    cors_enabled=True)


class ExtendedContentType(Enum):
    Image = 'image/png'

"""[markdown]

# Focusing on Perceptually Relevant Features of Audio 

> "Here's a single quote"


- this
- is
- a
- list

We'll start with some audio

"""

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

@bytes_conjure(collection, content_type=ExtendedContentType.Image, read_hook=lambda x: x)
def time_frequency_plot(spec: np.ndarray) -> bytes:
    bio = BytesIO()
    plt.matshow(spec[::-1, :])
    plt.savefig(bio)
    plt.clf()
    bio.seek(0)
    return bio.read()

@bytes_conjure(collection, content_type=ExtendedContentType.Image, read_hook=lambda x: x)
def gammatone_plot(filterbank: np.ndarray) -> bytes:
    bio = BytesIO()
    for i, filt in enumerate(filterbank):
        plt.plot(filt * 10 + (i * 20), color='black')
    
    plt.savefig(bio)
    plt.clf()
    bio.seek(0)
    return bio.read()


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


def main() -> Dict[str, Any]:
    
    
    audio = get_audio_segment(
        url='https://music-net.s3.amazonaws.com/1728',
        target_samplerate=22050,
        start_sample=128,
        duration_samples=2**16
    )
    
    _, _, spec = stft(audio)
    spec = np.abs(spec)
    
    _, tf = time_frequency_plot.result_and_meta(spec)

    _, audio_metadata = encode_audio.result_and_meta(audio)
    
    fb = gammatone_filter_bank(128, 128)
    
    _, gt_full_meta = gammatone_plot.result_and_meta(fb)
    
    _, gt_zoomed_meta = gammatone_plot.result_and_meta(fb[16:32])
    
    spec = ImageComponent(
        src=tf.public_uri.geturl(),
        height=400
    )
    
    full_gammatone = ImageComponent(
        src=gt_full_meta.public_uri.geturl(), 
        height=400)
    
    zoomed_gammatone = ImageComponent(
        src=gt_zoomed_meta.public_uri.geturl(),
        height=400)
    
    audio = AudioComponent(
        src=audio_metadata.public_uri.geturl(), 
        height=100, 
        scale=1, 
        controls=True)
    
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
        spec=spec)

"""[markdown]

# Citations

If you'd like to cite this article, please use the following:
"""

# citation

"""
Here's a normal comment
"""

if __name__ == '__main__':
    display = main()
    conjure_article(__file__, 'html', **display)