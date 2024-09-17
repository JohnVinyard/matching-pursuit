import requests
from io import BytesIO
from librosa import load
import numpy as np

def get_audio_segment(
        url: str,
        target_samplerate: int,
        start_sample: int,
        duration_samples: int) -> np.ndarray:

    resp = requests.get(url)
    bio = BytesIO(resp.content)
    bio.seek(0)

    samples, _ = load(bio, sr=target_samplerate, mono=True)

    segment = samples[start_sample: start_sample + duration_samples]

    diff = duration_samples - segment.shape[0]
    if diff > 0:
        segment = np.pad(segment, [(0, diff)])

    return segment.astype(np.float32)
