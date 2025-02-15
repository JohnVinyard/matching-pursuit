from config import Config
from data.datastore import iter_audio_segments
import os
import torch
from iterativedecomposition import Model as IterativeDecompositionModel
from modules.eventgenerators.overfitresonance import OverfitResonanceModel
import numpy as np

n_samples = 2 ** 17
samples_per_event = 2048

# this is cut in half since we'll mask out the second half of encoder activations
n_events = (n_samples // samples_per_event) // 2
context_dim = 32

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256

# log_amplitude = True

n_frames = n_samples // transform_step_size

proj = np.random.uniform(-1, 1, (context_dim, 8192))


def project_event_vectors(vectors: torch.Tensor) -> np.ndarray:
    x = vectors.data.cpu().numpy().reshape((-1, context_dim))
    x = x @ proj
    indices = np.argsort(x, axis=-1)[:, -8:]

    sparse = np.zeros_like(x, dtype=np.bool8)
    np.put_along_axis(sparse, indices, values=np.ones_like(indices, dtype=np.bool8), axis=-1,)

    sparse = np.logical_or.reduce(sparse, axis=0)
    print(sparse.shape)
    print(sparse)
    print(sparse.sum())

    return sparse

def make_key(full_path: str, start: int, stop: int) -> str:
    path, filename = os.path.split(full_path)
    fn, ext = os.path.splitext(filename)
    return f'{fn}_{start}_{stop}'

def filepath_from_key(key: str) -> str:
    _id, start, end = key.split('_')
    fp = os.path.join(Config.audio_path(), f'{_id}.wav')
    return fp

def slice_from_key(key: str) -> slice:
    _id, start, end = key.split('_')
    return slice(int(start), int(end))


def iter_chunks():
    for key, chunk in iter_audio_segments(
            Config.audio_path(),
            '*.wav',
            chunksize=n_samples,
            make_key=make_key):
        yield key, chunk

def iter_encodings():
    hidden_channels = 512
    wavetable_device = 'cpu'

    model = IterativeDecompositionModel(
        in_channels=1024,
        hidden_channels=hidden_channels,
        resonance_model=OverfitResonanceModel(
            n_noise_filters=64,
            noise_expressivity=4,
            noise_filter_samples=128,
            noise_deformations=32,
            instr_expressivity=4,
            n_events=1,
            n_resonances=4096,
            n_envelopes=64,
            n_decays=64,
            n_deformations=64,
            n_samples=n_samples,
            n_frames=n_frames,
            samplerate=samplerate,
            hidden_channels=hidden_channels,
            wavetable_device=wavetable_device,
            fine_positioning=True,
            fft_resonance=True
        ))

    with open('iterativedecomposition14.dat', 'rb') as f:
        model.load_state_dict(torch.load(f, map_location=lambda storage, loc: storage))

    for key, chunk in iter_chunks():
        chunk = chunk.view(1, 1, n_samples)
        channels, vectors, schedules = model.iterative(chunk)

        x = project_event_vectors(vectors)
        yield key, chunk, vectors


if __name__ == '__main__':
    for key, chunk, vectors in iter_encodings():
        print(key, chunk.shape, vectors.shape)