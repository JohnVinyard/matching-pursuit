import pickle
from random import choice

import librosa
import zounds

from config import Config
from data.datastore import iter_audio_segments, load_audio_chunk
import os
import torch
from iterativedecomposition import Model as IterativeDecompositionModel
from modules import max_norm
from modules.eventgenerators.overfitresonance import OverfitResonanceModel
import numpy as np

from modules.search import BruteForceSearch
from util.playable import listen_to_sound

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

    # compute graph edges
    x = x[:, None, :] - x[:, :, None]

    x = x.reshape((-1, context_dim))

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

def play_index_entry(key: str) -> None:
    sl = slice_from_key(key)
    fp = filepath_from_key(key)

    print('Playing', key)

    # samples, sr = librosa.load(fp)
    # samples = torch.from_numpy(samples)


    samples = load_audio_chunk(fp, sl, device='cpu')


    samples = max_norm(samples)
    # print(samples.shape)
    # samples = samples[sl]
    # print(fp, sl, samples.shape, samples)

    samples = zounds.AudioSamples(samples.data.cpu().numpy(), zounds.SR22050())
    listen_to_sound(samples, wait_for_user_input=True)


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
        yield key, chunk, x

def build_index():
    keys = []
    vecs = []

    for key, chunk, vectors in iter_encodings():
        keys.append(key)
        vecs.append(torch.from_numpy(vectors)[None, :])
        print(key, vectors.shape, vectors)
        print(f'Nascent index has {len(keys)} entries')

        if len(keys) > 0 and len(keys) % 64 == 0:
            index_vecs = torch.cat(vecs, dim=0)
            search = BruteForceSearch(index_vecs, keys, n_results=16, visualization_dim=2)
            with open('hyperdimensionalindex.dat', 'wb') as f:
                pickle.dump(search, f, pickle.HIGHEST_PROTOCOL)
            print('Storing index', keys, index_vecs.shape)

    index_vecs = torch.cat(vecs, dim=0)
    search = BruteForceSearch(index_vecs, keys, n_results=16, visualization_dim=2)
    with open('hyperdimensionalindex.dat', 'wb') as f:
        pickle.dump(search, f, pickle.HIGHEST_PROTOCOL)


def evaluate_index():
    with open('hyperdimensionalindex.dat', 'rb') as f:
        index: BruteForceSearch = pickle.load(f)
        index.embeddings = index.embeddings.to(torch.float32)


    while True:
        print('=============================')

        key, embedding = index.choose_random()
        print('Here is the query')
        play_index_entry(key)

        keys, embeddings = index.search(embedding)
        print(len(keys))

        for i, key in enumerate(keys):
            print(f'Here is the {i}th most similar')
            play_index_entry(key)

        input('Next example...')

if __name__ == '__main__':
    # build_index()
    evaluate_index()