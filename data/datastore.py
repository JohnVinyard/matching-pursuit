
from typing import Iterable, Tuple
from librosa import load, to_mono
from fnmatch import fnmatch
import os
import numpy as np
from random import choice

import torch

#  TODO: Switch to using conjure for this
from data.conjure import LmdbCollection, cache

collection = LmdbCollection('audio')

def iter_files(base_path, pattern):
    for dirpath, _, filenames in os.walk(base_path):
        audio_files = filter(
            lambda x: fnmatch(x, pattern),
            (os.path.join(dirpath, fn) for fn in filenames))
        yield from audio_files

def iter_files_in_random_order(base_path, pattern):
    filenames = list(iter_files(base_path, pattern))
    perm = np.random.permutation(len(filenames))
    yield from [filenames[index] for index in perm]

# TODO: Switch to using conjure for this
@cache(collection)
def audio(path):
    print(f'loading {path} from disk')
    x, _ = load(path)
    x = to_mono(x)
    return x


def load_audio_chunk(path: str, slce: slice, device = None) -> torch.Tensor:
    data = audio(path)[:]
    data = torch.from_numpy(data).float()
    data = data[slce]
    
    if device is not None:
        data = data.to(device)
    
    return data

def iter_chunks(path, pattern, chunksize):
    """
    Return an iterable of tuples of (filepath, start_sample, end_sample)
    """
    for fp in iter_files(path, pattern):
        data = audio(fp)
        total_samples = len(data)
        for i in range(0, total_samples, chunksize):
            start = i
            stop = i + chunksize
            yield fp, start, stop


def iter_audio_segments(
        path: str, 
        pattern: str, 
        chunksize: int, 
        make_key = lambda fp, start, stop: f'{fp}_{start}_{stop}',
        device = None) -> Iterable[Tuple[str, torch.Tensor]]:
    
    for fp in iter_files_in_random_order(path, pattern):
        data = audio(fp)
        data = torch.from_numpy(data[:]).float().to(device).view(1, 1, -1)
        total_samples = data.shape[-1]
        
        for i in range(0, total_samples - chunksize, chunksize):
            start = i
            stop = i + chunksize
            key = make_key(fp, start, stop)
            chunk = data[:, :, start: stop]
            chunk = chunk / (chunk.max() + 1e-8)
            yield key, chunk

# def iter_audio_segment_batches(
#         batch_size: int, 
#         path: str, 
#         pattern: str, 
#         chunksize: int, 
#         make_key = lambda fp, start, stop: f'{fp}_{start}_{stop}', 
#         device = None):
    
#     current_batch = torch.zeros(batch_size, 1, chunksize, device=device)
#     batch_index = 0
    
#     for key, segment in iter_audio_segments(path, pattern, chunksize, make_key, device):
#         current_batch[batch_index: batch_index + 1, :, :] = segment
#         batch_index += 1
        
#         if batch_index == batch_size:
#             yield current_batch
#             current_batch = torch.zeros(batch_size, 1, chunksize, device=device)
#             batch_index = 0

def batch_stream(
        path, 
        pattern, 
        batch_size, 
        n_samples, 
        overfit=False, 
        normalize=False, 
        step_size=1,
        return_indices=False):
    
    paths = list(iter_files(path, pattern))

    batch_size = 1 if overfit else batch_size

    while True:
        batch = np.zeros((batch_size, n_samples), dtype=np.float32)
        indices = []
        
        for i in range(batch_size):
            path = choice(paths)
            data = audio(path)

            # get the possible starting positions, given the step size
            positions = (data.shape[0] - n_samples) // step_size

            # choose one of the start positions, snapped to step size
            start = np.random.randint(0, positions) * step_size

            end = start + n_samples
            indices.append((start, end))
            
            batch[i, :] = data[start: end]

        if normalize:
            batch = batch / (np.abs(batch.max(axis=-1, keepdims=True)) + 1e-12)
        
        if return_indices:
            yield batch, indices
        else:
            yield batch

        if overfit:
            while True:
                yield batch
