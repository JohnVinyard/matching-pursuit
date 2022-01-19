
from librosa import load, to_mono
from fnmatch import fnmatch
import os
import numpy as np
from random import choice

from data.conjure import LmdbCollection, cache

collection = LmdbCollection('audio')


def iter_files(base_path, pattern):

    for dirpath, _, filenames in os.walk(base_path):
        audio_files = filter(
            lambda x: fnmatch(x, pattern),
            (os.path.join(dirpath, fn) for fn in filenames))
        yield from audio_files


@cache(collection)
def audio(path):
    print(f'loading {path} from disk')
    x, _ = load(path)
    x = to_mono(x)
    return x


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


def batch_stream(path, pattern, batch_size, n_samples, overfit=False):
    paths = list(iter_files(path, pattern))

    batch_size = 1 if overfit else batch_size

    while True:
        batch = np.zeros((batch_size, n_samples), dtype=np.float32)
        for i in range(batch_size):
            path = choice(paths)
            data = audio(path)
            start = np.random.randint(0, data.shape[0] - n_samples)
            batch[i, :] = data[start: start + n_samples]
        yield batch

        if overfit:
            while True:
                yield batch
