from collections import defaultdict
from scipy.ndimage import gaussian_filter
from dict_learning_step import dict_learning_step, unit_norm
from os import PathLike
from typing import Union
from sparse2 import freq_decompose, freq_recompose, resample, initialize_dictionary, sparse_decode, sparse_encode
from datastore import batch_stream
import numpy as np
import soundfile
from matplotlib import pyplot as plt


# n_bands = 6

# atom_sizes = [(512, 512)] * n_bands

# # number of examples to train on at a time
# batch_size = 128

# # signal size when testing encoding/decoding for quality evaluation
# signal_size = 2**15

# # size of signal we train on
# segment_size = 1024

# # sparse code iterations during training
# sparse_code_iterations = 32

# # where to find training examples
# path = '/hdd/musicnet/train_data'
# pattern = '*.wav'


def initialize_multilevel_dictionary(shapes: 'list[tuple(int, int)]'):
    return {i: initialize_dictionary(*shape) for i, shape in enumerate(shapes)}


# def interactive_multilevel_sparse_encode(d: 'dict[int, np.ndarray]'):

#     sr = 22050

#     batch = next(batch_stream(path, pattern, 1, signal_size))
#     batch /= np.max(np.abs(batch), axis=-1, keepdims=True) + 1e-12

#     soundfile.write('orig.wav', batch.squeeze(), sr)

#     encoding = defaultdict(dict)

#     bands = freq_decompose(batch, n_bands)
#     band_sizes = sorted(bands.keys())
#     bands = {i: bands[k] for i, k in enumerate(band_sizes)}

#     while True:
#         for i in bands.keys():
#             sig_size = bands[i].shape[1]
#             b = bands[i]
#             dct = d[i]
#             atom_size = dct.shape[1]
#             b = np.pad(b, [(0, 0), (atom_size, atom_size)])
#             instances, residual = sparse_encode(50, sig_size, b, dct)
#             encoding[i].update(instances)
#             bands[i] = residual[:, atom_size:-atom_size]

#         decoded = multilevel_sparse_decode(1, band_sizes, d, encoding)
#         decoded = {size: v for size, v in zip(band_sizes, decoded.values())}
#         signal = freq_recompose(decoded)

#         # re normalize
#         signal /= np.max(np.abs(signal), axis=-1, keepdims=True) + 1e-12
#         soundfile.write('listen.wav', signal.squeeze(), sr)
#         input('Listen....')


def multilevel_sparse_encode(
        sparse_code_iterations: 'list[int]',
        batch: 'dict[int, np.ndarray]',
        d: 'dict[int, np.ndarray]',
        thresholds: 'list[float]' = None):

    encoding = {}

    for i, iterations in enumerate(sparse_code_iterations):
        sig_size = batch[i].shape[1]
        b = batch[i]
        dct = d[i]
        atom_size = dct.shape[1]

        b = np.pad(b, [(0, 0), (atom_size, atom_size)])

        instances, _ = sparse_encode(
            iterations, sig_size, b, dct, thresholds[i] if thresholds is not None else None)
        encoding[i] = instances

    return encoding


def multilevel_sparse_decode(batch_size, signal_sizes, d, encodings):
    decoded = {}
    for i, size in enumerate(signal_sizes):
        atom_size = d[i].shape[1]
        sig = sparse_decode(batch_size, size, d[i], encodings[i])
        decoded[size] = sig[:, atom_size:-atom_size]
    return decoded


def multilevel_batch_stream(
        path: Union[str, PathLike],
        pattern: str,
        batch_size: int,
        signal_size: int,
        n_bands: int):
    """
    Problem: dictionary keys won't be stable as the signal_size changes

    Other options for keys that *will* be stable:
        - freq bands
        - index
    """
    for batch in batch_stream(path, pattern, batch_size, signal_size):

        batch /= np.max(np.abs(batch), axis=-1, keepdims=True) + 1e-12

        bands = freq_decompose(batch, n_bands)
        yield {i: bands[k] for i, k in enumerate(sorted(bands.keys()))}


thresholds = [5.0, 2.25, 1.7, 0.6, 0.12, 0.09]


def encode(signal, n_bands, d):
    """
    Encode a single signal (batch size of 1)

    Represent the signal as a set of tuples of:
        (band_size/index, atom_index, time_pos, magnitude)
    """
    bands = freq_decompose(signal, n_bands)
    band_sizes = sorted(bands.keys())
    bands = {i: bands[k] for i, k in enumerate(sorted(bands.keys()))}
    encoded = multilevel_sparse_encode(
        [128] * len(bands), bands, d, thresholds)

    for i, size in enumerate(band_sizes):
        for instances in encoded[i].values():
            for instance in instances:
                atom, pos, mag, batch = instance
                yield size, atom, pos, mag


def preview(path, pattern, signal_size, d, n_bands):
    sr = 22050

    batch = next(batch_stream(path, pattern, 1, signal_size))
    batch /= np.max(np.abs(batch), axis=-1, keepdims=True) + 1e-12

    soundfile.write('orig.wav', batch.squeeze(), sr)

    print('encoding....')
    bands = freq_decompose(batch, n_bands)
    band_sizes = sorted(bands.keys())
    bands = {i: bands[k] for i, k in enumerate(sorted(bands.keys()))}
    encoded = multilevel_sparse_encode(
        [128] * len(bands), bands, d, thresholds)

    print(encoded.keys())

    print('decoding......')
    decoded = multilevel_sparse_decode(1, band_sizes, d, encoded)
    decoded = {size: v for size, v in zip(band_sizes, decoded.values())}
    signal = freq_recompose(decoded)

    # re normalize
    signal /= np.max(np.abs(signal), axis=-1, keepdims=True) + 1e-12

    soundfile.write('listen.wav', signal.squeeze(), sr)


def learn_multilevel_dict(
        atom_sizes,
        batch_size,
        signal_size,
        segment_size,
        path,
        pattern,
        sparse_code_iterations,
        max_iters=None):

    n_bands = len(atom_sizes)
    d = initialize_multilevel_dictionary(atom_sizes)

    for iteration, batch in enumerate(multilevel_batch_stream(
            path, pattern, batch_size, signal_size, n_bands)):

        if iteration >= max_iters:
            break

        print('=============================')

        for band_index in range(n_bands):
            full_band_batch = batch[band_index]
            band_batch = unit_norm(full_band_batch[:, :segment_size])
            band_d = d[band_index]
            dict_learning_step(
                band_batch,
                atom_sizes[band_index][1],
                sparse_code_iterations,
                band_batch.shape[1],
                band_d,
                iteration)

        if iteration % 5 == 0:
            preview(path, pattern, signal_size, d, n_bands)

    return d


# if __name__ == '__main__':
#     learn_multilevel_dict(
#         atom_sizes,
#         batch_size,
#         signal_size,
#         segment_size,
#         path,
#         pattern,
#         sparse_code_iterations)
