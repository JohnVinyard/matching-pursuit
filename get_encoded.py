import os
from datastore import audio, iter_chunks
from multilevel_sparse import encode, learn_multilevel_dict
from conjure import cache, LmdbCollection, clear_cache, dump_pickle, hash_args, hash_function, load_pickle
import numpy as np
from multiprocessing.pool import ThreadPool

n_bands = 6

atom_sizes = [(512, 512)] * n_bands

# number of examples to train on at a time
batch_size = 128

# signal size when testing encoding/decoding for quality evaluation
signal_size = 2**15

# size of signal we train on
segment_size = 1024

# sparse code iterations during training
sparse_code_iterations = 32

# where to find training examples
path = '/hdd/musicnet/train_data'
pattern = '*.wav'

collection = LmdbCollection('sparse_coded')


@cache(collection, serialize=dump_pickle, deserialze=load_pickle)
def learn_dict():
    print('learning dict...')
    d = learn_multilevel_dict(
        atom_sizes,
        batch_size,
        signal_size,
        segment_size,
        path,
        pattern,
        sparse_code_iterations,
        max_iters=30)
    return d


def hash_encode_chunk_args(*args, **kwargs):
    return hash_args(args[:-1])


@cache(
    collection,
    serialize=dump_pickle,
    deserialze=load_pickle,
    arg_hasher=hash_encode_chunk_args)
def encode_chunk(path, start, stop, d):

    # get normalized audio
    data = audio(path)[start: stop]
    data /= np.abs(data).max() + 1e-12

    diff = signal_size - len(data)
    data = np.pad(data, [(0, diff)]).reshape((1, signal_size))
    encoded = list(encode(data, n_bands, d))

    fn = os.path.split(path)[-1]
    print(f'{fn} {start} {stop} [{encoded[0]} ... {encoded[-1]}]')
    return encoded


def iter_training_examples():
    while True:
        for txn, key in collection.iter_prefix(encode_chunk.func_hash):
            raw = txn.get(key)
            encoded = load_pickle(raw, txn)
            yield encoded

if __name__ == '__main__':
    sparse_dict = learn_dict()

    for example in iter_training_examples():
        print(len(example))

    # for i, chunk in enumerate(iter_chunks(path, pattern, signal_size)):
    #     print('==============================================')
    #     print(chunk)
    #     encoded = encode_chunk(*chunk, sparse_dict)
    #     print(i, len(encoded))

    # pool = ThreadPool(processes=8)
    # pool.map(lambda x: encode_chunk(*x, sparse_dict),
    #          iter_chunks(path, pattern, signal_size))
