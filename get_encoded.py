import os
from datastore import audio, iter_chunks
from multilevel_sparse import encode, learn_multilevel_dict
from conjure import cache, LmdbCollection, clear_cache, dump_pickle, hash_args, hash_function, load_pickle
import numpy as np
from multiprocessing.pool import ThreadPool
from random import choice

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

def iter_keys():
    for txn, key in collection.iter_prefix(encode_chunk.func_hash):
        yield key

def iter_training_examples():
    keys = list(iter_keys())

    while True:
        key = choice(keys)
        chunks = collection.iter_prefix(key, encode_chunk.func_hash)

        txn, key1 = next(chunks)
        try:
            txn, key2 = next(chunks)
        except StopIteration:
            key2 = None

        if key2 is None:
            value, txn = collection[key1]
            encoded = load_pickle(value, txn)
            yield encoded
        else:
            # print('TWO OF EM')
            start = np.random.uniform(0, 1)

            value1, txn = collection[key1]
            value2, txn = collection[key2]
            encoded1 = load_pickle(value1, txn)
            encoded2 = load_pickle(value2, txn)

            # print(encoded1)
        
            # for sig_size, atom, pos, mag in encoded:

            e1 = filter(lambda x: x[2] >= (start * x[0]), encoded1)
            e1 = map(lambda x: (x[0], x[1], x[2] - int(start * x[0]), x[3]), e1)

            e2 = filter(lambda x: x[2] <= (start * x[0]), encoded2)
            e2 = map(lambda x: (x[0], x[1], x[2] + (x[0] - (start * x[0])), x[3]), e2)
            
            combined = list(e1) + list(e2)
            # print(combined)
            yield combined


        # chunks = collection.iter_from(key.encode(), keys=False, values=True)
        # txn, chunk1 = next(chunks)

        # try:
        #     txn, chunk2 = next(chunks)
        # except StopIteration:
        #     chunk2 = None
        
        # if chunk2 is None:
        #     yield load_pickle(chunk1, txn)
        # else:
        #     pass


        # value, txn = collection[key]
        # encoded = load_pickle(value, txn)
        # yield encoded
        
        # for txn, key in collection.iter_prefix(encode_chunk.func_hash):
        #     raw = txn.get(key)
        #     encoded = load_pickle(raw, txn)
        #     yield encoded

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
