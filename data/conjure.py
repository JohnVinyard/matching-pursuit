from hashlib import sha1
import pickle
import numpy as np
import lmdb
import struct


class NoCache(Exception):
    def __init__(self, value):
        super().__init__()
        self.value = value


class Wrapped(object):
    def __init__(self, callable, func_hash):
        super().__init__()
        self.func_hash = func_hash
        self.callable = callable

    def __call__(self, *args, **kwargs):
        return self.callable(*args, **kwargs)


def hash_function(f):
    h = sha1()

    if f.__closure__ is not None:
        freevars = [x.cell_contents for x in f.__closure__]
    else:
        freevars = None

    h.update(pickle.dumps(freevars))
    try:
        h.update(pickle.dumps(f.__code__.co_consts))
    except Exception as e:
        pass
    h.update(f.__name__.encode())
    h.update(f.__code__.co_code)
    value = h.hexdigest()
    return value


def hash_args(*args, **kwargs):
    args_hash = sha1()
    args_hash.update(pickle.dumps(args))
    args_hash.update(pickle.dumps(kwargs))
    args_hash = args_hash.hexdigest()
    return args_hash


def non_generator_func(f, h, collection, serialize, deserialize, arg_hasher):
    def x(*args, **kwargs):
        args_hash = arg_hasher(*args, **kwargs)
        key = f'{h}:{args_hash}'.encode()
        try:
            cached = deserialize(*collection[key])
            return cached
        except KeyError:
            pass

        try:
            result = f(*args, **kwargs)
            collection[key] = serialize(result)
        except NoCache as nc:
            result = nc.value
        return result

    return x


def dump_pickle(x):
    s = pickle.dumps(x, pickle.HIGHEST_PROTOCOL)
    return memoryview(s)


def numpy_array_dumpb(arr):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    shape = pickle.dumps(arr.shape)
    shape_bytes = struct.pack('i', len(shape))
    return memoryview(shape_bytes + shape + arr.tobytes())


def load_pickle(memview, txn):
    return pickle.loads(memview)


def numpy_array_loadb(memview, txn):
    shape_len = struct.unpack('i', memview[:4])[0]
    shape = pickle.loads(memview[4: 4 + shape_len])
    raw = np.asarray(memview[4 + shape_len:], dtype=np.uint8)
    arr = raw.view(dtype=np.float32).reshape(shape)
    return NumpyWrapper(arr, txn)


def cache(
        collection,
        serialize=numpy_array_dumpb,
        deserialze=numpy_array_loadb,
        arg_hasher=hash_args):
    def deco(f):
        h = hash_function(f)
        return Wrapped(
            non_generator_func(f, h, collection, serialize, deserialze, arg_hasher), h)

    return deco


class NumpyWrapper(object):
    def __init__(self, arr, txn):
        super().__init__()
        self.txn = txn
        self.arr = arr

    def __len__(self):
        return len(self.arr)

    @property
    def shape(self):
        return self.arr.shape

    def __getitem__(self, item):
        data = self.arr[item].copy()
        self.txn.commit()
        return data


class LmdbCollection(object):
    def __init__(self, path):
        self.path = path
        self.env = lmdb.open(
            self.path,
            max_dbs=10,
            map_size=10e10,
            writemap=True,
            map_async=True,
            metasync=True)

    def iter_prefix(self, start_key, prefix=None):

        if isinstance(start_key, str):
            start_key = start_key.encode()

        if prefix is None:
            prefix = start_key
        elif isinstance(prefix, str):
            prefix = prefix.encode()

        with self.env.begin(write=True, buffers=True) as txn:
            cursor = txn.cursor()
            cursor.set_range(start_key)

            it = cursor.iternext(keys=True, values=False)
            for key in it:
                key = bytes(key)
                if not key.startswith(prefix):
                    break
                yield txn, key

    def __setitem__(self, key, value):
        with self.env.begin(write=True, buffers=True) as txn:
            txn.put(key, value)

    def __getitem__(self, key):
        txn = self.env.begin(buffers=True, write=False)
        value = txn.get(key)
        if value is None:
            raise KeyError(key)
        return value, txn


def clear_cache(func, collection):
    for txn, key in collection.iter_prefix(func.func_hash):
        print(key)
        txn.delete(key)
