import argparse
from dict_learning_step import activations, apply_atom, dict_learning_step, unit_norm
import librosa
import numpy as np
from collections import defaultdict, Counter
from matplotlib import pyplot as plt
import soundfile
from data.datastore import batch_stream
from scipy.fft import rfft, irfft


def mu_law(x, mu=255):
    s = np.sign(x)
    x = np.abs(x)
    return s * (np.log(1 + (mu * x)) / np.log(1 + mu))


def inverse_mu_law(x, mu=255):
    s = np.sign(x)
    x = np.abs(x)
    x *= np.log(1 + mu)
    x = (np.exp(x) - 1) / mu
    return x * s


def freq_decompose(signal, channels):
    """
    signal - A batch of signals of size `(batch_size, signal)`
    """
    highest = int(np.log2(signal.shape[1]))
    lowest = highest - (channels - 1)

    freq = rfft(signal, axis=-1, norm='ortho')

    bands = {}

    for i in range(lowest, highest + 1):
        size = 2 ** i
        half = size // 2

        if i == lowest:
            mask = 1
        else:
            mask = np.ones((1, half + 1))
            mask[:, :half // 2 + 1] = 0

        sig = irfft(freq[:, :half + 1] * mask, norm='ortho')
        bands[size] = sig

    return bands


def resample(signal, size):
    if size == signal.shape[1]:
        return signal

    diff = size - signal.shape[1]

    freq = rfft(signal, axis=-1, norm='ortho')

    padded = np.pad(freq, [(0, 0), (0, diff // 2)])

    sig = irfft(padded, axis=-1, norm='ortho')

    return sig


def freq_recompose(bands):
    size = max(bands.keys())
    return sum([resample(sig, size) for sig in bands.values()])


def sparse_decode(batch_size, signal_size, d, atoms):
    """
    Reconstruct a signal that has been encoding using dictionary `d`

    Parameters
    -------------
    batch_size - The number of examples in the batch to be decoded
    signal_size - The decoded signal size
    d - the dictionary used to encode the signal
    atoms - a list of four tuples of (atom, location, coeff, batch_num)

    Returns
    -------------
    x - a batch of decoded signals

    """
    n_components, atom_size = d.shape
    decoded = np.zeros((batch_size, signal_size + (atom_size * 2)))
    for occurences in atoms.values():
        for atom in occurences:
            # print(atom)
            comp, pos, coeff, batch = atom
            decoded[batch], _ = apply_atom(
                decoded[batch], signal_size, d[comp], pos, coeff, np.add)
    return decoded


def listen(batch, d):
    atom_size = d.shape[1]
    b = batch
    soundfile.write('orig.wav', b.squeeze() * 10, 22050)

    b = np.pad(b, [(0, 0), (atom_size, atom_size)])
    instances, residual = sparse_encode(128, 16384, b, d)
    decoded = sparse_decode(1, 16384, d, instances)

    soundfile.write('listen.wav', decoded.squeeze() * 10, 22050)


def initialize_dictionary(n_components, atom_size):
    """
    Create a randomly-initialized dictionary

    Parameters
    -----------
    n_components - the number of atoms
    atom_size - the dimension of the atoms

    Returns
    -----------
    dictionary - A dictionary of shape `(n_components, atom_size)`
    """
    x = np.random.normal(0, 1, (n_components, atom_size))
    x = unit_norm(x)
    return x



def plot_atoms(d, n_components):
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    indices = np.random.randint(0, n_components, 16)

    for i in range(4):
        for j in range(4):
            axs[i, j].plot(d[indices[i * 4 + j]])
            axs[i, j].tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.savefig('atoms.png')


def sparse_encode(sparse_code_iterations, signal_size, batch, d, threshold=None):
    # signal_size = batch.shape[1]
    atom_size = d.shape[1]

    # record positions and strengths of all atoms
    instances = defaultdict(list)

    iterations = 0

    start_norm = np.linalg.norm(batch[:, atom_size:-atom_size], axis=-1).mean()

    while True:
        # acts is a batch-size-length list of three-tuples of
        # `(atom, location, coeff)`
        acts = activations(batch[:, atom_size:-atom_size], d)

        for i, act in enumerate(acts):
            # remove the best atom from each signal in the batch
            atom, location, coeff = act

            # record position, strength and batch of this atom
            instances[atom].append(act + (i,))

            # remove the atom from the signal
            batch[i], _ = apply_atom(
                batch[i], signal_size, d[atom], location, coeff, np.subtract)

        iterations += 1

        if threshold is not None:
            norm = np.linalg.norm(
                batch[:, atom_size:-atom_size], axis=-1).mean()
            if norm <= threshold:
                print(
                    f'BELOW THRESH done sparse coding {signal_size} iterations {iterations}')
                break

        if iterations > sparse_code_iterations:
            norm = np.linalg.norm(
                batch[:, atom_size:-atom_size], axis=-1).mean()
            print(
                f'MAX ITER {signal_size}: start {start_norm} norm {norm} ratio {norm / start_norm}')
            break

    residual = batch
    return instances, residual


def test_freq_decompose(filename):
    signal, sr = librosa.load(filename, mono=True)
    batch = next(batch_stream(signal, 2, 16384))
    bands = freq_decompose(batch, 5)

    for k, v in bands.items():
        print(k, v.shape)
        r = resample(v, 16384)
        soundfile.write(f'{k}.wav', r[0], 22050)

    rec = freq_recompose(bands)
    soundfile.write('rec.wav', rec[0], 22050)


def learn_dictionary(
        path,
        atom_size,
        signal_size,
        n_components,
        batch_size,
        sparse_code_iterations):
    """
    Learn a dictionary suitable for sparse coding from an input signal

    Parameters
    ------------
    filename - The filename we'll learn a dictionary for
    atom_size - The size of each atom
    n_components - The number of atoms
    batch_size - the number of examples per batch
    sparse_code_iterations - the number of atoms to extract for
        each learning iteration

    Returns
    ------------
    dictionary - The learned dictionary
    """
    d = initialize_dictionary(n_components, atom_size)

    for iteration, batch in enumerate(batch_stream(path, '*.wav', batch_size, signal_size)):
        batch = unit_norm(batch)
        dict_learning_step(
            batch,
            atom_size,
            sparse_code_iterations,
            signal_size,
            d,
            iteration)
        if iteration % 10 == 0:
            b = next(batch_stream(path, '*.wav', 1, 16384))
            listen(unit_norm(b), d)

    return d





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--filepath',
        help='The directory to draw training examples from',
        required=True)
    parser.add_argument(
        '--atom-size',
        type=int,
        default=512,
        help='The dimension of atoms and signals')
    parser.add_argument(
        '--signal-size',
        type=int,
        default=512,
        help='The size of the signals to encode during training')
    parser.add_argument(
        '--n-components',
        type=int,
        default=517,
        help='The number of dictionary atoms')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='The number of examples per batch')
    parser.add_argument(
        '--sparse-code-iterations',
        type=int,
        default=32,
        help='Sparse coding iterations per batch')

    args = parser.parse_args()

    d = learn_dictionary(
        args.filepath,
        args.atom_size,
        args.signal_size,
        args.n_components,
        args.batch_size,
        args.sparse_code_iterations)

