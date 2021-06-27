import numpy as np
from collections import defaultdict
from scipy.fft import rfft, irfft
from multiprocessing.pool import ThreadPool
from matplotlib import pyplot as plt

pool = ThreadPool(processes=8)


def unit_norm(x):
    """
    Give each sample unit norm

    Parameters
    ------------
    x - an array of size `(batch, signal_dim)`

    Returns
    ------------
    x - an array where each sample has unit norm
    """
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-8)


def batch_fft_convolve1d(signal, d):
    """
    Convolves the dictionary/filterbank `d` with `signal`

    Parameters
    ------------
    signal - A signal of shape `(batch_size, signal_size)`
    d - A dictionary/filterbank of shape `(n_components, atom_size)`

    Returns
    ------------
    x - the results of convolving the dictionary with the batched signals
        of dimensions `(batch, n_components, signal_size)`
    """
    signal_size = signal.shape[1]
    atom_size = d.shape[1]
    diff = signal_size - atom_size

    half_width = atom_size // 2

    # TODO: Is it possible to leverage the zero padding and/or
    # difference in atom and signal size to avoid multiplying
    # every frequency position
    px = np.pad(signal, [(0, 0), (half_width, half_width)])
    py = np.pad(d, [(0, 0), (0, px.shape[1] - atom_size)])

    fpx = rfft(px, axis=-1)
    fpy = rfft(py[..., ::-1], axis=-1)

    c = fpx[:, None, :] * fpy[None, :, :]

    c = irfft(c, axis=-1)

    c = np.roll(c, signal_size - diff, axis=-1)
    c = c[..., atom_size - 1:-1]

    return c


def fft_convolve(signal, d):
    if signal.shape[0] == 1:
        return batch_fft_convolve1d(signal, d)
    
    results = pool.map(
        lambda x: batch_fft_convolve1d(x, d),
        [signal[i: i + 8, :] for i in range(0, len(signal), 8)])
    return np.concatenate(results, axis=0)


def activations(signal, d):
    """
    Parameters
    ------------
    signal - A signal of shape `(batch_size, signal_size)`
    d - A dictionary/filterbank of shape `(n_components, atom_size)`

    Returns
    ------------
    activations - A list of 3-tuples of `(atom_index, atom_location, coefficient)`
    """

    batch_size, signal_size = signal.shape
    _, atom_size = d.shape

    fm = fft_convolve(signal, d)

    # find the max atom and position for each example
    flattened = fm.reshape((batch_size, -1))
    mx = np.argmax(flattened, axis=1)

    activation = flattened[np.arange(batch_size), mx]

    # TODO: This should likely be signal_size instead
    atom = mx // signal_size
    time = mx % signal_size

    return zip(atom, time, activation)


def apply_atom(signal, signal_size, atom, position, activation, op):
    """
    "Apply" an atom to a signal at a location using op

    Parameters
    -------------
    signal - a `(signal_size + len(atom) * 2)` array
    signal_size - the size of the signal before padding by atom size on either side
    atom - a 1D array
    position - the position of the atom's center, _not adjusted/translated for the padding_
    activation - the coefficient value to apply to the atom before applying `op`
    op - An [arithmetic operation] to be applied to the atom and signal

    Returns
    -------------
    signal - the new signal, in its entirety
    segment - the specific segment where the atom was applied
    """

    atom_size = len(atom)
    expected = signal_size + (len(atom) * 2)
    actual = len(signal)

    # print(len(signal), signal_size, len(atom))

    if expected != actual:
        raise ValueError(
            f'Signal dimension should be {expected} but was {actual}')

    # the signal is padded by atom_size and the position is relative to
    # the atom's center
    translated_start_pos = (position + atom_size) - (atom_size // 2)
    end_pos = translated_start_pos + atom_size

    # TODO: Consider just using the `out` parameter here to avoid
    # the second line
    result = op(signal[translated_start_pos: end_pos], atom * activation)
    signal[translated_start_pos: end_pos] = result

    return signal, result


def dict_learning_step(
        batch: np.ndarray,
        atom_size: int,
        sparse_code_iterations: int,
        signal_size: int,
        d: np.ndarray,
        iteration: int = 0):

    start_norm = np.linalg.norm(batch, axis=-1).mean()

    # zero pad the batch
    batch = np.pad(batch, [(0, 0), (atom_size, atom_size)])

    # record positions and strengths of all atoms
    instances = defaultdict(list)

    # TODO: Refactor to use sparse encoding step
    # sparse coding
    for _ in range(sparse_code_iterations):
        # acts is a batch-size-length list of three-tuples of
        # `(atom, location, coeff)`
        acts = activations(batch[:, atom_size:-atom_size], d)

        for i, act in enumerate(acts):
            # remove the best atom from each signal in the batch
            atom, location, coeff = act

            # atom_counts[atom] += 1

            # record position, strength and batch of this atom
            instances[atom].append(act + (i,))

            # remove the atom from the signal
            batch[i], _ = apply_atom(
                batch[i], signal_size, d[atom], location, coeff, np.subtract)

        # print(np.linalg.norm(
        #     batch[:, atom_size:-atom_size], axis=-1).mean())

    print(
        f'{iteration}: {start_norm} -> {np.linalg.norm(batch[:, atom_size:-atom_size], axis=-1).mean()}')
    
    # batch is now our residual

    # update each atom that was used in the sparse coding phase
    # of the current batch, one atom at a time
    for atom_index, acts in instances.items():
        segments = np.zeros((len(acts), atom_size))
        coeffs = np.zeros((len(acts)))

        for i, act in enumerate(acts):
            atom, location, coeff, batch_num = act
            # add the atom back
            batch[batch_num], segment = apply_atom(
                batch[batch_num], signal_size, d[atom], location, coeff, np.add)
            segments[i] = segment
            coeffs[i] = coeff

        # update the atom and give it unit norm
        d[atom_index, :] = unit_norm(np.dot(segments.T, coeffs))

        # finally subtract the updated atom from the residual
        for i, act in enumerate(acts):
            atom, location, coeff, batch_num = act
            batch[batch_num], _ = apply_atom(
                batch[batch_num], signal_size, d[atom], location, coeff, np.subtract)
