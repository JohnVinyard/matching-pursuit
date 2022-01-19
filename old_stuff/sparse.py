
import librosa
from matplotlib import pyplot as plt
from soundfile import write
import numpy as np
import os


filename = 'Kevin_MacLeod_-_P_I_Tchaikovsky_Dance_of_the_Sugar_Plum_Fairy.ogg'
signal, sr = librosa.load(filename, mono=True)

num = 512

# the max size of an atom
atom_size = num

# the number of atoms
n_components = 517

# the number of samples to train against.  This could
# really be anything, but should at least be as large
# as atom_size
n_samples = num

# the number of segments to train on at a time
batch_size = 8

# the number of sparse coding iterations to use
# before each dictionary update
sparse_code_iterations = 32


def make_d():
    """
    Make a dictionary that is (n_components * atom_size)
    """
    d = []
    for i in range(n_components):
        start = np.random.randint(0, atom_size - 1)
        size = np.random.randint(1, atom_size - start)
        atom = np.random.normal(0, 1e-8, atom_size)
        atom[start: start + size] = np.random.normal(0, 1, size)
        d.append(atom[None, :])
    d = np.concatenate(d, axis=0)
    d = unit_norm(d)
    return d


def get_batch(size):
    batch = []
    for i in range(size):
        pos = np.random.randint(0, len(signal) - atom_size)
        sig = signal[None, pos: pos + atom_size]
        # sig = sig / (sig.max() + 1e-12)
        batch.append(sig)
    sig = np.concatenate(batch, axis=0)
    sig = unit_norm(sig)
    return sig


def batch_fft_convolve1d(x, y):

    half_width = n_samples // 2

    px = np.pad(x, [(0, 0), (half_width, half_width)])
    py = np.pad(y, [(0, 0), (0, n_samples)])

    fpx = np.fft.rfft(px, axis=-1)
    fpy = np.fft.rfft(py[..., ::-1], axis=-1)

    c = fpx[:, None, :] * fpy[None, :, :]

    c = np.fft.irfft(c, axis=-1)

    c = np.fft.fftshift(c, axes=-1)
    c = c[..., n_samples - 1:-1]

    return c


def unit_norm(x):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-8)


def remove_atom(d, signal, atom, time, activation):

    half_width = atom_size // 2

    residuals = []

    i = 0
    for at, act, t in zip(atom, activation, time):
        start = t - half_width
        end = t + half_width

        sig_st = max(start, 0)
        sig_end = min(end, n_samples)

        at_st = max(-start, 0)
        at_end = at_st + (sig_end - sig_st)

        a = (d[at] * act)

        partial = np.zeros(n_samples)
        partial[sig_st:sig_end] = a[at_st:at_end]

        residual = signal[i] - partial
        residuals.append(residual[None, ...])
        i += 1

    return np.concatenate(residuals, axis=0)


def update_dict_2(d, coding, residual):
    residual = np.pad(residual, [(0, 0), (atom_size, atom_size)])

    for i in range(n_components):
        active = coding[:, i, :]
        
        if active.sum() == 0:
            continue

        batch, pos = np.nonzero(active)

        acts = []

        for b, p in zip(batch, pos):
            acts.append(coding[b, i, p])



def update_dict(d, coding, signal):
    print(d.shape, coding.shape, signal.shape)

    """
    d = (n_components, atom_size)
    coding = (batch, n_components, signal_size)
    signal AKA residual = (batch, signal_size)

    """

    for i in range(n_components):
        active = coding[:, i, :]

        if active.sum() == 0:
            # no instances of this atom were employed for this batch
            continue

        batch, pos = np.nonzero(active)

        # for each instance of activity, get an aligned signal
        half_width = atom_size // 2

        sig = []
        acts = []

        for b, p in zip(batch, pos):
            acts.append(coding[b, i, p])

            start = p - half_width
            end = p + half_width

            if start < 0:
                left_padding = np.abs(start)
                slice_start = 0
            else:
                left_padding = 0
                slice_start = start

            if end >= n_samples:
                right_padding = end - n_samples
            else:
                right_padding = 0

            slice_end = slice_start + atom_size
            s = np.pad(signal[b], [(left_padding, right_padding)])

            sig.append(s[slice_start: slice_end][None, :])

        acts = np.array(acts)

        # pull together the signals aligned with the atom
        sig = np.concatenate(sig, axis=0)

        # add the activations back to the aligned residuals
        with_atom = sig + (d[i, :] * acts[:, None])

        # dictionary[:, k] = np.dot(R, code[k, :])
        # print(with_atom.shape, acts.shape, coding[:, i, :].shape)


        # update this atom
        d[i, :] = unit_norm(np.dot(with_atom.T, acts))

        


def loss(d, b):
    act = np.zeros((batch_size, n_components * n_samples))

    start_norm = np.linalg.norm(b)

    for i in range(sparse_code_iterations):

        # convolve atoms with signals in the frequency domain
        fm = batch_fft_convolve1d(b, d)

        # find the max atom and position for each example
        flattened = fm.reshape((batch_size, -1))
        mx = np.argmax(flattened, axis=1)

        activation = flattened[np.arange(batch_size), mx]

        act[np.arange(batch_size), mx] += flattened[np.arange(batch_size), mx]

        atom = mx // atom_size
        time = mx % atom_size

        b = remove_atom(d, b, atom, time, activation)
    
    print(f'{start_norm} -> {np.linalg.norm(b)}')

    update_dict(d, act.reshape((batch_size, n_components, n_samples)), b)

    return d, np.linalg.norm(b)


def plot_atoms(d, orig):
    fig, axs = plt.subplots(4, 4, figsize=(15, 15))

    indices = np.random.randint(0, n_components, 16)

    for i in range(4):
        for j in range(4):
            # axs[i, j].plot(orig[indices[i * 4 + j]])
            axs[i, j].plot(d[indices[i * 4 + j]])
            axs[i, j].tick_params(
                left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)
    plt.savefig('atoms.png')


if __name__ == '__main__':
    d = make_d()
    orig = d.copy()

    i = 0
    while True:
        b = get_batch(batch_size)
        d, _ = loss(d, b)
        if i % 50 == 0:
            plot_atoms(d, orig)
            np.save('D.dat', d)
        i += 1
