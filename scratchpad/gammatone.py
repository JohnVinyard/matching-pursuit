import numpy as np
from scipy.signal import gammatone
from matplotlib import pyplot as plt

# from modules.normalization import max_norm


def make_filterbank(n_filters, kernel_size):
    return np.stack([
        gammatone(
            freq=x, 
            ftype='fir', 
            order=4, 
            numtaps=kernel_size, 
            fs=22050)[0] for x in np.geomspace(20, 11000, n_filters)
        ], axis=0)


if __name__ == '__main__':
    bank = make_filterbank(64, 512)
    bank = bank / (bank.max(axis=1, keepdims=True) + 1e-8)
    # spec = np.abs(np.fft.rfft(bank, axis=-1, norm='ortho'))

    for filt in bank:
        plt.plot(filt)
        plt.show()