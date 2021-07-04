import numpy as np

# def batch_fft_convolve1d(signal, d):
#     """
#     Convolves the dictionary/filterbank `d` with `signal`

#     Parameters
#     ------------
#     signal - A signal of shape `(batch_size, signal_size)`
#     d - A dictionary/filterbank of shape `(n_components, atom_size)`

#     Returns
#     ------------
#     x - the results of convolving the dictionary with the batched signals
#         of dimensions `(batch, n_components, signal_size)`
#     """
#     signal_size = signal.shape[1]
#     atom_size = d.shape[1]
#     diff = signal_size - atom_size

#     half_width = atom_size // 2

#     # TODO: Is it possible to leverage the zero padding and/or
#     # difference in atom and signal size to avoid multiplying
#     # every frequency position
#     px = np.pad(signal, [(0, 0), (half_width, half_width)])
#     py = np.pad(d, [(0, 0), (0, px.shape[1] - atom_size)])

#     fpx = rfft(px, axis=-1)
#     fpy = rfft(py[..., ::-1], axis=-1)

#     c = fpx[:, None, :] * fpy[None, :, :]

#     c = irfft(c, axis=-1)

#     c = np.roll(c, signal_size - diff, axis=-1)
#     c = c[..., atom_size - 1:-1]

#     return c

def fft_conolve(signal, kernel, axis=-1):
    pass


if __name__ == '__main__':
    oned_kernel = np.random.normal(0, 1, (512, 512))
    oned_signal = np.random.normal(0, 1, (1024,))

    nd_kernel = np.random.normal(0, 1, (512, 64, 8, 17))
    nd_signal = np.random.normal(0, 1, (64, 32, 17))


    print(512 * 32768 * 512)
    print(64 * 8 * 513 * 32768 * 512)