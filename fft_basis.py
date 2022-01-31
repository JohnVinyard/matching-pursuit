from config import Config
import numpy as np
import zounds
from data.datastore import batch_stream
from scipy.signal import morlet

def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def morlet_filter_bank(
        samplerate,
        kernel_size,
        scale,
        scaling_factor,
        normalize=True):
    
    basis_size = len(scale)
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    try:
        if len(scaling_factor) != len(scale):
            raise ValueError('scaling factor must have same length as scale')
    except TypeError:
        scaling_factor = np.repeat(float(scaling_factor), len(scale))

    sr = int(samplerate)

    for i, band in enumerate(scale):
        scaling = scaling_factor[i]
        w = band.center_frequency / (scaling * 2 * sr / kernel_size)
        basis[i] = morlet(
            M=kernel_size,
            w=w,
            s=scaling)

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8

    return basis

def fft_basis(fft_size):
    n = np.arange(fft_size)
    k = n.reshape((fft_size, 1))
    basis = np.exp(-2j * np.pi * k * n / fft_size)
    return basis

def geom_basis(fft_size):
    # x = np.geomspace(1, fft_size, fft_size)
    # # k = n.reshape((fft_size, 1))
    # # basis = np.exp(-2j * np.pi * k * n / fft_size)
    # basis = np.pi**-0.25 * (np.exp(1j*w*x) - exp(-0.5*(w**2))) * np.exp(-0.5*(x**2))
    # return basis
    sr = zounds.SR22050()
    return morlet_filter_bank(
        sr, 
        fft_size, 
        zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), fft_size), 
        np.linspace(0.05, 0.99, fft_size))


def short_time_transform(x, ws=512, ss=256, basis_func=None):
    basis = basis_func(ws)

    windowed = zounds.nputil.sliding_window(x, ws, ss)
    windowed *= np.hamming(ws)[None, :]
    freq_domain = np.dot(windowed, basis.T)

    return freq_domain



if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    sr = zounds.SR22050()
    n_samples = 2**15
    stream = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)
    samples = next(stream).reshape((n_samples,))

    linear_basis = fft_basis(512)
    g_basis = geom_basis(512)

    linear_transform = short_time_transform(samples, basis_func=fft_basis)
    geom_transform = short_time_transform(samples, basis_func=geom_basis)

    baseline = np.abs(zounds.spectral.stft(zounds.AudioSamples(samples, sr)))

    linear_spec = np.abs(linear_transform)[:, :257]
    geom_spec = np.abs(geom_transform)[:, :257]


    input('waiting...')