from torch import hann_window
from config import Config
import numpy as np
import zounds
from data.datastore import batch_stream
from scipy.signal import morlet, hann

from modules.ddsp import overlap_add

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

def fft_basis(fft_size, normalize=True):
    n = np.arange(fft_size)
    k = n.reshape((fft_size, 1))
    basis = np.exp(-2j * np.pi * k * n / fft_size)

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8
    
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
        0.1)


def short_time_transform(x, ws=512, ss=256, basis_func=None):
    basis = basis_func(ws)
    windowed = zounds.nputil.sliding_window(x, ws, ss)
    windowed = windowed * hann(ws)[None, :]
    freq_domain = np.dot(windowed, basis.T)

    return windowed, freq_domain


# def flip_inverted(x):
    
#     batch, channels, frames, samples = x.shape
#     step_size = samples // 2

#     x = x.reshape(batch, channels, frames, step_size, 2)
#     x[:, :, :, :, 0] = x[:, :, :, ::-1, 0]
#     # x[:, :, :, :, 1] = x[:, :, :, ::-1, 1]

#     x = x.reshape(batch, channels, frames, samples)
#     return x


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    sr = zounds.SR22050()
    n_samples = 2**15
    stream = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)
    samples = next(stream).reshape((n_samples,))

    orig = zounds.AudioSamples(samples, sr).pad_with_silence()

    linear_basis = fft_basis(512)
    g_basis = geom_basis(512)

    windowed, linear_transform = short_time_transform(samples, basis_func=fft_basis)
    gw, geom_transform = short_time_transform(samples, basis_func=geom_basis)


    linear_inverted = np.dot(linear_transform, linear_basis).real[None, None, ...][..., ::-1]
    geom_inverted = np.dot(geom_transform, g_basis).real[None, None, ...][..., ::-1]
    

    linear_final = zounds.AudioSamples(overlap_add(linear_inverted, apply_window=False).squeeze(), sr).pad_with_silence()
    geom_final = zounds.AudioSamples(overlap_add(geom_inverted, apply_window=False).squeeze(), sr).pad_with_silence()

    baseline = np.abs(zounds.spectral.stft(zounds.AudioSamples(samples, sr)))

    linear_spec = np.abs(linear_transform)[:, :257]
    geom_spec = np.abs(geom_transform)[:, :257]


    input('waiting...')