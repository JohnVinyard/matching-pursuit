from config.dotenv import Config
import zounds
import numpy as np
from scipy.signal import morlet

from data.datastore import batch_stream

# w = band.center_frequency / (scaling * 2 * sr / kernel_size)

def filter(support, frequency, scaling):
    if frequency <= 0 or frequency > 1:
        raise ValueError('Frequency should range between 0 - 1, with 1 representing the nyquist frequency')
    

    w = (frequency / 2) / (scaling * 2 / support)

    return morlet(support, w, scaling)


def filter_bank(support, n_filters):
    freqs = np.linspace(0.01, 0.99, n_filters)
    scaling = 1
    filters = []
    for freq in freqs:
        filters.append(filter(support, freq, scaling)[None, ...])
    
    bank = np.concatenate(filters, axis=0)
    return bank
    


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread()

    sr = zounds.SR22050()
    stream = batch_stream(Config.audio_path(), '*.wav', 1, 2**15)
    samples = next(stream).reshape(-1)

    # stft
    windowed = zounds.nputil.sliding_window(samples, 512, 256)
    windowed = windowed * np.hamming(512)[None, ...]
    spec = np.fft.fft(windowed, axis=-1)

    # FFT of complex filter bank
    bank = filter_bank(512, 512)
    spec_bank = np.fft.fft(bank, axis=-1)

    # convolve in frequency domain
    conv = np.matmul(spec_bank, spec.T)
    
    # return result to (complex) time domain
    spectrogram = np.fft.ifft(conv, axis=-1)
    print(spectrogram.shape, spectrogram.dtype)

    w = np.log(0.01 + np.abs(spectrogram))
    stft = np.log(0.01 + np.abs(spec))[: ,:256]

    input('waiting...')