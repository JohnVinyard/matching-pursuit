import matplotlib
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from torch import nn
from torch.optim import Adam
from config.dotenv import Config
import zounds
import numpy as np
from scipy.signal import morlet
import torch
from torch.nn import functional as F

from data.datastore import batch_stream
from modules.ddsp import band_filtered_noise
from modules.psychoacoustic import PsychoacousticFeature
from modules.stft import stft_relative_phase
from util import device

# w = band.center_frequency / (scaling * 2 * sr / kernel_size)


def filter(support, frequency, scaling):
    if frequency <= 0 or frequency > 1:
        raise ValueError(
            'Frequency should range between 0 - 1, with 1 representing the nyquist frequency')

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


# stft
# windowed = zounds.nputil.sliding_window(samples, 512, 256)
# windowed = windowed * np.hamming(512)[None, ...]
# spec = np.fft.fft(windowed, axis=-1)

# # FFT of complex filter bank
# bank = filter_bank(512, 512)
# spec_bank = np.fft.fft(bank, axis=-1)

# # convolve in frequency domain
# conv = np.matmul(spec_bank, spec.T)

# # return result to (complex) time domain
# spectrogram = np.fft.ifft(conv, axis=-1)
# print(spectrogram.shape, spectrogram.dtype)

# w = np.log(0.01 + np.abs(spectrogram))
# stft = np.log(0.01 + np.abs(spec))[: ,:256]


def image(timestep):
    chunks = []
    for k, v in f.items():
        chunks.append(v[:, timestep, :])

    full = np.concatenate(chunks, axis=-1)
    return np.log(0.01 + full)


def make_spec():
    chunks = []
    for k, v in f.items():
        chunks.append(np.linalg.norm(v, axis=-1))
    
    spec = np.concatenate(chunks, axis=0)
    spec = np.log(0.01 + spec).T

    return spec

def spectrograms():
    loc = 880 / sr.nyquist
    samples = band_filtered_noise(
        2**15, 512, 256, mean=loc, std=0.001).data.cpu().numpy().squeeze()
    noise_samples = zounds.AudioSamples(samples, sr).pad_with_silence()
    noise_spec = np.abs(zounds.spectral.stft(noise_samples))

    synth = zounds.SineSynthesizer(sr)
    samples = synth.synthesize(sr.frequency * 2**15, [880]).astype(np.float32)
    sine_samples = zounds.AudioSamples(
        samples.squeeze(), sr).pad_with_silence()
    sine_spec = np.abs(zounds.spectral.stft(sine_samples))

    return noise_spec, noise_samples, sine_spec, sine_samples


if __name__ == '__main__':

    # app = zounds.ZoundsApp(locals=locals(), globals=globals())
    # app.start_in_thread()

    sr = zounds.SR22050()
    stream = batch_stream(Config.audio_path(), '*.wav', 1, 2**15)

    samples = next(stream)
    # ns, n, ss, s = spectrograms()
    # input('----')
    # exit()


    # loc = 880 / sr.nyquist
    # samples = band_filtered_noise(
    #     2**15, 512, 256, mean=loc, std=0.001).data.cpu().numpy()
    # print('noise', samples.shape)

    # synth = zounds.SineSynthesizer(sr)
    # samples = synth.synthesize(sr.frequency * 2**15, [880]).astype(np.float32)
    # print('sine', samples.shape)

    orig = zounds.AudioSamples(samples.squeeze(), sr).pad_with_silence()
    orig.save('sound.wav')

    samples = torch.from_numpy(samples).view(1, 1, 2**15)

    feature = PsychoacousticFeature()
    feat = feature.compute_feature_dict(samples)

    f = {k: v.data.cpu().numpy().squeeze() for k, v in feat.items()}

    # TODO: helper function for making a movie
    matplotlib.use('tkagg', force=True)
    from matplotlib import pyplot as plt

    data = []
    for i in range(32):
        data.append(image(i)[None, ...])
    data = np.concatenate(data, axis=0)

    print(data.shape)
    spec = make_spec()
    print(spec.shape)
    plt.matshow(spec)

    fig = plt.figure()
    plot = plt.imshow(data[0])

    def init():
        plot.set_data(data[0])
        return [plot]

    def update(frame):
        plot.set_data(data[frame])
        return [plot]

    ani = FuncAnimation(
        fig,
        update,
        frames=np.arange(0, 32, 1),
        init_func=init,
        blit=True,
        interval=46)
    plt.show()
