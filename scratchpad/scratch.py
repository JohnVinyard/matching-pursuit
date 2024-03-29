from time import time
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
from util import device


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



def image(timestep):
    chunks = []
    for k, v in f.items():
        current = v[:, timestep, :]
        current = np.log(1e-4 + v[:, timestep, :])
        # norm = np.linalg.norm(current, axis=-1, keepdims=True)
        chunks.append(current)

        

    full = np.concatenate(chunks, axis=-1)
    return full


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

    # loc = 880 / sr.nyquist
    # samples = band_filtered_noise(
    #     2**15, 512, 256, mean=loc, std=0.001).data.cpu().numpy()
    # print('noise', samples.shape)

    # synth = zounds.SineSynthesizer(sr)
    # samples = synth.synthesize(sr.frequency * 2**15, [880]).astype(np.float32)
    # print('sine', samples.shape)

    orig = zounds.AudioSamples(samples.squeeze(), sr).pad_with_silence()

    samples = torch.from_numpy(samples).view(1, 1, 2**15).to(device)

    feature = PsychoacousticFeature(kernel_sizes=[128] * 6).to(device)
    feat = feature.compute_feature_dict(
        samples, constant_window_size=256, time_steps=64)

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
