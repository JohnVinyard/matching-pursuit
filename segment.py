import numpy as np
import torch
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from data import get_one_audio_segment
from modules import stft

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt

if __name__ == '__main__':
    n_samples = 2**15
    signal = get_one_audio_segment(n_samples=2**15)
    signal = signal.view(-1, 1, n_samples)

    spec = stft(signal, ws=2048, step=256, pad=True, log_amplitude=True).view(1, -1, 1025)

    freq = torch.linspace(0, 1, spec.shape[-1], device=signal.device)
    time = torch.linspace(0, 1, spec.shape[1], device=signal.device)

    t = torch.zeros(spec.shape[0], spec.shape[1], 1025, 3).to(signal.device)
    t[:, :, :, 0:1] += time[None, :, None, None]
    t[:, :, :, 1:2] += freq[None, None, :, None]
    t[:, :, :, 2:3] += spec[..., None]

    t = t.data.cpu().numpy()
    t /= t.max()

    plt.matshow(t[0])
    plt.show()

    orig_shape = t.shape
    t = t.reshape((-1, 3))
    kmeans = KMeans(n_clusters=64)
    kmeans.fit(t)
    centers = kmeans.cluster_centers_
    indices = kmeans.predict(t)
    quantized = centers[indices]
    quantized = quantized.reshape(orig_shape)

    plt.matshow(quantized[0])
    plt.show()