import torch
from torch.optim.adam import Adam
import zounds
from torch import nn

from datastore import batch_stream
from torch.nn import functional as F
import numpy as np

sr = zounds.SR22050()
batch_size = 64
n_samples = 2**15
do_cumsum = False
final_activation = False
n_embedding_freqs = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


path = '/home/john/workspace/audio-data/musicnet/train_data'


torch.backends.cudnn.benchmark = True




def transform(x):
    coeffs = np.fft.rfft(x, axis=-1)
    mag = np.abs(coeffs)
    phase = np.angle(coeffs)
    return mag, phase


def inverse_transform(mag, phase):
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    coeffs = mag * np.exp(1j * phase)
    x = np.fft.irfft(coeffs, axis=-1)
    return x



def get_batch():
    sig = next()
    mag, phase = transform(sig)
    return mag, phase

"""

overall
====================
envelope = 32 [0 - 1] envelope * (harmonic + noise)

harmonic
=====================================
envelope = 32 [0 - 1]
f0 = 32 [0 - 1] - what's a good max frequency here?
harmonics = [32] - multiples of f0 [1-10]
harmonic_amp = [32 x 32] - [0 - 1] to allow for evolving harmonic relationships, also multiples of envelope

noise
=============================
envelope = 32 [0 - 1]
noise = 32 * 32 unit-norm vectors
"""

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    stream = batch_stream(path, '*.wav', batch_size, n_samples)

    for batch in stream:
        orig = zounds.AudioSamples(batch[0].squeeze(), sr).pad_with_silence()


        m, p = transform(batch)

        indices = np.argsort(m, axis=-1)[:, :-8192]
        m[:, indices] = 0

        r = inverse_transform(m, p)

        a = zounds.AudioSamples(r[0].squeeze(), sr).pad_with_silence()
        input('Next..')