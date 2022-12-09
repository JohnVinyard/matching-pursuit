import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import resample, stft

'''
Resonator
-------------------
f0 * n_frames
harm_amp * n_harmonics
decay * n_harmonics

Impulse
--------------------
amp * n_frames
freq domain envelope
'''

def naive(impulse, decay, n_harmonics, harmonic_amps):
    current = np.zeros(n_harmonics)
    output = []
    for i in range(len(impulse)):
        current += impulse[i] * harmonic_amps
        current = np.clip(current, 0, 1)
        output.append(current)
        current *= decay
    return np.stack(output)

if __name__ == '__main__':

    # hyperparameters
    samplerate = 22050
    nyquist = samplerate // 2
    seq_len = 256
    n_harmonics = 32
    harmonic_factors = np.arange(1, n_harmonics + 1, step=1)
    n_samples = 2**15
    n_noise_coeffs = n_samples // 2 + 1
    freq_domain_noise_filter_size = 128

    # params
    f0 = 220
    change = np.zeros(seq_len)
    change[:] = 0.2
    change = np.sin(np.cumsum(change)) * 5
    f0 = f0 + change

    osc = f0[None, :] * harmonic_factors[:, None]
    decay = np.linspace(0.99, 0.9, n_harmonics)
    harmonic_amps = np.geomspace(1, 0.95, n_harmonics)
    harmonic_amps[1::2] = 0
    radians = (osc / nyquist) * np.pi
    radians = resample(radians, n_samples, axis=-1)
    # radians = radians[:, None].repeat(n_samples, axis=-1)
    osc = np.sin(np.cumsum(radians, axis=-1))

    impulse = np.zeros(seq_len)
    impulse[1] = 1
    impulse[25:50] = -0.01
    impulse[seq_len // 2] = 1
    impulse[200] = 0.2
    impulse_full = resample(impulse, n_samples)

    n = np.random.uniform(-1, 1, n_samples)
    noise_filter = np.hamming(freq_domain_noise_filter_size)
    n_coeffs = resample(noise_filter, n_noise_coeffs)
    noise_spec = np.fft.rfft(n, norm='ortho')
    spec = n_coeffs * noise_spec
    noise = np.fft.irfft(spec, norm='ortho')
    noise = noise * impulse_full


    # generate    
    x = naive(impulse, decay, n_harmonics, harmonic_amps)
    x = resample(x, n_samples, axis=0).T

    x = x * osc

    x = noise[None, :] + x
    x = np.sum(x, axis=0)

    _, _, spec = stft(x)

    spec = np.flipud(np.abs(spec))

    # display
    plt.matshow(np.abs(spec))
    plt.show()


