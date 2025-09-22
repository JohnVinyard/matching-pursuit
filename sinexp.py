from matplotlib import pyplot as plt
import numpy as np


if __name__ == '__main__':
    n_samples = 2 ** 13
    samplerate = 22050
    nyquist = samplerate // 2
    freq = 0.01 * nyquist
    t = np.linspace(0, np.pi, n_samples)
    decay = np.linspace(1, 0, n_samples) ** 50
    signal = np.sin(t * freq)

    tone = signal
    decaying_tone = signal * decay

    decay_spec = np.fft.rfft(decay, norm='ortho')
    tone_spec = np.fft.rfft(tone, norm='ortho')
    decaying_tone_spec = np.fft.rfft(decaying_tone, norm='ortho')

    plt.plot(decaying_tone)
    plt.title('Decaying Tone - time domain')
    plt.show()

    plt.plot(decay)
    plt.title('Envelope - time domain')
    plt.show()

    plt.plot(np.abs(decay_spec))
    plt.title('Envelope - frequency domain')
    plt.show()

    plt.plot(np.abs(tone_spec))
    plt.title('Tone - frequency domain')
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(decaying_tone_spec.real, decaying_tone_spec.imag)
    ax.set_xlim(-3, 3)  # x-axis from 0 to 10
    ax.set_ylim(-3, 3)  # y-axis from 0 to 10
    plt.show()

    # plt.plot(np.abs(decaying_tone_spec))
    # plt.title('Decaying Tone - frequency domain mag')
    # plt.show()
    #
    # plt.plot(np.unwrap(np.angle(decaying_tone_spec)))
    # plt.title('Decaying Tone - frequency domain phase')
    # plt.show()

    # dfd = decay_spec * tone_spec
    # dfd[:16] = 0
    # plt.plot(np.abs(dfd))
    # plt.title('Decaying Tone - synthetic frequency domain')
    # plt.show()
    #
    # dfd = np.fft.irfft(dfd, norm='ortho')
    # plt.plot(dfd)
    # plt.title('Decaying Tone - synthetic time domain')
    # plt.show()