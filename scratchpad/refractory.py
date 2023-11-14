import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    total_size = 8

    ret = np.linspace(1, 0, total_size - 1) ** 2
    sig = np.ones(total_size)
    sig[1:] = -ret

    # control = np.random.binomial(1, p=0.05, size=512)
    control = np.zeros(512)
    control[128: 512] = np.linspace(0, 1, 384)
    plt.plot(control)
    plt.show()

    active = (control > 0).sum()
    print('control active', active)


    padded = np.pad(sig, (0, 512 - total_size))

    ret_spec = np.fft.rfft(padded, norm='ortho')
    control_spec = np.fft.rfft(control, norm='ortho')
    spec = ret_spec * control_spec

    final = np.fft.irfft(spec, norm='ortho')
    final = np.clip(final, 0, np.inf)
    active = (final > 0).sum()
    print('final active', active)

    plt.plot(final)
    plt.show()