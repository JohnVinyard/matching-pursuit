# from matplotlib import pyplot as plt
import numpy as np
import zounds

import matplotlib
matplotlib.use('qt5agg', force=True)
import matplotlib.pyplot as plt

def midi_to_hz(n: int):
    return 440 * (2 ** ((n - 69) / 12))


def musical_scale(start_midi=1, stop_midi=129):
    x = midi_to_hz(np.arange(start_midi, stop_midi))
    for freq in x:
        yield zounds.FrequencyBand(freq - 1, freq + 1)


class MusicalScale(zounds.FrequencyScale):
    def __init__(self, start_midi=1, stop_midi=129):
        x = list(musical_scale(start_midi, stop_midi))
        band = zounds.FrequencyBand(x[0].center_frequency, x[-1].center_frequency)
        super().__init__(band, len(x))
        self._bands = x
    

if __name__ == '__main__':
    x = midi_to_hz(np.arange(1, 128))

    band = zounds.FrequencyBand(20, zounds.SR22050().nyquist)
    scale = zounds.MelScale(band, 128)

    plt.plot(x, label='musical')
    plt.plot(list(scale.center_frequencies), label='mel')
    plt.legend()
    plt.show()