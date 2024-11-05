import numpy as np

def midi_to_hz(n: int):
    return 440 * (2 ** ((n - 69) / 12))


class FrequencyBand(object):
    def __init__(self, start_hz, stop_hz):
        super().__init__()
        self.start_hz = start_hz
        self.stop_hz = stop_hz
    
    @property
    def center_frequency(self):
        return self.stop_hz - self.start_hz


def musical_scale(start_midi=1, stop_midi=129):
    x = midi_to_hz(np.arange(start_midi, stop_midi))
    for freq in x:
        yield FrequencyBand(freq - 1, freq + 1)


def musical_scale_hz(start_midi=21, stop_midi=106, n_steps=512) -> np.ndarray:
    return midi_to_hz(np.linspace(start_midi, stop_midi, n_steps))

