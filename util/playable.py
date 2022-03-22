import numpy as np
import zounds


def playable(x, samplerate):
    if not isinstance(x, np.ndarray):
        x = x.data.cpu().numpy()
    return zounds.AudioSamples(x[0].squeeze(), samplerate).pad_with_silence()
