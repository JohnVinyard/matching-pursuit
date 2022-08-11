import numpy as np
import zounds


def playable(x, samplerate):
    if not isinstance(x, np.ndarray):
        x = x.data.cpu().numpy()
    
    if len(x.shape) != 1:
        x = x[0].reshape(-1)
    
    return zounds.AudioSamples(x, samplerate).pad_with_silence()
