import numpy as np
import zounds


def playable(x, samplerate, normalize=False):
    if not isinstance(x, np.ndarray):
        x = x.data.cpu().numpy()
    
    if len(x.shape) != 1:
        x = x[0].reshape(-1)
    
    samples  = zounds.AudioSamples(x, samplerate).pad_with_silence()
    if normalize:
        mx = samples.max()
        samples = samples / (mx + 1e-8)
    return samples
