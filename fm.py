import numpy as np
import zounds

from util import playable

n_samples = 2 ** 15
sr = zounds.SR22050()

def fm_synth(freq_hz, carrier_factor=1, amp=0.1):
    
    accum = np.zeros((n_samples,))
    accum[:] = (freq_hz / sr.nyquist)

    carrier = accum * carrier_factor
    carrier = np.sin(np.cumsum(carrier, axis=-1))


    signal = amp * np.sin((freq_hz / sr.nyquist) * np.cumsum(carrier, axis=-1))
    return signal



if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    sig = fm_synth(220, 8)

    sig = playable(sig[None, ...], sr)
    spec = np.log(1e-4 + np.abs(zounds.spectral.stft(sig)))

    input('waiting...')