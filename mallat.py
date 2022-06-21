import zounds
import numpy as np

samplerate = zounds.SR22050()


def sine(amp, duration, freq):
    accum = np.zeros(duration)
    accum[:] = freq
    accum = np.cumsum(accum)
    return amp * np.sin(accum * np.pi)


def silence(duration):
    return np.zeros(duration)


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    signal = zounds.AudioSamples(np.concatenate([

        silence(8192),
        sine(1, 8192, 440 / samplerate.nyquist),

        silence(8192),
        sine(1 + sine(0.1, 8192, 15 / samplerate.nyquist), 8192, 440),

        silence(8192),
        sine(1, 8192, (440 / samplerate.nyquist) + sine(0.1, 8192, 15)),

        silence(8192)
    ]), samplerate)

    input('waiting...')
