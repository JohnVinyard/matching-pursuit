import zounds
import numpy as np

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    samplerate = zounds.SR22050()
    synth = zounds.SineSynthesizer(samplerate)

    signal = synth.synthesize(zounds.Seconds(10), [110, 220, 440, 880])

    spec = zounds.spectral.stft(signal)

    freqs = np.fft.rfftfreq(1024)[1:]


    mag = np.abs(spec)
    phase = np.angle(spec)
    phase = np.diff(phase, axis=-1) * freqs[None, :]

    input('waiting...')
