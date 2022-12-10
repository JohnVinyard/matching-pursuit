import zounds
import numpy as np

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    samplerate = zounds.SR22050()
    n_samples = 2**15

    impulse = np.random.uniform(-1, 1, n_samples) * np.hamming(n_samples)

    start_hz = 220
    end_hz = 300

    start_radians = (start_hz / samplerate.nyquist) * np.pi
    stop_radians = (end_hz / samplerate.nyquist) * np.pi

    tremolo_freq = 6
    tremolo_radians = (tremolo_freq / samplerate.nyquist) * np.pi
    f = np.zeros(n_samples)
    f[:] = tremolo_radians
    tr = np.sin(np.cumsum(f)) * 0.002

    # freqs = np.linspace(start_radians, stop_radians, n_samples)

    freqs = start_hz + tr
    signal = np.sin(np.cumsum(freqs))

    impulse_spec = np.fft.rfft(impulse, axis=-1, norm='ortho')
    signal_spec = np.fft.rfft(signal, axis=-1, norm='ortho')

    conv = signal_spec * impulse_spec

    final = np.fft.irfft(conv)

    samples = zounds.AudioSamples(final, samplerate).pad_with_silence()

    spec = np.abs(zounds.spectral.stft(samples))

    input('waiting...')