import numpy as np
import zounds

'''
Ideas to improve gradient flow:

https://stackoverflow.com/questions/39049631/how-to-implement-a-cumulative-product-table

- cumulative sum of logarithms
- table of pre-computed cumulative products (resolution, n_samples)
- 
'''

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    samplerate = zounds.SR22050()

    tick_synth = zounds.TickSynthesizer(samplerate)
    sine_synth = zounds.SineSynthesizer(samplerate)

    ticks = tick_synth.synthesize(zounds.Seconds(10), zounds.Seconds(2))

    transfer = sine_synth.synthesize(zounds.Seconds(10), np.array([x for x in range(20, 1000, 150)]))
    transfer[:32] += ticks[:32]
    transfer += np.random.normal(0, 0.25, transfer.shape)
    transfer = transfer * np.linspace(1, 0, transfer.shape[0]) ** 15

    tick_spec = np.fft.rfft(ticks, axis=-1, norm='ortho')
    transfer_spec = np.fft.rfft(transfer, axis=-1, norm='ortho')
    transfer_spec = transfer_spec * (np.linspace(1, 0, transfer_spec.shape[0]) ** 15)

    spec = tick_spec * transfer_spec
    samples = np.fft.irfft(spec, axis=-1, norm='ortho')

    samples = zounds.AudioSamples(samples, samplerate).pad_with_silence()
    spec = np.abs(zounds.spectral.stft(samples))

    input('Waiting...')