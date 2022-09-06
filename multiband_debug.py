import zounds
import torch
import numpy as np
from modules.stft import morlet_filter_bank
from util.playable import playable
from modules import fft_frequency_decompose, fft_frequency_recompose


def fir_delay():
    samplerate = zounds.SR22050()

    bank = morlet_filter_bank(samplerate, 512, zounds.MelScale(zounds.FrequencyBand(20, samplerate.nyquist), 128), 0.1).real
    bank = bank.reshape(-1, 512)[:50, :].sum(axis=0)
    bank = np.pad(bank, [(0, (2**15) - 512)])
    bank_spec = np.fft.rfft(bank, axis=-1, norm='ortho')

    synth = zounds.TickSynthesizer(samplerate)
    samples = synth.synthesize(samplerate.duration * (2**15), zounds.Seconds(1))
    samples = np.pad(samples, [(0, 1)])

    delay = np.zeros(2**15)

    d = 250
    periodic = delay[::d]
    delay[::d] = 1

    damping = np.array([0.95] * periodic.shape[0])
    damping = np.cumprod(damping)
    delay[::d] *= damping
    

    sample_spec = np.fft.rfft(samples, axis=-1, norm='ortho')
    delay_spec = np.fft.rfft(delay, axis=-1, norm='ortho')
    spec = sample_spec * delay_spec * bank_spec

    final = np.fft.irfft(spec, axis=-1, norm='ortho')

    # final = (samples + final) * 0.5

    return zounds.AudioSamples(final, samplerate).pad_with_silence(), delay
    

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread()

    # samplerate = zounds.SR22050()

    # synth = zounds.SineSynthesizer(zounds.SR22050())
    # samples = synth.synthesize(samplerate.duration * (2**15), [440])
    # samples = torch.from_numpy(samples).view(1, 1, 2**15) 


    # bands = fft_frequency_decompose(samples, 512)
    # print(len(bands))

    # new_bands = {k: torch.zeros_like(v).uniform_(-0.01, 0.01) for k, v in bands.items()}

    # recon = fft_frequency_recompose(new_bands, 2**15)


    recon, delay = fir_delay()

    recon = playable(recon, zounds.SR22050())
    spec = np.abs(zounds.spectral.stft(recon))

    input('waiting...')