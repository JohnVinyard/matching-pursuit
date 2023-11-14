import zounds
import torch
from torch.nn import functional as F
import numpy as np

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)
    n_samples = 2 ** 15

    samplerate = zounds.SR22050()
    synth = zounds.SineSynthesizer(samplerate)
    signal = synth.synthesize(samplerate.frequency * n_samples, [220, 440, 880])

    env = ((np.linspace(1, 0, n_samples) ** 5) * 1)

    ea = signal * env


    # 513
    # sig_spec = np.fft.rfft(signal)

    # ???
    # env_spec = np.fft.rfft(env)
    

    # conv = sig_spec * env_spec[:sig_spec.shape[0]]

    # final = np.zeros_like(env_spec)
    # final[:sig_spec.shape[0]] = conv

    # ea = np.fft.irfft(final)
    ea = zounds.AudioSamples(ea, samplerate)
    print(ea.shape)

    input('waiting...')

