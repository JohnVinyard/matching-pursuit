import zounds
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from scipy.signal.windows import tukey
from torch.nn import functional as F

from modules.stft import stft

sr = zounds.SR22050()
n_samples = 2**14

a_note = 440
c_note = 261.625565


def make_signal():
    synth = zounds.SineSynthesizer(sr)

    duration = sr.frequency * (n_samples / 2)
    first_part = synth.synthesize(duration, [a_note / 2, a_note, a_note * 2])
    second_part = synth.synthesize(duration, [c_note / 2, c_note, c_note * 2])

    window = tukey(n_samples // 2)

    full = np.concatenate([first_part * window, second_part * window])
    samples = zounds.AudioSamples(full, sr)

    spec = np.abs(zounds.spectral.stft(samples))

    t = torch.from_numpy(samples).float()

    return samples, spec, t


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        init_value = 0.01
        self.latent = nn.Parameter(
            torch.FloatTensor(6, 32).uniform_(-init_value, init_value))


    def forward(self, x):
        x = torch.clamp(self.latent, 0, 1)
        amp = x[:3, :]
        freq = x[3:, :]

        amp = F.upsample(amp[None, ...], size=n_samples, mode='linear').view(3, n_samples)
        freq = F.upsample(freq[None, ...], size=n_samples, mode='linear').view(3, n_samples)

        signal = torch.sin(torch.cumsum(freq, axis=-1) * np.pi) * amp
        signal = signal.mean(axis=0)
        return signal


def time_domain_loss(inp, t):
    return F.mse_loss(inp, t)

def spectral_mag_loss(inp, t):
    inp = stft(inp)
    t = stft(t)
    return F.mse_loss(inp, t)

def fake_spec():
    audio = zounds.AudioSamples(result.data.cpu().numpy().squeeze(), sr)
    sp = np.abs(zounds.spectral.stft(audio))
    return sp

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread()

    audio, spectrogram, target = make_signal()

    model = Model()
    optim = Adam(model.parameters(), lr=1e-3, betas=(0, 0.9))

    while True:
        optim.zero_grad()
        result = model.forward(None)
        loss = spectral_mag_loss(result, target)
        loss.backward()
        optim.step()
        print(loss.item())

        fake_spec()

    input('Waiting...')
