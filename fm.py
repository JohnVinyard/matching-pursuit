import numpy as np
import zounds
import torch
from torch import nn
from data.audiostream import audio_stream
from modules.ddsp import band_filtered_noise
from modules.pif import AuditoryImage
from train.optim import optimizer

from util import playable

from torch.nn import functional as F

n_samples = 2 ** 15
n_frames = n_samples // 256
sr = zounds.SR22050()

band = zounds.FrequencyBand(30, sr.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    sr, 512, scale, 0.1, normalize_filters=True, a_weighting=False)
aim = AuditoryImage(512, 128, do_windowing=True, check_cola=True)


def feature(x):
    x = fb.forward(x, normalize=False)
    x = aim(x)
    return x


def feat_loss(inp, target):
    inp = feature(inp)
    target = feature(target)
    return F.mse_loss(inp, target)


class FreqModModel(nn.Module):

    def __init__(
            self,
            n_osc=8,
            freq_hz_range=(40, 4000),
            samplerate=zounds.SR22050(),
            reduce=torch.mean):

        super().__init__()
        self.freq_hz_range = freq_hz_range
        # self.n_osc = 8

        # self.baselines = nn.Parameter(torch.zeros(n_osc, 2).normal_(0, 1))

        self.min_freq = self.freq_hz_range[0] / samplerate.nyquist
        self.max_freq = self.freq_hz_range[1] / samplerate.nyquist
        self.freq_range = self.max_freq - self.min_freq
        self.reduce = reduce

        self.dim = 10

    def _translate(self, x, translate=0, scale=1):
        angle = torch.angle(torch.complex(
            x[:, :, 0, :], x[:, :, 1, :])) / np.pi
        return translate + (angle * scale)

    def forward(self, x):
        batch, n_osc, channels, time = x.shape

        # amp and f0
        basic = x[:, :, 0:2, :]
        # baselines = self.baselines[None, :, :, None]
        # basic = (basic * 0.1) + baselines

        mix = x[:, :, 2:4, :]
        noise_std = x[:, :, 4:6, :]
        mod_factor = x[:, :, 6:8, :]
        b = x[:, :, 8: 10, :]

        amp = torch.norm(basic, dim=2)  # (batch, n_osc, 1, time)

        amp = F.interpolate(amp, size=n_samples, mode='linear')

        # (batch, n_osc, time)
        f0 = self._translate(basic, self.min_freq, self.freq_range)
        mix = self._translate(mix, 1, 0.5)
        mix = F.interpolate(mix, size=n_samples, mode='linear')

        harm_amp = amp * mix
        noise_amp = amp * (1 - mix)

        noise_std = self._translate(noise_std, self.min_freq, self.freq_range)
        mod_factor = self._translate(mod_factor, 0.25, 4.75)
        b = self._translate(b, 0.1, 9.9)


        bfn = band_filtered_noise(n_samples, mean=f0, std=noise_std)
        noise = noise_amp * bfn

        f0 = F.interpolate(f0, size=n_samples, mode='linear')
        mod_factor = F.interpolate(mod_factor, size=n_samples, mode='linear')
        b = F.interpolate(b, size=n_samples, mode='linear')

        mod = b * torch.sin(torch.cumsum(mod_factor, dim=-1))

        sig = harm_amp * torch.sin(torch.cumsum(f0, axis=-1) + mod)

        mixed = noise + sig

        summed = self.reduce(mixed, axis=1, keepdim=True)

        return summed


def fm_synth(freq_hz=220, mod_factor=1, mod_amp=1, amp=0.1):
    """
    https://en.wikipedia.org/wiki/Frequency_modulation_synthesis#Spectral_analysis
    """

    freq_radians = freq_hz / sr.nyquist
    mod_radians = freq_radians * mod_factor

    mod_accum = np.zeros((n_samples,))
    mod_accum[:] = mod_radians
    mod = mod_amp * np.sin(np.cumsum(mod_accum, axis=-1))

    accum = np.zeros((n_samples,))
    accum[:] = freq_radians

    sig = amp * np.sin(np.cumsum(accum, axis=-1) + mod)

    return sig


def make(freq, factor, b=1):
    sig = fm_synth(freq, factor, mod_amp=b)
    p = playable(sig[None, ...], sr)
    spec = np.log(1e-8 + np.abs(zounds.spectral.stft(p)))
    return p, spec


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_voices = 3
        self.inputs = nn.Parameter(torch.zeros(
            (1, self.n_voices, 10, n_frames)).normal_(0, 1))
        self.fm = FreqModModel(freq_hz_range=(40, 900))

    def forward(self, x):
        x = self.fm.forward(self.inputs)
        return x


model = Model()
optim = optimizer(model, lr=1e-4)

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    stream = audio_stream(1, n_samples, overfit=True,
                          normalize=True, as_torch=True)
    batch = next(stream)

    def real():
        return playable(batch, sr)

    def fake():
        return playable(recon, sr)

    while True:
        optim.zero_grad()
        recon = model.forward(None)
        loss = feat_loss(recon, batch)
        loss.backward()
        optim.step()

        print(loss.item())

    # input('waiting...')
