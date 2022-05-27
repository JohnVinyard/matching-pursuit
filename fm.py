import numpy as np
import zounds
import torch
from torch import nn
from data.audiostream import audio_stream
from modules import stft
from modules.ddsp import NoiseModel, band_filtered_noise
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer

from util import playable

from torch.nn import functional as F

from util.weight_init import make_initializer

n_samples = 2 ** 15
n_frames = n_samples // 256
sr = zounds.SR22050()

band = zounds.FrequencyBand(30, sr.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    sr, 512, scale, 0.1, normalize_filters=True, a_weighting=False)
aim = AuditoryImage(512, 128, do_windowing=True, check_cola=True)


mel_scale = MelScale()
codec = AudioCodec(mel_scale)


def feature(x):
    # x = fb.forward(x, normalize=False)
    # x = aim(x)
    # x = codec.to_frequency_domain(x.view(-1, n_samples))[..., 0]
    x = stft(x)
    return x


def feat_loss(inp, target):
    inp = feature(inp)
    target = feature(target)
    return F.mse_loss(inp, target)


class HarmonicModel(nn.Module):
    def __init__(
            self,
            n_voices=8,
            n_profiles=16,
            n_harmonics=64,
            freq_hz_range=(40, 4000),
            samplerate=zounds.SR22050(),
            reduce=torch.sum):

        super().__init__()
        self.n_voices = n_voices
        self.freq_hz_range = freq_hz_range
        self.reduce = reduce
        self.n_profiles = n_profiles
        self.samplerate = samplerate
        self.n_harmonics = n_harmonics

        self.min_freq = self.freq_hz_range[0] / self.samplerate.nyquist
        self.max_freq = self.freq_hz_range[1] / self.samplerate.nyquist
        self.freq_interval = self.max_freq = self.min_freq

        # harmonic profiles
        self.profiles = nn.Parameter(torch.zeros(
            n_profiles, n_harmonics).uniform_(0, 0.1))

        self.baselines = nn.Parameter(
            torch.zeros(self.n_voices, 2).uniform_(0, 0.05))

        # harmonic ratios to the fundamental
        self.register_buffer('ratios', torch.arange(2, 2 + self.n_harmonics) ** 2)

    def forward(self, f0, harmonics):
        batch = f0.shape[0]

        f0 = f0.view(f0.shape[0], self.n_voices, 2, -1)
        # f0 = self.baselines[None, :, :, None] + (0.1 * f0)
        harmonics = harmonics.view(
            harmonics.shape[0], self.n_voices, self.n_profiles, -1)

        f0_amp = torch.norm(f0, dim=-2) ** 2
        f0 = (torch.angle(torch.complex(f0[:, :, 0, :], f0[:, :, 1, :])) / np.pi)

        f0 = f0 ** 2
        f0 = self.min_freq + (f0 * self.freq_interval)

        # harmonics are whole-number multiples of fundamental
        harmonic_freqs = f0[:, :, None, :] * self.ratios[None, None, :, None]
        harmonic_freqs = torch.clamp(harmonic_freqs, 0, 1)

        # harmonic amplitudes are factors of fundamental amplitude
        harmonics = harmonics.permute(0, 1, 3, 2)
        harmonics = torch.softmax(harmonics, dim=-1)
        harmonics = harmonics @ self.profiles
        harmonic_amp = harmonics.permute(0, 1, 3, 2)
        harmonic_amp = torch.clamp(harmonic_amp, 0, 1)
        harmonic_amp = f0_amp[:, :, None, :] * harmonic_amp

        full_freq = torch.cat([f0[:, :, None, :], harmonic_freqs], dim=2)
        full_amp = torch.cat([f0_amp[:, :, None, :], harmonic_amp], dim=2)

        full_freq = full_freq.view(
            batch * self.n_voices, self.n_harmonics + 1, n_frames)
        full_amp = full_amp.view(
            batch * self.n_voices, self.n_harmonics + 1, n_frames)

        full_freq = F.interpolate(full_freq, size=n_samples, mode='linear')
        full_amp = F.interpolate(full_amp, size=n_samples, mode='linear')

        signal = full_amp * torch.sin(torch.cumsum(full_freq, dim=-1) * np.pi)

        signal = signal.view(batch, self.n_voices,
                             self.n_harmonics + 1, n_samples)

        signal = torch.sum(signal, dim=(1, 2)).view(batch, 1, n_samples)

        return signal


# class FreqModModel(nn.Module):

#     def __init__(
#             self,
#             n_osc=8,
#             freq_hz_range=(40, 4000),
#             samplerate=zounds.SR22050(),
#             n_samples=n_samples,
#             reduce=torch.mean):

#         super().__init__()
#         self.freq_hz_range = freq_hz_range
#         self.n_osc = 8
#         self.n_samples = n_samples
#         self.baselines = nn.Parameter(torch.zeros(n_osc, 2).normal_(0, 0.5))

#         self.min_freq = self.freq_hz_range[0] / samplerate.nyquist
#         self.max_freq = self.freq_hz_range[1] / samplerate.nyquist
#         self.freq_range = self.max_freq - self.min_freq
#         self.reduce = reduce

#         self.dim = 10

#     def _translate(self, x, translate=0, scale=1, squared=False):
#         angle = torch.angle(torch.complex(
#             x[:, :, 0, :], x[:, :, 1, :])) / np.pi
#         if squared:
#             angle = angle ** 2
#         return translate + (angle * scale)

#     def forward(self, x):
#         batch, n_osc, channels, time = x.shape

#         # amp and f0
#         basic = x[:, :, 0:2, :]
#         baselines = self.baselines[None, :, :, None]
#         basic = (basic * 0.01) + baselines

#         mix = x[:, :, 2:4, :]
#         noise_std = x[:, :, 4:6, :]
#         mod_factor = x[:, :, 6:8, :]
#         b = x[:, :, 8: 10, :]

#         amp = torch.norm(basic, dim=2) ** 2  # (batch, n_osc, 1, time)

#         amp = F.interpolate(amp, size=n_samples, mode='linear')

#         # (batch, n_osc, time)
#         f0 = self._translate(basic, self.min_freq,
#                              self.freq_range, squared=True)
#         mix = self._translate(mix, 1, 0.5)
#         mix = F.interpolate(mix, size=n_samples, mode='linear')

#         harm_amp = amp * mix
#         noise_amp = amp * (1 - mix)

#         noise_std = self._translate(noise_std, self.min_freq, self.freq_range)
#         mod_factor = self._translate(mod_factor, 0.25, 4.75)
#         b = self._translate(b, 0.1, 9.9)

#         mod_factor[:] = 2

#         bfn = band_filtered_noise(n_samples, mean=f0, std=noise_std)
#         noise = noise_amp * bfn

#         f0 = F.interpolate(f0, size=n_samples, mode='linear')
#         mod_factor = F.interpolate(mod_factor, size=n_samples, mode='linear')
#         b = F.interpolate(b, size=n_samples, mode='linear')

#         mod = b * torch.sin(torch.cumsum(mod_factor * f0, dim=-1))

#         sig = harm_amp * torch.sin(torch.cumsum(f0, axis=-1) + mod)

#         mixed = noise + sig

#         summed = self.reduce(mixed, axis=1, keepdim=True)

#         return summed


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
        self.n_voices = 16
        self.n_harmonics = 8
        self.n_profiles = 16
        self.noise_frames = n_frames * 4
        self.noise_channels = 16
        self.n_rooms = 8

        self.f0 = nn.Parameter(torch.zeros(
            1, self.n_voices, 2, n_frames).normal_(0, 0.1))
        self.harm = nn.Parameter(torch.zeros(
            1, self.n_voices, self.n_profiles, n_frames).uniform_(0, 1))
        self.harm_model = HarmonicModel(
            self.n_voices, self.n_profiles, self.n_harmonics)

        self.noise_params = nn.Parameter(torch.zeros(
            1, self.noise_channels, n_frames).normal_(0, 0.1))
        self.noise = NoiseModel(self.noise_channels,
                                n_frames, self.noise_frames, n_samples, 16, squared=True)

        self.verb_params = nn.Parameter(
            torch.zeros(1, self.n_rooms).normal_(0, 1))
        self.mix = nn.Parameter(torch.zeros(1).fill_(0.5))

        self.verb = NeuralReverb(n_samples, self.n_rooms)
        self.apply(make_initializer(0.1))

    def forward(self, x):
        harm = self.harm_model.forward(self.f0, self.harm)
        noise = self.noise(self.noise_params)

        dry = harm + noise
        wet = self.verb.forward(dry, torch.softmax(self.verb_params, dim=1))
        mix = torch.sigmoid(self.mix)

        signal = (dry * mix) + (wet * (1 - mix))
        return signal


model = Model()
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    stream = audio_stream(1, n_samples, overfit=True,
                          normalize=True, as_torch=True)
    batch = next(stream)

    def real():
        return playable(batch, sr)
    
    def real_spec():
        return np.log(1e-4 + np.abs(zounds.spectral.stft(real())))

    def fake():
        return playable(recon, sr)

    def fake_spec():
        return np.log(1e-4 + np.abs(zounds.spectral.stft(fake())))
    
    def profiles():
        return model.harm_model.profiles.data.cpu().numpy()

    while True:
        optim.zero_grad()
        recon = model.forward(None)
        loss = feat_loss(recon, batch)
        loss.backward()
        optim.step()

        print(loss.item())

    # input('waiting...')
