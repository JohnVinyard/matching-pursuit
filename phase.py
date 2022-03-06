import numpy as np
import torch
from torch.nn import functional as F
import zounds
from config.dotenv import Config
from data.datastore import batch_stream
import librosa

from modules.ddsp import overlap_add


def stft(audio_batch, window_size, step_size, samplerate):
    batch_size = audio_batch.shape[0]
    audio_batch = F.pad(audio_batch, (0, step_size))
    windowed = audio_batch.unfold(-1, window_size, step_size)
    window = torch.hann_window(window_size).to(audio_batch.device)
    spec = torch.fft.rfft(windowed * window, dim=-1, norm='ortho')
    n_coeffs = (window_size // 2) + 1
    spec = spec.reshape(batch_size, -1, n_coeffs)
    return spec


def istft(spec):
    windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
    signal = overlap_add(windowed[:, None, :, :], apply_window=False)
    return signal


def rfft_freqs(window_size):
    freq_ratios = torch.fft.rfftfreq(window_size)
    freq_ratios[0] = 1e-12
    return freq_ratios


class STFT(object):

    def __init__(self):
        super().__init__()
        self.window_size = 512
        self.step_size = 256
        self.samplerate = zounds.SR22050()

    def to_frequency_domain(self, audio_batch):
        return stft(
            audio_batch, self.window_size, self.step_size, int(self.samplerate))

    def to_time_domain(self, spec):
        return istft(spec)

    @property
    def center_frequencies(self):
        return rfft_freqs(self.window_size)


def mag_phase_decomposition(spec, freqs):
    batch_size = spec.shape[0]
    mag = torch.abs(spec)
    phase = torch.angle(spec)
    phase = phase.data.cpu().numpy()
    phase = np.unwrap(phase, axis=1)
    phase = torch.from_numpy(phase).to(spec.device)
    phase = torch.diff(
        phase,
        dim=1,
        prepend=torch.zeros(batch_size, 1, spec.shape[-1]).to(spec.device))
    phase = phase * freqs[None, None, :]
    return torch.cat([mag[..., None], phase[..., None]], dim=-1)


def mag_phase_recomposition(spec, freqs):
    real = spec[..., 0]
    phase = spec[..., 1]
    imag = torch.cumsum(phase / freqs[None, None, :], dim=1)
    imag = (imag + np.pi) % (2 * np.pi) - np.pi
    spec = real * torch.exp(1j * imag)
    return spec


class CQT(object):
    def __init__(self):
        super().__init__()
        self.n_bins = 256
        self.samplerate = zounds.SR22050()
        self.hop_length = 512
        self.bins_per_octave = 48

    def to_frequency_domain(self, audio_batch):
        specs = []
        ab = audio_batch.data.cpu().numpy()
        for item in ab:
            spec = librosa.cqt(
                item.squeeze(),
                n_bins=self.n_bins, 
                sparsity=0, 
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                scale=True).T[None, ...]
            print(spec.shape)
            specs.append(torch.from_numpy(spec).to(audio_batch.device))
        return torch.cat(specs, dim=0)

    def to_time_domain(self, spec):
        samples = []
        device = spec.device
        spec = spec.data.cpu().numpy()
        for item in spec:
            print(item.shape)
            samp = librosa.icqt(
                item.T, 
                sparsity=0, 
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                scale=True)[None, ...]
            print(samp.shape)
            samp = torch.from_numpy(samp).to(device)
            samples.append(samp)
        return torch.cat(samples, dim=0)

    @property
    def center_frequencies(self):
        print('===================================')
        freqs = librosa.cqt_frequencies(
            self.n_bins, 
            fmin=librosa.note_to_hz('C1'), 
            bins_per_octave=self.bins_per_octave)
        print(freqs)
        freqs /= int(self.samplerate)
        print(freqs)
        return freqs


class AudioCodec(object):
    def __init__(self, short_time_transform):
        super().__init__()
        self.short_time_transform = short_time_transform

    def to_frequency_domain(self, audio_batch):
        spec = self.short_time_transform.to_frequency_domain(audio_batch)
        return mag_phase_decomposition(
            spec, self.short_time_transform.center_frequencies)

    def to_time_domain(self, spec):
        spec = mag_phase_recomposition(
            spec, self.short_time_transform.center_frequencies)
        return self.short_time_transform.to_time_domain(spec)


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    n_samples = 2 ** 15
    samplerate = zounds.SR22050()
    window_size = 512
    step_size = window_size // 2

    stream = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)

    transformer = AudioCodec(CQT())

    while True:
        batch = next(stream)
        o = zounds.AudioSamples(
            batch.squeeze(), samplerate).pad_with_silence()
        batch = torch.from_numpy(batch).float()
        print(batch.shape)
        spec = transformer.to_frequency_domain(batch)

        mag = spec[..., 0].data.cpu().numpy().squeeze()
        phase = spec[..., 1].data.cpu().numpy().squeeze()

        recon = transformer.to_time_domain(spec)
        r = zounds.AudioSamples(recon.data.cpu()\
            .numpy().squeeze(), samplerate).pad_with_silence()
        input('Next...')

    # freq_ratios = torch.fft.rfftfreq(512)
    # frequencies =  freq_ratios * int(samplerate)

    # osc_freqs = list([int(x) for x in frequencies[[1, 10, 50, 100]]])

    # synth = zounds.SineSynthesizer(samplerate)
    # audio = synth.synthesize(zounds.Seconds(5), osc_freqs)

    # ta = torch.from_numpy(audio).reshape(1, 1, -1)
    # spec = to_spectrogram(ta, window_size, step_size, int(samplerate))

    # mag = spec[..., 0].data.cpu().numpy()
    # phase = spec[..., 1].data.cpu().numpy() * (freq_ratios.data.cpu().numpy()[None, None, :])

    # pa = phase[:, :, 1].squeeze()
    # pb = phase[:, :, 10].squeeze()
    # pc = phase[:, :, 50].squeeze()
    # pd = phase[:, :, 100].squeeze()

    input()
