import numpy as np
import torch
from torch.nn import functional as F
import zounds
from config.dotenv import Config
from data.datastore import batch_stream
import librosa
from scipy.signal import morlet, hann

from modules.ddsp import overlap_add


def windowed_audio(audio_batch, window_size, step_size):
    audio_batch = F.pad(audio_batch, (0, step_size))
    windowed = audio_batch.unfold(-1, window_size, step_size)
    window = torch.hann_window(window_size).to(audio_batch.device)
    return windowed * window


def stft(audio_batch, window_size, step_size, samplerate):
    # TODO: Use windowed_audio utility method from above
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


def mag_phase_decomposition(spec, freqs):
    batch_size = spec.shape[0]
    mag = torch.abs(spec)
    phase = torch.angle(spec)

    # unwrap
    phase = phase.data.cpu().numpy()
    phase = np.unwrap(phase, axis=1)
    phase = torch.from_numpy(phase).to(spec.device)

    phase = torch.diff(
        phase,
        dim=1,
        prepend=torch.zeros(batch_size, 1, spec.shape[-1]).to(spec.device))

    freqs = freqs * 2 * np.pi
    # subtract the expected value
    phase = phase - freqs[None, None, :]

    return torch.cat([mag[..., None], phase[..., None]], dim=-1)


def mag_phase_recomposition(spec, freqs):
    real = spec[..., 0]
    phase = spec[..., 1]

    freqs = freqs * 2 * np.pi
    # add expected value
    phase = phase + freqs[None, None, :]

    imag = torch.cumsum(phase, dim=1)
    
    imag = (imag + np.pi) % (2 * np.pi) - np.pi
    spec = real * torch.exp(1j * imag)
    return spec


def morlet_filter_bank(
        samplerate,
        kernel_size,
        scale,
        scaling_factor,
        normalize=True):
    
    basis_size = len(scale)
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    try:
        if len(scaling_factor) != len(scale):
            raise ValueError('scaling factor must have same length as scale')
    except TypeError:
        scaling_factor = np.repeat(float(scaling_factor), len(scale))

    sr = int(samplerate)

    for i, band in enumerate(scale):
        scaling = scaling_factor[i]
        w = band.center_frequency / (scaling * 2 * sr / kernel_size)
        basis[i] = morlet(
            M=kernel_size,
            w=w,
            s=scaling)

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8

    return basis.astype(np.complex64)


def short_time_transform(x, ws=512, ss=256, basis_func=None):
    basis = basis_func(ws)
    windowed = zounds.nputil.sliding_window(x, ws, ss)
    windowed = windowed * hann(ws)[None, :]
    freq_domain = np.dot(windowed, basis.T)
    return windowed, freq_domain


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
            (spec.shape)
            specs.append(torch.from_numpy(spec).to(audio_batch.device))
        return torch.cat(specs, dim=0)

    def to_time_domain(self, spec):
        samples = []
        device = spec.device
        spec = spec.data.cpu().numpy()
        for item in spec:
            samp = librosa.icqt(
                item.T,
                sparsity=0,
                hop_length=self.hop_length,
                bins_per_octave=self.bins_per_octave,
                scale=True)[None, ...]
            samp = torch.from_numpy(samp).to(device)
            samples.append(samp)
        return torch.cat(samples, dim=0)

    @property
    def center_frequencies(self):
        freqs = librosa.cqt_frequencies(
            self.n_bins,
            fmin=librosa.note_to_hz('C1'),
            bins_per_octave=self.bins_per_octave)
        freqs /= int(self.samplerate)
        return freqs

class MelScale(object):
    def __init__(self):
        super().__init__()
        self.samplerate = zounds.SR22050()
        self.fft_size = 512
        self.freq_band = zounds.FrequencyBand(20, self.samplerate.nyquist)
        self.scale = zounds.MelScale(self.freq_band, self.fft_size // 2)
        self.basis = morlet_filter_bank(
            self.samplerate, self.fft_size, self.scale, 0.1)
    
    def to_time_domain(self, spec):
        spec = spec.data.cpu().numpy()
        windowed = (spec @ self.basis).real[..., ::-1]
        windowed = torch.from_numpy(windowed.copy())
        td = overlap_add(windowed[:, None, :, :], apply_window=False)
        return td

    def to_frequency_domain(self, audio_batch):
        windowed = windowed_audio(
            audio_batch, self.fft_size, self.fft_size // 2)
        windowed = windowed.data.cpu().numpy()
        freq_domain = windowed @ self.basis.T
        freq_domain = torch.from_numpy(freq_domain)
        return freq_domain

    @property
    def center_frequencies(self):
        return np.array(list(self.scale.center_frequencies)) / int(self.samplerate)


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
    

    def listen(self, spec):
        with torch.no_grad():
            audio = self.to_time_domain(spec)
            return zounds.AudioSamples(
                audio[0].data.cpu().numpy().reshape(-1), zounds.SR22050()).pad_with_silence()


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    n_samples = 2 ** 15
    samplerate = zounds.SR22050()
    window_size = 512
    step_size = window_size // 2

    stream = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)

    basis = MelScale()
    transformer = AudioCodec(basis)

    while True:
        batch = next(stream)
        o = zounds.AudioSamples(
            batch.squeeze(), samplerate).pad_with_silence()
        batch = torch.from_numpy(batch).float()
        spec = transformer.to_frequency_domain(batch)

        mag = spec[..., 0].data.cpu().numpy().squeeze()
        phase = spec[..., 1].data.cpu().numpy().squeeze()

        recon = transformer.to_time_domain(spec)
        r = zounds.AudioSamples(recon.data.cpu()
                                .numpy().squeeze(), samplerate).pad_with_silence()
        input('Next...')


    # Synthetic Test
    # frequencies = basis.center_frequencies
    # hz = frequencies * samplerate.nyquist
    # indices = [10, 100, 150, 200]

    # synth = zounds.SineSynthesizer(samplerate)
    # audio = synth.synthesize(zounds.Seconds(5), hz[indices])
    # ta = torch.from_numpy(audio).reshape(1, -1)
    # print(ta.shape)

    # spec = transformer.to_frequency_domain(ta)

    # mag = spec[..., 0].data.cpu().numpy()
    # phase = spec[..., 1].data.cpu().numpy()

    # phases = phase[:, :, indices].squeeze()

    # mean_phases = phases[1:-1].mean(axis=0)


    
    input()
