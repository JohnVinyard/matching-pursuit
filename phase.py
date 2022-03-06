import numpy as np
import torch
from torch.nn import functional as F
import zounds
from config.dotenv import Config
from data.datastore import batch_stream

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


def to_spectrogram(
        audio_batch,
        window_size,
        step_size,
        samplerate,
        get_center_freqs):
    spec = stft(audio_batch, window_size, step_size, samplerate)
    decomp = mag_phase_decomposition(spec, get_center_freqs(window_size))
    return decomp


def from_spectrogram(
        spec,
        window_size,
        step_size,
        samplerate,
        get_center_freqs):

    spec = mag_phase_recomposition(spec, get_center_freqs(window_size))
    return istft(spec)


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    n_samples = 2 ** 15
    samplerate = zounds.SR22050()
    window_size = 512
    step_size = window_size // 2

    stream = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)

    while True:
        batch = next(stream)
        o = zounds.AudioSamples(batch.squeeze(), samplerate).pad_with_silence()
        batch = torch.from_numpy(batch).float()

        spec = to_spectrogram(batch, window_size, step_size,
                              int(samplerate), rfft_freqs)
        mag = spec[..., 0].data.cpu().numpy().squeeze()
        phase = spec[..., 1].data.cpu().numpy().squeeze()

        recon = from_spectrogram(
            spec, window_size, step_size, int(samplerate), rfft_freqs)

        r = zounds.AudioSamples(recon.data.cpu().numpy(
        ).squeeze(), samplerate).pad_with_silence()
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
