import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from scipy.signal import hann

from .normal_pdf import pdf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def noise_spec(n_audio_samples, ws=512, step=256):
    """
    Create a spectrogram of white noise with shape
    (n_noise_frames, n_coeffs)
    """

    # create time-domain noise
    x = torch.FloatTensor(n_audio_samples).uniform_(-1, 1).to(device)
    x = F.pad(x, (0, ws))
    x = x.unfold(-1, ws, step)

    # take the STFT of the noise
    window = torch.hamming_window(ws).to(device)
    x = x * window[None, :]
    x = torch.fft.rfft(x, norm='ortho')
    return x


def band_filtered_noise(n_audio_samples, ws=512, step=256, mean=0.5, std=0.1):
    spec = noise_spec(n_audio_samples, ws, step)
    n_coeffs = spec.shape[-1]

    mean = mean * n_coeffs
    std = std * n_coeffs

    filt = pdf(torch.arange(0, n_coeffs, 1), mean, std)

    # normalize frequency-domain filter to have peak at 1
    filt = filt / filt.max()
    spec = spec * filt[None, :]

    windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = overlap_add(windowed[None, None, :, :])
    return samples[..., :n_audio_samples]


def noise_bank2(x):
    batch, magnitudes, samples = x.shape
    window_size = (magnitudes - 1) * 2
    hop_size = window_size // 2
    total_samples = hop_size * samples

    # create the noise
    noise = torch.FloatTensor(
        batch, total_samples).uniform_(-1, 1).to(x.device)
    # window the noise
    noise = F.pad(noise, (0, hop_size))
    noise = noise.unfold(-1, window_size, hop_size)

    noise_coeffs = torch.fft.rfft(noise, norm='ortho')

    x = x.permute(0, 2, 1)

    noise_coeffs = noise_coeffs * x

    filtered = noise_coeffs

    # recover the filtered noise in the time domain
    audio = torch.fft.irfft(
        filtered, n=window_size, norm='ortho')

    audio = overlap_add(audio[:, None, :, :], apply_window=True)
    audio = audio[..., :total_samples]
    audio = audio.view(batch, 1, -1)
    return audio


def _torch_overlap_add(x, apply_window=True, flip=False):
    batch, channels, frames, samples = x.shape

    if apply_window:
        window = torch.from_numpy(hann(samples, False)).to(x.device).float()
        # window = torch.hamming_window(samples, periodic=False).to(x.device)
        # window = torch.hann_window(samples, periodic=False).to(x.device)
        x = x * window[None, None, None, :]

    hop_size = samples // 2
    first_half = x[:, :, :, :hop_size].contiguous().view(batch, channels, -1)
    second_half = x[:, :, :, hop_size:].contiguous().view(batch, channels, -1)
    first_half = F.pad(first_half, (0, hop_size))
    second_half = F.pad(second_half, (hop_size, 0))

    if flip:
        first_half = first_half[:, :, ::-1]
    
    output = first_half + second_half
    return output


def _np_overlap_add(x, apply_window=True, flip=False):
    batch, channels, frames, samples = x.shape

    if apply_window:
        window = hann(samples)
        x = x * window[None, None, None, :]

    hop_size = samples // 2
    first_half = x[:, :, :, :hop_size].reshape((batch, channels, -1))
    second_half = x[:, :, :, hop_size:].reshape((batch, channels, -1))


    first_half = np.pad(first_half, [(0, 0), (0, 0), (0, hop_size)])
    second_half = np.pad(second_half, [(0, 0), (0, 0), (hop_size, 0)])

    if flip:
        first_half = first_half[:, :, ::-1]

    output = first_half + second_half
    return output


def overlap_add(x, apply_window=True, flip=False):
    # batch, channels, frames, samples = x.shape

    # if apply_window:
    #     window = torch.from_numpy(hann(samples, False)).to(x.device).float()
    #     # window = torch.hamming_window(samples, periodic=False).to(x.device)
    #     # window = torch.hann_window(samples, periodic=False).to(x.device)
    #     x = x * window[None, None, None, :]

    # hop_size = samples // 2
    # first_half = x[:, :, :, :hop_size].contiguous().view(batch, channels, -1)
    # second_half = x[:, :, :, hop_size:].contiguous().view(batch, channels, -1)
    # first_half = F.pad(first_half, (0, hop_size))
    # second_half = F.pad(second_half, (hop_size, 0))
    # output = first_half + second_half
    # return output

    if isinstance(x, np.ndarray):
        return _np_overlap_add(x, apply_window, flip)
    else:
        return _torch_overlap_add(x, apply_window, flip)


class DDSP(nn.Module):
    def __init__(
            self,
            channels,
            band_size,
            constrain,
            noise_frames=None,
            separate_components=False,
            linear_constrain=False,
            amp_nl=lambda x: torch.clamp(x, 0, 1),
            freq_nl=lambda x: torch.clamp(x, 0, 1),
            noise_nl=lambda x: torch.clamp(x, 0, 1)):

        super().__init__()
        self.channels = channels
        self.band_size = band_size
        self.separate_components = separate_components
        self.amp_nl = amp_nl
        self.freq_nl = freq_nl
        self.noise_nl = noise_nl

        if noise_frames is not None:
            self.noise_frames = noise_frames
            self.noise_samples = self.band_size // self.noise_frames
            # note: window size is twice noise_samples
            self.noise_coeffs = self.noise_samples + 1
        else:
            self.noise_samples = 64
            self.noise_frames = band_size // self.noise_samples
            self.noise_coeffs = (self.noise_frames // 2) + 1

        self.constrain = constrain
        self.amp = nn.Conv1d(64, 64, 1, 1, 0)
        self.freq = nn.Conv1d(64, 64, 1, 1, 0)

        if linear_constrain:
            bands = np.linspace(0.01, 1, 64) * np.pi
        else:
            bands = np.geomspace(0.01, 1, 64) * np.pi
        bp = np.concatenate([[0], bands])
        spans = np.diff(bp)

        self.bands = torch.from_numpy(bands).float().to(device)
        self.spans = torch.from_numpy(spans).float().to(device)

        self.noise = nn.Conv1d(64, self.noise_coeffs, 1, 1, 0)
        self.noise_factor = nn.Parameter(torch.FloatTensor(1).fill_(1))

    def forward(self, x):
        batch, channels, time = x.shape
        batch, channels, time = x.shape

        noise = self.noise_nl(self.noise(x))

        if time == self.band_size:
            noise = F.avg_pool1d(noise, self.noise_samples, self.noise_samples)

        noise = noise_bank2(noise) * self.noise_factor

        amp = self.amp_nl(self.amp(x))
        freq = self.freq_nl(self.freq(x))

        if time == self.band_size:
            amp = F.avg_pool1d(amp, 64, 1, 32)[..., :-1]
            freq = F.avg_pool1d(freq, 64, 1, 32)[..., :-1]
        else:
            amp = F.upsample(amp, size=self.band_size, mode='linear')
            freq = F.upsample(freq, size=self.band_size, mode='linear')

        if self.constrain:
            freq = self.bands[None, :, None] + \
                (freq * self.spans[None, :, None])

        freq = torch.sin(torch.cumsum(freq, dim=-1)) * amp
        freq = torch.mean(freq, dim=1, keepdim=True)

        if self.separate_components:
            return freq, noise
        else:
            return freq + noise


class OscillatorBank(nn.Module):
    def __init__(
            self,
            input_channels,
            n_osc,
            n_audio_samples,
            constrain=False,
            log_frequency=False,
            log_amplitude=False,
            activation=torch.sigmoid,
            amp_activation=None,
            return_params=False):

        super().__init__()
        self.n_osc = n_osc
        self.n_audio_samples = n_audio_samples
        self.input_channels = input_channels
        self.constrain = constrain
        self.log_amplitude = log_amplitude
        self.activation = activation
        self.return_params = return_params
        self.amp_activation = amp_activation or self.activation

        if log_frequency:
            bands = np.linspace(0.01, 1, n_osc)
        else:
            bands = np.geomspace(0.01, 1, n_osc)

        bp = np.concatenate([[0], bands])
        spans = np.diff(bp)

        self.bands = torch.from_numpy(bands).float().to(device)
        self.spans = torch.from_numpy(spans).float().to(device)

        self.amp = nn.Conv1d(input_channels, self.n_osc, 1, 1, 0)
        self.freq = nn.Conv1d(input_channels, self.n_osc, 1, 1, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_channels, -1)

        amp = self.amp(x)
        freq = self.freq(x)

        amp = self.amp_activation(amp)
        if self.log_amplitude:
            amp = amp ** 2

        freq = self.activation(freq)

        if self.constrain:
            freq = self.bands[None, :, None] + \
                (freq * self.spans[None, :, None])

        freq_params = freq

        amp = F.upsample(amp, size=self.n_audio_samples, mode='linear')
        freq = F.upsample(freq, size=self.n_audio_samples, mode='linear')

        x = torch.sin(torch.cumsum(freq * torch.pi, dim=-1)) * amp
        x = torch.mean(x, dim=1, keepdim=True)
        if self.return_params:
            return x, freq_params
        else:
            return x


class NoiseModel(nn.Module):
    def __init__(
            self,
            input_channels,
            input_size,
            n_noise_frames,
            n_audio_samples,
            channels,
            activation=lambda x: torch.clamp(x, -1, 1),
            squared=False):

        super().__init__()
        self.input_channels = input_channels
        self.n_noise_frames = n_noise_frames
        self.n_audio_samples = n_audio_samples
        self.input_size = input_size
        self.channels = channels
        self.activation = activation
        self.squared = squared

        noise_step = n_audio_samples // n_noise_frames
        noise_window = noise_step * 2
        self.noise_coeffs = (noise_window // 2) + 1

        layers = int(np.log2(n_noise_frames) - np.log2(input_size))

        self.initial = nn.Conv1d(input_channels, channels, 1, 1, 0)

        self.upscale = nn.Sequential(*[
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2))
            for _ in range(layers)])

        self.final = nn.Conv1d(channels, self.noise_coeffs, 1, 1, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_channels, self.input_size)
        x = self.initial(x)
        x = self.upscale(x)
        x = self.final(x)
        x = self.activation(x)
        if self.squared:
            x = x ** 2
        x = noise_bank2(x)
        return x
