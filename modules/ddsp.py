import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
from scipy.signal import hann
from config.dotenv import Config
from modules.linear import LinearOutputStack
from modules.physical import Window
from modules.reverb import NeuralReverb
from modules.overlap_add import overlap_add

from modules.upsample import ConvUpsample, FFTUpsampleBlock

from .normal_pdf import pdf
# import zounds

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def noise_spec(n_audio_samples, ws=512, step=256, device=None):
    """
    Create a spectrogram of white noise with shape
    (n_noise_frames, n_coeffs)
    """

    # create time-domain noise
    x = torch.FloatTensor(n_audio_samples).uniform_(-1, 1)
    if device is not None:
        x = x.to(device)
    x = F.pad(x, (0, step))
    x = x.unfold(-1, ws, step)

    # take the STFT of the noise
    window = torch.hamming_window(ws).to(device)
    x = x * window[None, :]
    x = torch.fft.rfft(x, norm='ortho')

    # output shape
    # (n_audio_samples / step, ws // 2 + 1)

    return x


def band_filtered_noise(n_audio_samples, ws=512, step=256, mean=0.5, std=0.1):

    batch, atoms, seq_len = mean.shape
    frames = n_audio_samples // step

    spec = noise_spec(n_audio_samples, ws, step, device=mean.device)
    n_coeffs = spec.shape[-1]

    mean = mean * n_coeffs
    std = std * n_coeffs


    filt = pdf(
        torch.arange(0, n_coeffs, 1).view(1, 1, n_coeffs, 1).to(mean.device),
        mean[:, :, None, :],
        std[:, :, None, :])

    # normalize frequency-domain filter to have peak at 1
    filt = filt / filt.max()
    spec = spec.T[None, None, ...] * filt
    spec = spec.view(batch, atoms, n_coeffs, frames).permute(0, 1, 3, 2)

    windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = overlap_add(windowed)
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
    window = torch.hann_window(window_size).to(x.device)
    noise = noise * window[None, None, :]

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
            activation=torch.sigmoid,
            amp_activation=None,
            return_params=False,
            lowest_freq=0.01,
            complex_valued=False,
            use_wavetable=False,
            wavetable_size=1024):

        super().__init__()
        self.n_osc = n_osc
        self.n_audio_samples = n_audio_samples
        self.input_channels = input_channels
        self.constrain = constrain
        self.activation = activation
        self.return_params = return_params
        self.amp_activation = amp_activation or self.activation
        self.lowest_freq = lowest_freq
        self.complex_valued = complex_valued

        if log_frequency:
            bands = np.geomspace(lowest_freq, 1, n_osc)
        else:
            bands = np.linspace(lowest_freq, 1, n_osc)

        bp = np.concatenate([[0], bands])
        spans = np.diff(bp)

        self.bands = torch.from_numpy(bands).float().to(device)
        self.spans = torch.from_numpy(spans).float().to(device)

        self.amp = nn.Conv1d(input_channels, self.n_osc, 1, 1, 0)
        self.freq = nn.Conv1d(input_channels, self.n_osc, 1, 1, 0)

        self.use_wavetable = use_wavetable
        self.wavetable_size = wavetable_size

        if use_wavetable:
            table = torch.sin(torch.linspace(-np.pi, np.pi, wavetable_size))[None, :]
            self.register_buffer('table', table)
            self.win = Window(wavetable_size, 0, 1, range_shape=(1, 1, 1, wavetable_size))


    def forward(self, x, add_noise=False):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_channels, -1)

        amp = self.amp(x)
        freq = self.freq(x)

        if self.complex_valued:
            z = torch.cat([amp[:, :, None, :], freq[:, :, None, :]], dim=2)
            a = torch.norm(z, dim=2)
            f = torch.angle(torch.complex(amp, freq)) / np.pi

            amp = a
            freq = f
        else:
            amp = self.amp_activation(amp)
            freq = self.activation(freq)

        if self.constrain:
            freq = self.bands[None, :, None] + \
                (freq * self.spans[None, :, None])

        amp_params = amp
        freq_params = freq


        if add_noise:
            amp = amp + torch.zeros_like(amp).normal_(0, 0.1)
            freq = freq + torch.zeros_like(freq).normal_(0, 0.1)

        amp = F.interpolate(amp, size=self.n_audio_samples, mode='linear')
        freq = F.interpolate(freq, size=self.n_audio_samples, mode='linear')
        cum_freq = torch.cumsum(freq * np.pi, dim=-1)

        if not self.use_wavetable:        
            x = torch.sin(cum_freq) * amp
        else:
            pos = cum_freq % 1
            std = torch.zeros_like(pos).fill_(0.01)
            
            sampling_kernels = self.win.forward(pos[..., None], std[..., None])

            print('KERNELS', sampling_kernels.shape)
            sig = sampling_kernels @ self.table.T
            print('SIG', sig.shape)
            sig = sig.permute(0, 2, 1)
            print(sig.shape)

            sig = sig.view(batch_size, -1, self.n_audio_samples)

        x = torch.mean(x, dim=1, keepdim=True)

        if self.return_params:
            return x, freq_params, amp_params
        else:
            return x


class UnconstrainedOscillatorBank(nn.Module):
    def __init__(
            self,
            input_channels,
            n_osc,
            n_audio_samples,
            min_frequency_hz=40,
            max_frequency_hz=9000,
            samplerate=22050,
            fft_upsample=False,
            baselines=True):

        super().__init__()
        self.to_osc = nn.Conv1d(input_channels, n_osc * 2, 1, 1, 0)
        self.n_audio_samples = n_audio_samples

        self.baselines = baselines

        if self.baselines:
            self._baselines = nn.Parameter(
                torch.zeros(n_osc, 2).uniform_(0, 0.1))

        self.fft_upsample = fft_upsample

        if self.fft_upsample:
            self.upsample = FFTUpsampleBlock(n_osc, n_osc, 128, factor=256)

        self.min_frequency_hz = min_frequency_hz
        self.max_frequency_hz = max_frequency_hz
        self.samplerate = samplerate
        self.n_osc = n_osc

    def forward(self, x):
        batch, channels, time = x.shape

        x = self.to_osc(x)
        x = x.view(batch, self.n_osc, 2, time)

        if self.baselines:
            x = (x * 0.01) + self._baselines[None, :, :, None]

        amp = torch.norm(x, dim=2)
        r = x[:, :, 0, :]
        i = x[:, :, 1, :]
        freq = torch.angle(torch.complex(r, i)) / np.pi

        freq = freq ** 2
        amp = amp ** 2

        base_freq = self.min_frequency_hz / self.samplerate.nyquist
        max_freq = self.max_frequency_hz / self.samplerate.nyquist
        freq_range = max_freq - base_freq
        amp = base_freq + (amp * freq_range)

        if self.fft_upsample:
            amp = self.upsample(amp)
            freq = self.upsample(freq)
        else:
            amp = F.interpolate(amp, size=self.n_audio_samples, mode='linear')
            freq = F.interpolate(
                freq, size=self.n_audio_samples, mode='linear')

        # phase accumulator
        osc = torch.sin(torch.cumsum(freq * np.pi, dim=-1)) * amp
        # sum over all oscillators
        osc = torch.sum(osc, dim=1, keepdim=True)
        return osc


class NoiseModel(nn.Module):
    def __init__(
            self,
            input_channels,
            input_size,
            n_noise_frames,
            n_audio_samples,
            channels,
            activation=lambda x: torch.clamp(x, -1, 1),
            squared=False,
            mask_after=None,
            return_params=False,
            batch_norm=False,
            layer_norm=False,
            weight_norm=False):

        super().__init__()
        self.return_params = return_params
        self.input_channels = input_channels
        self.n_noise_frames = n_noise_frames
        self.n_audio_samples = n_audio_samples
        self.input_size = input_size
        self.channels = channels
        self.activation = activation
        self.squared = squared
        self.mask_after = mask_after
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        noise_step = n_audio_samples // n_noise_frames
        noise_window = noise_step * 2
        self.noise_coeffs = (noise_window // 2) + 1

        self.upscale = ConvUpsample(
            input_channels, 
            channels, 
            start_size=input_size, 
            end_size=n_noise_frames, 
            mode='learned', 
            out_channels=self.noise_coeffs, 
            from_latent=False, 
            batch_norm=batch_norm,
            layer_norm=layer_norm,
            weight_norm=weight_norm)


    def forward(self, x, add_noise=False):
        batch_size = x.shape[0]
        x = x.view(batch_size, self.input_channels, self.input_size)
        x = self.upscale(x)
        x = self.activation(x)
        if self.squared:
            x = x ** 2

        if self.mask_after is not None:
            x[:, :self.mask_after, :] = 1

        noise_params = x
        if add_noise:
            x = x + torch.zeros_like(x).normal_(0, 0.1)
        x = noise_bank2(x)
        if self.return_params:
            return x, noise_params
        else:
            return x


class HarmonicModel(nn.Module):
    def __init__(
            self,
            n_voices=8,
            n_profiles=16,
            n_harmonics=64,
            freq_hz_range=(40, 4000),
            samplerate=22050,
            reduce=torch.sum,
            n_frames=64,
            n_samples=2**14):

        super().__init__()
        self.n_voices = n_voices
        self.freq_hz_range = freq_hz_range
        self.reduce = reduce
        self.n_profiles = n_profiles
        self.samplerate = samplerate
        self.n_harmonics = n_harmonics
        self.n_frames = n_frames
        self.n_samples = n_samples

        self.min_freq = self.freq_hz_range[0] / self.samplerate.nyquist
        self.max_freq = self.freq_hz_range[1] / self.samplerate.nyquist
        self.freq_interval = self.max_freq = self.min_freq

        # harmonic profiles
        self.profiles = nn.Parameter(torch.zeros(
            n_profiles, n_harmonics).uniform_(0, 0.1))

        self.baselines = nn.Parameter(
            torch.zeros(self.n_voices, 2).uniform_(0, 0.05))

        # harmonic ratios to the fundamental
        self.register_buffer('ratios', torch.arange(
            2, 2 + self.n_harmonics) ** 2)

    def forward(self, f0, harmonics):
        batch = f0.shape[0]

        f0 = f0.view(f0.shape[0], self.n_voices, 2, -1)
        # f0 = self.baselines[None, :, :, None] + (0.1 * f0)
        harmonics = harmonics.view(
            harmonics.shape[0], self.n_voices, self.n_profiles, -1)

        f0_amp = torch.norm(f0, dim=-2) ** 2
        f0 = (torch.angle(torch.complex(
            f0[:, :, 0, :], f0[:, :, 1, :])) / np.pi)

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
            batch * self.n_voices, self.n_harmonics + 1, self.n_frames)
        full_amp = full_amp.view(
            batch * self.n_voices, self.n_harmonics + 1, self.n_frames)

        full_freq = F.interpolate(
            full_freq, size=self.n_samples, mode='linear')
        full_amp = F.interpolate(full_amp, size=self.n_samples, mode='linear')

        signal = full_amp * torch.sin(torch.cumsum(full_freq, dim=-1) * np.pi)

        signal = signal.view(batch, self.n_voices,
                             self.n_harmonics + 1, self.n_samples)

        signal = torch.sum(signal, dim=(1, 2)).view(batch, 1, self.n_samples)

        return signal


class AudioModel(nn.Module):
    def __init__(
            self, 
            n_samples, 
            model_dim, 
            samplerate, 
            n_frames, 
            n_noise_frames, 
            batch_norm=False, 
            use_wavetable=False,
            complex_valued_osc=False):
    
        super().__init__()
        self.n_samples = n_samples
        self.model_dim = model_dim
        self.n_frames = n_frames

        self.osc = OscillatorBank(
            model_dim, 
            model_dim, 
            n_samples, 
            constrain=not use_wavetable, 
            lowest_freq=40 / (samplerate // 2),
            amp_activation=lambda x: x ** 2,
            complex_valued=complex_valued_osc,
            use_wavetable=use_wavetable)
        
        self.noise = NoiseModel(
            model_dim,
            n_frames,
            n_noise_frames,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1,
            batch_norm=batch_norm)
        
        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate, n_samples)

        self.to_rooms = LinearOutputStack(model_dim, 1, out_channels=self.verb.n_rooms, norm=lambda channels: nn.LayerNorm((channels,)))
        self.to_mix = LinearOutputStack(model_dim, 1, out_channels=1, norm=lambda channels: nn.LayerNorm((channels,)))

    
    def forward(self, x):
        x = x.view(-1, self.model_dim, self.n_frames)

        agg = x.mean(dim=-1)
        room = torch.softmax(self.to_rooms(agg), dim=-1)
        # TODO: Support gumbel softmax option
        mix = torch.sigmoid(self.to_mix(agg)).view(-1, 1, 1)

        harm = self.osc.forward(x)
        noise = self.noise.forward(x)

        
        dry = harm + noise
        wet = self.verb(dry, room)
        signal = (dry * mix) + (wet * (1 - mix))
        return signal