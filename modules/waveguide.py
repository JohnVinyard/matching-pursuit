import numpy as np
from torch import nn
import torch
from torch.nn import functional as F
from modules.ddsp import overlap_add
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, unit_norm

from modules.upsample import ConvUpsample


def karplus_strong_synth(
        impulse_env: torch.Tensor,
        delay_vector: torch.Tensor,
        filter_vector: torch.Tensor,
        decay: torch.Tensor,
        n_samples: int):

    batch, _, n_frames = impulse_env.shape
    batch, max_delay = delay_vector.shape
    
    # ensure decay is in the range [0 - 1]
    decay = torch.sigmoid(decay)
    delay_vector = torch.sigmoid(delay_vector, dim=-1)
    filter_vector = unit_norm(filter_vector, dim=-1)

    noise = torch.zeros((batch, 1, n_samples), device=impulse_env.device).uniform_(-1, 1)
    imp = F.upsample(impulse_env, size=n_samples, mode='linear')
    signal = noise * imp

    for i in range(max_delay, n_samples):
        start = i - max_delay
        stop = start + max_delay
        segment = signal[:, :, start: stop]
        delay = delay_vector[:, None, :] @ segment
        filtered = F.conv1d(
            delay, filter_vector.view(1, 1, max_delay), padding=max_delay // 2)[..., :max_delay]
        decayed = filtered * decay.view(-1, 1, 1)
        signal[:, :, i] = signal[:, :, i] + decayed.sum(dim=-1, keepdim=True)
    
    return signal


class WaveguideSynth(nn.Module):
    def __init__(self, max_delay=512, n_samples=2**15, filter_kernel_size=512):
        super().__init__()

        self.n_delays = max_delay
        self.n_samples = n_samples
        self.filter_kernel_size = filter_kernel_size

        delays = torch.zeros(max_delay, n_samples)
        for i in range(max_delay):
            delays[i, ::(i + 1)] = 1

        self.register_buffer('delays', delays)

    def forward(self, impulse, delay_selection, damping, filt):
        batch = delay_selection.shape[0]
        n_frames = filt.shape[-1]

        # filter (TODO: Should I allow a time-dependent transfer function?)

        f = torch.sigmoid(filt).view(-1, 1, 16)
        f = F.interpolate(f, size=self.n_samples // 2, mode='linear')
        f = F.pad(f, (0, 1))
        filt_spec = f

        # filt = torch.fft.irfft(f, dim=-1, norm='ortho').view(batch, self.filter_kernel_size)
        # filt = filt * torch.hamming_window(self.filter_kernel_size, device=impulse.device)[None, :]
        # diff = self.n_samples - self.filter_kernel_size
        # filt = torch.cat([filt, torch.zeros(batch, diff)], dim=-1).view(batch, 1, self.n_samples)

        # Impulse / energy
        impulse = impulse.view(batch, 1, -1) ** 2
        impulse = F.interpolate(impulse, size=self.n_samples, mode='linear')
        noise = torch.zeros(batch, 1, self.n_samples,
                            device=impulse.device).uniform_(-1, 1)
        impulse = impulse * noise

        # Damping for delay (single value per batch member/item)
        damping = torch.sigmoid(damping.view(batch, 1)) * 0.9999
        powers = torch.linspace(
            1, damping.shape[-1], steps=n_frames, device=impulse.device)
        damping = damping[:, :, None] ** powers[None, None, :]
        damping = F.interpolate(damping, size=self.n_samples, mode='nearest')

        # delay count (TODO: should this be sparsified?)
        delay_selection = delay_selection.view(batch, self.n_delays, -1)
        delay_selection = torch.softmax(delay_selection, dim=1)
        delay_selection = F.interpolate(
            delay_selection, size=self.n_samples, mode='nearest')

        d = (delay_selection * self.delays).sum(dim=1, keepdim=True) * damping

        delay_spec = torch.fft.rfft(d, dim=-1, norm='ortho')
        impulse_spec = torch.fft.rfft(impulse, dim=-1, norm='ortho')
        # filt_spec = torch.fft.rfft(filt, dim=-1, norm='ortho')

        spec = delay_spec * impulse_spec * filt_spec

        final = torch.fft.irfft(spec, dim=-1, norm='ortho')
        return final


class TransferFunctionSegmentGenerator(nn.Module):
    def __init__(self, model_dim, n_frames, window_size, n_samples, cumulative=False):
        super().__init__()
        self.model_dim = model_dim
        self.n_frames = n_frames
        self.window_size = window_size
        self.n_samples = n_samples
        self.cumulative = cumulative

        self.env = ConvUpsample(
            model_dim, model_dim, 4, n_frames, mode='nearest', out_channels=1, norm=ExampleNorm())

        n_coeffs = window_size // 2 + 1
        self.n_coeffs = n_coeffs

        if self.cumulative:
            self.transfer = LinearOutputStack(
                model_dim, 3, out_channels=n_coeffs * 2)
        else:
            self.transfer = ConvUpsample(
                model_dim, model_dim, 4, n_frames, mode='nearest', out_channels=n_coeffs * 2, batch_norm=True
            )

    def forward(self, x):
        x = x.view(-1, self.model_dim)

        # TODO: envelope generator
        env = self.env(x) ** 2
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples,
                            device=env.device).uniform_(-1, 1)
        env = env * noise

        tf = self.transfer(x)
        if self.cumulative:
            tf = tf.view(-1, self.n_coeffs * 2, 1).repeat(1, 1, self.n_frames)

        tf = tf.view(-1, self.n_coeffs, 2, self.n_frames)
        norm = torch.norm(tf, dim=2, keepdim=True)

        unit_norm = tf / (norm + 1e-8)
        clamped_norm = torch.clamp(norm, 0, 0.9999)
        tf = unit_norm * clamped_norm

        tf = tf.view(-1, self.n_coeffs * 2, self.n_frames)

        real = tf[:, :self.n_coeffs, :]
        imag = tf[:, self.n_coeffs:, :]

        # real = real * torch.cos(imag)
        # imag = real * torch.sin(imag)

        tf = torch.complex(real, imag)

        if self.cumulative:
            # tf = torch.exp(torch.cumsum(torch.log(tf + 1e-4), dim=-1))
            # pow = torch.linspace(1, self.n_frames + 1, self.n_frames, device=tf.device)[None, None, :]
            # tf = tf ** pow
            tf = torch.cumprod(tf, dim=-1)

        tf = torch.fft.irfft(tf, dim=1, norm='ortho')
        tf = tf.permute(0, 2, 1).view(-1, 1, self.n_frames, self.window_size) * \
            torch.hamming_window(self.window_size, device=tf.device)[
            None, None, None, :]

        # TODO: Option to cut off in overlap add
        tf = overlap_add(tf)[..., :self.n_samples]

        env_spec = torch.fft.rfft(env, dim=-1, norm='ortho')
        tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
        spec = env_spec * tf_spec
        final = torch.fft.irfft(spec, dim=-1, norm='ortho')

        return final


def waveguide_synth(
        impulse: np.ndarray,
        delay: np.ndarray,
        damping: np.ndarray,
        filter_size: int) -> np.ndarray:

    n_samples = impulse.shape[0]

    output = impulse.copy()
    buf = np.zeros_like(impulse)

    for i in range(n_samples):
        delay_val = 0

        delay_amt = delay[i]

        if i > delay_amt:
            damping_amt = damping[i]
            delay_val += output[i - delay_amt] * damping_amt

        buf[i] = delay_val

        filt_size = filter_size[i]
        filt_slice = buf[i - filt_size: i]
        if filt_slice.shape[0]:
            new_val = np.mean(filt_slice)
        else:
            new_val = delay_val

        output[i] += new_val

    return output
