from functools import reduce
from typing import Collection, List, Union
import torch
from torch import nn

from modules import stft, unit_norm, max_norm

from modules.ddsp import overlap_add
from modules.normal_pdf import pdf2
from modules.pos_encode import pos_encoded
from modules.softmax import sparse_softmax
from modules.upsample import ConvUpsample
import numpy as np
from torch.nn import functional as F
from scipy.signal import square, sawtooth


class ExponentialTransform(nn.Module):
    def __init__(self, window_size: int, step: int, n_exponents: int, n_frames: int, max_exponent: float = 100):
        super().__init__()
        self.max_exponent = max_exponent
        self.window_size = window_size
        self.step = step
        self.n_exponents = n_exponents
        self.n_frames = n_frames

        bank = \
            torch.linspace(1, 0, n_frames)[None, :] \
            ** torch.linspace(2, self.max_exponent, self.n_exponents)[:, None]
        self.register_buffer('bank', unit_norm(bank))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, n_events, _ = audio.shape
        audio = audio.view(audio.shape[0], n_events, -1)
        spec = stft(audio, self.window_size, self.step, pad=True)
        batch, n_events, time, bands = spec.shape
        spec = spec.permute(0, 1, 3, 2)[:, :, :, None, :]
        b = self.bank[None, None, None, :, :]
        result = fft_convolve(spec, b)
        result = result.view(batch, n_events, bands, self.n_exponents, time)
        result = result.view(batch, n_events, bands * self.n_exponents, time)
        return result


def hierarchical_dirac(elements: torch.Tensor, soft: bool = False):
    """
    Produce a dirac/one-hot encoding from a binary encoded
    tensor of the shape (..., log2, 2)
    """

    # the shape except for the last two elements, which are (log2, 2)
    seq_shape = elements.shape[:-2]

    # the number of steps taken to expand, log2
    steps = elements.shape[-2]

    # starting size is 2**1 or 2
    current_size = 2

    if soft:
        chosen = torch.softmax(elements, dim=-1)
    else:
        chosen = sparse_softmax(elements, normalize=True, dim=-1)

    signal = torch.zeros(*seq_shape, 1, device=elements.device)

    for i in range(steps):

        if i == 0:
            signal = chosen[..., i, :]
        else:
            # new_size = signal.shape[-1] * 2

            new_size = current_size * 2

            # first, upsample with "holes" or by
            # filling in zeros, instead of some kind of
            # interpolation
            new_signal = torch.zeros(*seq_shape, new_size, device=elements.device)

            # print(new_signal.shape, signal.shape)

            new_signal[..., ::2] = signal

            diff = new_size - 2

            # pad the selection
            current = torch.cat([
                chosen[..., i, :],
                torch.zeros(*seq_shape, diff, device=elements.device)
            ], dim=-1)

            signal = fft_convolve(new_signal, current)

            current_size = new_size

    return signal


def gaussian_bandpass_filtered(
        means: torch.Tensor,
        stds: torch.Tensor,
        signals: torch.Tensor,
        normalize: bool = True):
    batch, _, samples = signals.shape
    n_coeffs = samples // 2 + 1
    gaussians = pdf2(means, stds, n_coeffs, normalize=normalize)

    spec = torch.fft.rfft(signals, dim=-1)
    spec = spec * gaussians
    filtered = torch.fft.irfft(spec)
    return filtered


def make_waves_vectorized(n_samples: int, f0s: np.ndarray, samplerate: int):
    n_frequencies = len(f0s)
    total_atoms = n_frequencies * 4

    f0s = f0s / (samplerate // 2)
    rps = f0s * np.pi
    radians = np.linspace(0, n_samples, n_samples)

    radians = rps[:, None] * radians[None, :]

    sawtooths = sawtooth(radians)
    squares = square(radians)
    triangles = sawtooth(radians, width=0.5)
    sines = np.sin(radians)

    waves = np.concatenate([sawtooths, squares, triangles, sines], axis=0)
    waves = torch.from_numpy(waves).view(total_atoms, n_samples).float()
    return waves


def make_waves(n_samples: int, f0s: List[float], samplerate: int):
    """
    Generate pure sines, sawtooth, triangle, and square waves
    with the provided fundamental frequencies
    """

    sawtooths = []
    squares = []
    triangles = []
    sines = []

    total_atoms = len(f0s) * 4

    for f0 in f0s:
        f0 = f0 / (samplerate // 2)
        rps = f0 * np.pi
        radians = np.linspace(0, rps * n_samples, n_samples)
        sq = square(radians)[None, ...]
        squares.append(sq)
        st = sawtooth(radians)[None, ...]
        sawtooths.append(st)
        tri = sawtooth(radians, 0.5)[None, ...]
        triangles.append(tri)
        sin = np.sin(radians)
        sines.append(sin[None, ...])

    waves = np.concatenate([
        sawtooths,
        squares,
        triangles,
        sines
    ], axis=0)
    waves = torch.from_numpy(waves).view(total_atoms, n_samples).float()
    return waves


def freq_domain_transfer_function_to_resonance(
        window_size: int,
        coeffs: torch.Tensor,
        n_frames: int,
        apply_decay: bool = True,
        start_phase: Union[None, torch.Tensor] = None,
        start_mags: Union[None, torch.Tensor] = None,
        log_space_scan: bool = True,
        phase_dither: torch.Tensor = None) -> torch.Tensor:
    step_size = window_size // 2
    total_samples = step_size * n_frames

    expected_coeffs = window_size // 2 + 1

    group_delay = torch.linspace(0, np.pi, expected_coeffs, device=coeffs.device)

    res = coeffs.reshape(-1, expected_coeffs, 1).repeat(1, 1, n_frames)

    if start_mags is not None:
        start_mags = start_mags.reshape(res.shape[0], expected_coeffs, 1)
    else:
        start_mags = torch.ones(res.shape[0], expected_coeffs, 1, device=res.device)

    # always start with full energy at every coefficient
    res = torch.cat([
        start_mags,
        res
    ], dim=-1)

    if apply_decay:
        if log_space_scan:
            res = torch.log(res + 1e-12)
            res = torch.cumsum(res, dim=-1)
            res = torch.exp(res)
        else:
            res = torch.cumprod(res, dim=-1)

    # remove the final frame
    spec = res[..., :n_frames]
    spec = spec.view(-1, expected_coeffs, n_frames).permute(0, 2, 1).view(-1, 1, n_frames, expected_coeffs)

    phase = torch.zeros_like(spec)
    # .uniform_(-np.pi, np.pi)
    gd = group_delay[None, None, None, :]
    phase[:, :, :, :] = gd

    if phase_dither is not None:
        # print(phase_dither.shape, phase.shape, group_delay.shape)
        # TODO: Experimental
        phase = phase + (torch.zeros_like(phase).uniform_(-1, 1) * group_delay[None, None, None, :] * phase_dither[:, None, :, :])

    phase = torch.cumsum(phase, dim=2)

    if start_phase is not None:
        # apply constant offset to each FFT bin
        phase = phase + start_phase.reshape(-1, 1, 1, expected_coeffs)

    # convert from polar coordinates
    spec = spec * torch.exp(1j * phase)

    windowed = torch.fft.irfft(spec, dim=-1).view(-1, 1, n_frames, window_size)
    audio = overlap_add(windowed, apply_window=False)[..., :total_samples]
    audio = audio.view(-1, 1, total_samples)
    audio = max_norm(audio)
    return audio


class ResonanceBank(nn.Module):
    def __init__(
            self, n_resonances,
            window_size,
            n_frames,
            initial,
            fft_based_resonance=False,
            learnable_resonances=True):

        super().__init__()
        self.n_coeffs = window_size // 2 + 1
        self.window_size = window_size
        self.n_resonances = n_resonances
        # self.n_samples = (window_size // 2) * n_frames
        self.n_samples = initial.shape[-1]
        self.fft_based_resonance = fft_based_resonance
        self.learnable_resonances = learnable_resonances
        self.n_frames = n_frames

        if self.learnable_resonances:
            self.res_samples = nn.Parameter(initial)
        else:
            self.register_buffer('res_samples', initial)

        self.base_resonance = 0.02
        self.res_factor = (1 - self.base_resonance) * 0.99

        self.decay = nn.Linear(n_resonances, self.n_frames)
        # self.amplitudes = nn.Parameter(torch.zeros(n_resonances, n_frames).uniform_(-6, 6))
        self.filters = nn.Parameter(torch.zeros(n_resonances, self.n_frames).uniform_(-1, 1))

        self.fft_res = nn.Parameter(torch.zeros(n_resonances, self.n_coeffs).uniform_(-6, -6))

    def forward(self, selection: torch.Tensor, initial_selection: torch.Tensor, filter_selection: torch.Tensor):

        batch_size = selection.shape[0]

        # choose a linear combination of filters
        filt = filter_selection @ self.filters
        filt = filt.view(-1, 1, self.n_frames)
        filt = filt * torch.hamming_window(self.n_frames, device=filt.device)[None, None, :]

        # choose a linear combination of envelopes

        decay = torch.sigmoid(self.decay(initial_selection))
        decay = self.base_resonance + (decay * self.res_factor)
        decay = torch.log(1e-12 + decay)
        decay = torch.cumsum(decay, dim=-1)

        decay = torch.exp(decay).view(batch_size, -1, self.n_frames)

        amp = F.interpolate(decay, size=self.n_samples, mode='linear')

        # amp = torch.sigmoid(torch.relu(initial_selection) @ self.amplitudes).view(-1, 1, 128)

        # amp = F.interpolate(amp, size=self.n_samples, mode='linear')

        if not self.fft_based_resonance:
            res = selection @ self.res_samples
        else:
            coeffs = torch.sigmoid(selection @ self.fft_res)
            res = freq_domain_transfer_function_to_resonance(self.window_size, coeffs, 128)

        amp = amp.view(*res.shape)

        # apply the envelopes
        res = res * amp

        # convolve with filters
        filt = F.pad(filt, (0, self.n_samples - self.n_frames))
        filt = filt.view(*res.shape)
        res = fft_convolve(filt, res)[..., :self.n_samples]
        return res


class TimeVaryingMix(nn.Module):
    def __init__(self, latent_dim, channels, n_mixer_channels, n_frames):
        super().__init__()
        self.latent_dim = latent_dim
        self.channels = channels
        self.n_mixer_channels = n_mixer_channels
        self.n_frames = n_frames

        self.to_time_varying_mix = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            mode='nearest',
            end_size=self.n_frames,
            out_channels=n_mixer_channels,
            from_latent=True,
            batch_norm=False,
            layer_norm=False,
            weight_norm=True)

    def forward(self, x, audio_channels):
        batch_size = x.shape[0]

        total_samples = audio_channels.shape[-1]
        mix = self.to_time_varying_mix(x).view(-1, self.n_mixer_channels, self.n_frames)

        # TODO: is softmax the best choice for the time-varying mix?
        # TODO: should softmax come before or after the upsampling
        mix = F.interpolate(mix, size=total_samples, mode='linear')
        mix = torch.softmax(mix, dim=1)

        # audio channels and mix will both be (batch * n_events, mix_channels, samples)
        x = audio_channels * mix
        # the final result will be (batch * n_events, samples)
        x = torch.sum(x, dim=1)
        # move back to (batch, n_events, samples)
        x = x.view(batch_size, -1, total_samples)
        return x


class ResonanceBlock(nn.Module):
    def __init__(self, n_atoms, window_size, n_frames, total_samples, mix_channels, channels, latent_dim, initial,
                 learnable_resonances):
        super().__init__()
        self.n_atoms = n_atoms
        self.window_size = window_size
        self.n_frames = n_frames
        self.total_samples = total_samples
        self.mix_channels = mix_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.initial = initial

        self.bank = ResonanceBank(n_atoms, window_size, n_frames, self.initial, fft_based_resonance=False,
                                  learnable_resonances=learnable_resonances)

        self.generate_mix = TimeVaryingMix(
            latent_dim, channels, mix_channels, n_frames)

        # produce a resonance for each element in the mix
        self.res_choices = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, n_atoms),
                nn.ReLU()
            )
            for _ in range(mix_channels)
        ])

        # produce a linear mixture of enevelopes
        self.init_choices = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, n_atoms),
                nn.ReLU()
            )
            for _ in range(mix_channels)
        ])

        # produce a linear mixture of filters
        self.filt_choice = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, n_atoms),
                nn.ReLU()
            )
            for _ in range(mix_channels)
        ])

        self.final_mix = nn.Linear(latent_dim, 2)

    def forward(self, x, impulse):
        # print('------------------------------')

        batch_size = x.shape[0]

        impulse_samples = impulse.shape[-1]

        final_mix = self.final_mix(x)
        # print(f'\t BATCH_SIZE {batch_size} FINAL_MIX {final_mix.shape}')

        final_mix = torch.softmax(final_mix, dim=-1)
        final_mix = final_mix.view(batch_size, -1, 1, 2)

        resonances = [res_choice(x)[:, None, ...] for res_choice in self.res_choices]
        inits = [init_choice(x)[:, None, ...] for init_choice in self.init_choices]
        filts = [filt_choice(x)[:, None, ...] for filt_choice in self.filt_choice]

        resonances = [self.bank.forward(res, inits[i], filts[i]) for i, res in enumerate(resonances)]
        # resonances = [unit_norm(r) for r in resonances]

        impulse = F.pad(impulse, (0, self.total_samples - impulse_samples))
        impulse = impulse.view(-1, 1, self.total_samples)
        # impulse = unit_norm(impulse)

        resonances = torch.cat(resonances, dim=1).view(-1, self.mix_channels, self.total_samples)

        # final will be (batch * n_events, mix_channels, samples)
        final = fft_convolve(resonances, impulse)

        # this will generate a mix over time of the different convolutions
        # with a resonance function
        mixed_down = self.generate_mix.forward(x, final)

        impulse = impulse.view(*mixed_down.shape)
        imp_and_res = torch.cat([impulse[..., None], mixed_down[..., None]], dim=-1)

        x = imp_and_res * final_mix
        x = torch.sum(x, dim=-1)

        return x


class ResonanceChain(nn.Module):
    def __init__(
            self,
            depth,
            n_atoms,
            window_size,
            n_frames,
            total_samples,
            mix_channels,
            channels,
            latent_dim,
            initial,
            learnable_resonances=True):
        super().__init__()
        self.n_atoms = n_atoms
        self.window_size = window_size
        self.n_frames = n_frames
        self.total_samples = total_samples
        self.mix_channels = mix_channels
        self.channels = channels
        self.latent_dim = latent_dim
        self.depth = depth
        self.initial = initial

        self.res = nn.ModuleList([
            ResonanceBlock(
                n_atoms,
                window_size,
                n_frames,
                total_samples,
                mix_channels,
                channels,
                latent_dim,
                initial,
                learnable_resonances)
            for _ in range(depth)
        ])

        self.to_mix = nn.Linear(latent_dim, depth)

    def forward(self, latent: torch.Tensor, impulse: torch.Tensor):
        batch_size = latent.shape[0]

        imp = impulse
        outputs = []

        for i in range(self.depth):
            imp = self.res[i].forward(latent, imp)
            outputs.append(imp[..., None])

        outputs = torch.cat(outputs, dim=-1)
        mx = self.to_mix(latent).view(batch_size, -1, 1, self.depth)

        # print(f'\t OUTPUTS {outputs.shape} {mx.shape}')

        outputs = outputs * mx
        outputs = torch.sum(outputs, dim=-1)

        return outputs


def fft_convolve(*args, correlation=False):
    args = list(args)

    n_samples = args[0].shape[-1]

    # pad to avoid wraparound artifacts
    padded = [F.pad(x, (0, x.shape[-1])) for x in args]

    specs = [torch.fft.rfft(x, dim=-1) for x in padded]

    if correlation:
        # perform the cross-correlation instead of convolution.
        # this is what torch's convolution functions and modules 
        # perform
        specs[1] = torch.conj(specs[1])

    spec = reduce(lambda accum, current: accum * current, specs[1:], specs[0])
    final = torch.fft.irfft(spec, dim=-1)

    # remove padding
    final = final[..., :n_samples]
    return final


def fft_shift(a, shift):
    n_samples = a.shape[-1]
    a = F.pad(a, (0, n_samples))
    shift_samples = (shift * 0.5) * n_samples
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs) * 2j * np.pi).to(a.device) / n_coeffs
    shift = torch.exp(-shift * shift_samples)
    spec = spec * shift
    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples


def position(x, clips, n_samples, sum_channels=False):
    if len(x.shape) != 2:
        raise ValueError('positions shoud be (batch, n_clips)')

    batch_size, n_clips = x.shape

    n_clips = clips.shape[1]

    # we'd like positions to be (batch, positions)
    x = x.view(-1, n_clips)

    # we'd like clips to be (batch, n_clips, n_samples)
    clips = clips.view(-1, n_clips, n_samples)

    if clips.shape[0] == 1:
        # we're using the same set of stems for every
        # batch
        clips = clips.repeat(batch_size, 1, 1)

    outer = []

    for i in range(batch_size):
        inner = []
        for j in range(n_clips):
            canvas = torch.zeros(n_samples, device=clips.device)
            current_index = (x[i, j] * n_samples).long()
            current_stem = clips[i, j]
            duration = n_samples - current_index
            canvas[current_index: current_index + duration] += current_stem[:duration]
            inner.append(canvas)
        canvas = torch.stack(inner)
        outer.append(canvas)

    outer = torch.stack(outer)
    if sum_channels:
        outer = torch.sum(outer, dim=1, keepdim=True)
    return outer


class ScalarPosition(torch.autograd.Function):

    def forward(self, positions, n_samples):
        indices = (positions * n_samples * 0.9999).long()
        self.save_for_backward(indices)
        batch, n_examples = positions.shape[:2]
        one_hot = torch.zeros(batch, n_examples, n_samples)
        one_hot = torch.scatter(one_hot, dim=-1, index=indices, src=torch.ones_like(positions))
        return one_hot

    def backward(self, *grad_outputs):
        x, = grad_outputs

        indices, = self.saved_tensors
        grads = []

        for b in range(x.shape[0]):
            for i, index in enumerate(indices[b]):
                left, right = x[b, i, index:], x[b, i, :index]
                scalar_grad = left.sum() - right.sum()
                grads.append(scalar_grad.view(-1))

        grads = torch.cat(grads).view(x.shape[0], -1, 1)

        # x = x.sum(dim=-1)
        # x = torch.cat
        # x = p2.grad.mean(dim=1)
        # x = torch.cat([x[50:], x[:50]])

        # indices, = self.saved_tensors
        # x = torch.gather(x, dim=-1, index=indices)

        return grads, None


scalar_position = ScalarPosition.apply


class FFTShifter(torch.autograd.Function):
    def forward(self, items, positions):
        positions.retain_grad()
        self.save_for_backward(items, positions)
        result = fft_shift(items, positions)
        return result

    def backward(self, *grad_outputs):
        x, = grad_outputs
        items, positions = self.saved_tensors
        return x


differentiable_fft_shift = FFTShifter.apply


class Position(torch.autograd.Function):
    counter = 0

    def forward(self, items, positions, targets):
        x = position(positions, items, items.shape[-1])
        self.save_for_backward(positions, targets, items, x)
        Position.counter += 1
        return x

    def backward(self, *grad_outputs: Collection[torch.Tensor]):
        x, = grad_outputs
        pos, targets, clips, recon = self.saved_tensors

        batch = x.shape[0]
        n_samples = x.shape[-1]

        # what's the best possible position for each stem?
        targets = targets.view(batch, 1, n_samples)
        clips = clips.view(-1, pos.shape[1], n_samples)

        conv = fft_convolve(targets, clips, correlation=True)

        real_best = torch.argmax(conv, dim=-1) / conv.shape[-1]

        # what's the difference between the scalar location
        # and the actual best position for this clip?
        pos_grad = (pos - real_best)

        p = real_best

        # gradient for clips based on the best possible render
        best_render = fft_shift(clips, p[..., None])
        clip_loss = best_render - targets
        # shift the gradients backward to align with the clip
        clip_loss = fft_shift(clip_loss, -p[..., None])

        return clip_loss, pos_grad, None


schedule_atoms = Position.apply


class AtomScheduler(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, items, positions, targets):
        return schedule_atoms(items, positions, targets)


class PosEncodedImpulseGenerator(nn.Module):
    def __init__(self, n_frames, final_size, softmax=lambda x: torch.softmax(x, dim=-1), scale_frequencies=False):
        super().__init__()
        self.n_frames = n_frames
        self.final_size = final_size
        self.softmax = softmax
        self.scale_frequencies = scale_frequencies

    def forward(self, p, softmax=None):
        sm = softmax or self.softmax

        batch, _ = p.shape

        norms = torch.norm(p, dim=-1, keepdim=True)
        p = p / (norms + 1e-8)

        pos = pos_encoded(batch, self.n_frames, 16, device=p.device)

        if self.scale_frequencies:
            scaling = torch.linspace(1, 0, steps=33, device=p.device) ** 2
            pos = pos * scaling[None, None, :]

        norms = torch.norm(pos, dim=-1, keepdim=True)
        pos = pos / (norms + 1e-8)

        sim = (pos @ p[:, :, None]).view(batch, 1, self.n_frames)
        orig_sim = sim

        sim = sm(sim)

        # sim = F.interpolate(sim, size=self.final_size, mode='linear')

        output = torch.zeros(batch, 1, self.final_size, device=sim.device)
        step = self.final_size // self.n_frames
        output[:, :, ::step] = sim

        return output, orig_sim


class ImpulseGenerator(nn.Module):
    def __init__(self, final_size, softmax=lambda x: torch.softmax(x, dim=-1)):
        super().__init__()
        self.final_size = final_size
        self.softmax = softmax

    def forward(self, x, softmax=None, return_logits=False):
        sm = softmax or self.softmax

        batch, time = x.shape
        x = x.view(batch, 1, time)
        x = sm(x)
        step = self.final_size // time
        output = torch.zeros(batch, 1, self.final_size, device=x.device)
        output[:, :, ::step] = x
        if return_logits:
            return output, x
        else:
            return output


class STFTTransferFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 512
        self.n_coeffs = self.window_size // 2 + 1
        self.n_samples = 2 ** 15
        self.step_size = self.window_size // 2
        self.n_frames = self.n_samples // self.step_size

        self.dim = self.n_coeffs * 2

    def forward(self, tf):
        batch, n_coeffs = tf.shape
        if n_coeffs != self.dim:
            raise ValueError(f'Expected (*, {self.dim}) but got {tf.shape}')

        tf = tf.view(-1, self.n_coeffs * 2, 1)
        tf = tf.repeat(1, 1, self.n_frames)
        tf = tf.view(-1, self.n_coeffs, 2, self.n_frames)

        tf = tf.view(-1, self.n_coeffs * 2, self.n_frames)

        real = torch.clamp(tf[:, :self.n_coeffs, :], 0, 1) * 0.9999
        imag = torch.clamp(tf[:, self.n_coeffs:, :], -1, 1) * np.pi

        real = real * torch.cos(imag)
        imag = real * torch.sin(imag)
        tf = torch.complex(real, imag)
        tf = torch.cumprod(tf, dim=-1)

        tf = tf.view(-1, self.n_coeffs, self.n_frames)
        tf = torch.fft.irfft(tf, dim=1, norm='ortho') \
            .permute(0, 2, 1) \
            .view(batch, 1, self.n_frames, self.window_size)
        tf = overlap_add(tf, trim=self.n_samples)
        return tf


# class TransferFunction(nn.Module):

#     def __init__(
#             self, 
#             samplerate: zounds.SampleRate, 
#             scale: zounds.FrequencyScale, 
#             n_frames: int, 
#             resolution: int,
#             n_samples: int,
#             softmax_func: Any,
#             is_continuous=False,
#             resonance_exp=1,
#             reduction=lambda x: torch.mean(x, dim=-1, keepdim=True)):

#         super().__init__()
#         self.samplerate = samplerate
#         self.scale = scale
#         self.n_frames = n_frames
#         self.resolution = resolution
#         self.n_samples = n_samples
#         self.softmax_func = softmax_func
#         self.is_continuous = is_continuous
#         self.resonance_exp = resonance_exp
#         self.reduction = reduction

#         bank = morlet_filter_bank(
#             samplerate, n_samples, scale, 0.1, normalize=False)\
#             .real.astype(np.float32)

#         self.register_buffer('filter_bank', torch.from_numpy(bank)[None, :, :])

#         resonances = (torch.linspace(0, 0.999, resolution) ** resonance_exp)\
#             .view(resolution, 1).repeat(1, n_frames)
#         resonances = torch.cumprod(resonances, dim=-1)
#         self.register_buffer('resonance', resonances)

#     @property
#     def n_bands(self):
#         return self.scale.n_bands

#     def forward(self, x: torch.Tensor):
#         batch, bands, resolution = x.shape

#         if not self.is_continuous:
#             if bands != self.n_bands or resolution != self.resolution:
#                 raise ValueError(
#                     f'Expecting tensor with shape (*, {self.n_bands}, {self.resolution})')
#             x = self.softmax_func(x)
#             x = x @ self.resonance
#         else:
#             x = (torch.clamp(x, 0, 1) ** self.resonance_exp) * 0.9999
#             x = torch.cumprod(x, dim=-1)

#         x = F.interpolate(x, size=self.n_samples, mode='linear')

#         x = x * self.filter_bank
#         x = self.reduction(x)
#         return x


'''
# TODO: try matrix rotation instead: https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf



def test():

    n_samples = 2 ** 15
    window_size = 1024
    step_size = window_size // 2
    n_coeffs = window_size // 2 + 1
    
    impulse = torch.zeros(1, 1, 2048).uniform_(-1, 1)
    impulse = F.pad(impulse, (0, n_samples - 2048))
    windowed = windowed_audio(impulse, window_size, step_size)
    
    n_frames = windowed.shape[-2]
    
    transfer_func = torch.zeros(1, n_coeffs).uniform_(0, 0.99)
    print(torch.norm(transfer_func).item())
    transfer_warp = torch.eye(n_coeffs)
    transfer_warp = torch.roll(transfer_warp, (0, 4), dims=(0, 1))
    
    
    frames = []
    
    for i in range(n_frames):
        
        transfer_func = transfer_func @ transfer_warp
        print(torch.norm(transfer_func).item())
        
        if i == 0:
            spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
            spec = spec * transfer_func
            audio = torch.fft.irfft(spec, dim=-1)
            frames.append(audio)
        else:
            prev = frames[i - 1]
            prev_spec = torch.fft.rfft(prev, dim=-1)
            prev_spec = advance_one_frame(prev_spec)
            
            current_spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
            spec = current_spec + prev_spec
            spec = spec * transfer_func
            audio = torch.fft.irfft(spec, dim=-1)
            frames.append(audio)
    
    
    frames = torch.cat([f[:, :, None, :] for f in frames], dim=2)
    audio = overlap_add(frames, apply_window=True)[..., :n_samples]
    
    return audio
'''


def to_polar(x):
    # mag = torch.abs(x)
    # phase = torch.angle(x)
    # return mag, phase
    real = x.real[..., None]
    imag = x.imag[..., None]
    return torch.cat([real, imag], dim=-1)


def to_complex(x):
    return torch.complex(x[..., 0], x[..., 1])


def advance_one_frame(x):
    batch, _, coeffs = x.shape

    x = to_polar(x)
    # print('BEFORE ROTATION', x.shape)
    # mag, phase = x[..., 0], x[..., 1]

    group_delay = torch.linspace(0, np.pi, x.shape[-2], device=x.device)
    s = torch.sin(group_delay)
    c = torch.cos(group_delay)
    rotation_matrices = torch.cat([c[:, None], -s[:, None], s[:, None], -c[:, None]], dim=-1).reshape(-1, 2, 2)
    # print(rotation_matrices.shape)

    # print(phase.shape, rotation_matrices.shape)
    # phase = phase @ rotation_matrices
    x = x.view(batch, coeffs, 1, 2) @ rotation_matrices
    # print('AFTER ROTATION', x.shape)
    x = x.view(batch, 1, coeffs, 2)
    # phase = phase + torch.linspace(0, np.pi, x.shape[-1], device=x.device)[None, None, :]
    # x = to_complex(mag, phase)
    return to_complex(x)

    # return x


class STFTResonanceGenerator(nn.Module):
    def __init__(self, window_size, n_samples, z_dim, inner_channels):
        super().__init__()
        self.window_size = window_size
        self.step_size = window_size // 2
        self.n_samples = n_samples
        self.n_coeffs = window_size // 2 + 1
        self.z_dim = z_dim
        self.n_frames = n_samples // self.step_size
        self.inner_channels = inner_channels

        self.to_initial_transfer_function = nn.Linear(self.z_dim, self.n_coeffs)

        self.base_resonance = 0.02
        self.resonance_range = (1 - self.base_resonance) * 0.99

        self.to_transfer_function = ConvUpsample(
            self.z_dim,
            self.inner_channels,
            start_size=8,
            end_size=self.n_frames,
            mode='nearest',
            out_channels=self.n_coeffs,
            weight_norm=True,
            from_latent=True)

        # self.hypernetwork = HyperNetworkLayer(
        #     self.inner_channels, self.z_dim, self.n_coeffs, self.n_coeffs)

    def forward(self, z, impulse):
        batch, n_events, impulse_samples = impulse.shape
        z_dim = z.shape[-1]

        impulse = F.pad(impulse, (0, self.n_samples - impulse_samples + self.window_size))
        windowed = windowed_audio(impulse, self.window_size, self.step_size)

        # generate the initial transfer function
        # transfer_func = \
        #     self.base_resonance \
        #     + (self.resonance_range * torch.sigmoid(self.to_initial_transfer_function(z)))

        z = z.view(-1, self.z_dim)
        transfer_funcs = self \
            .to_transfer_function(z) \
            .view(batch, n_events, self.n_coeffs, self.n_frames) \
            .permute(0, 1, 3, 2) \
            .view(batch, n_events, self.n_frames, self.n_coeffs)

        # make sure transfer funcs are in the allowable range        
        transfer_funcs = self.base_resonance + (torch.sigmoid(transfer_funcs) * self.resonance_range)

        frames = []

        for i in range(self.n_frames):

            # get the transformation matrix for this frame

            # apply the transformation to get the new transformation matrix
            transfer_func = transfer_funcs[:, :, i, :]

            if i == 0:
                spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
                spec = spec * transfer_func
                audio = torch.fft.irfft(spec, dim=-1)
                frames.append(audio)
            else:
                prev = frames[i - 1]
                prev_spec = torch.fft.rfft(prev, dim=-1)
                prev_spec = advance_one_frame(prev_spec)

                current_spec = torch.fft.rfft(windowed[:, :, i, :], dim=-1)
                spec = current_spec + prev_spec
                spec = spec * transfer_func
                audio = torch.fft.irfft(spec, dim=-1)
                frames.append(audio)

        frames = torch.cat([f[:, :, None, :] for f in frames], dim=2)

        frames = frames.view(batch, self.n_frames, n_events, self.window_size).permute(0, 2, 1, 3)

        audio = overlap_add(frames, apply_window=True)[..., :self.n_samples]

        return audio
