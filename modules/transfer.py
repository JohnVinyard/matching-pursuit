from functools import reduce
from typing import Any, Collection
import torch
from torch import nn
import zounds

from modules.ddsp import overlap_add
from modules.pos_encode import pos_encoded
from .phase import morlet_filter_bank
import numpy as np
from torch.nn import functional as F


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



class Position(torch.autograd.Function):

    def forward(self, items, positions, targets):
        x = position(positions, items, items.shape[-1])
        self.save_for_backward(positions, targets, items, x)
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
        real_best = torch.argmax(conv, dim=-1) / n_samples

        # what's the difference between the scalar location
        # and the actual best position for this clip?
        pos_grad = (pos - real_best)

        # gradient for clips based on the best possible render
        best_render = fft_shift(clips, real_best[..., None])
        clip_loss = best_render - targets
        # shift the gradients backward to align with the clip
        clip_loss = fft_shift(clip_loss, -real_best[..., None])

        return clip_loss, pos_grad, None


schedule_atoms = Position.apply

class AtomScheduler(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, items, positions, targets):
        return schedule_atoms(items, positions, targets)


class PosEncodedImpulseGenerator(nn.Module):
    def __init__(self, n_frames, final_size, softmax=lambda x: torch.softmax(x, dim=-1)):
        super().__init__()
        self.n_frames = n_frames
        self.final_size = final_size
        self.softmax = softmax
    
    def forward(self, p):
        batch, _ = p.shape

        norms = torch.norm(p, dim=-1, keepdim=True)
        p = p / (norms + 1e-8)

        pos = pos_encoded(batch, self.n_frames, 16, device=p.device)
        norms = torch.norm(pos, dim=-1, keepdim=True)
        pos = pos / (norms + 1e-8)

        sim = (pos @ p[:, :, None]).view(batch, 1, self.n_frames)
        orig_sim = sim

        sim = self.softmax(sim)

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
    
    def forward(self, x):
        batch, time = x.shape
        x = x.view(batch, 1, time)
        x = self.softmax(x)
        step = self.final_size // time
        output = torch.zeros(batch, 1, self.final_size, device=x.device)
        output[:, :, ::step] = x
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
        tf = torch.fft.irfft(tf, dim=1, norm='ortho')\
            .permute(0, 2, 1)\
            .view(batch, 1, self.n_frames, self.window_size)
        tf = overlap_add(tf, trim=self.n_samples)
        return tf



class TransferFunction(nn.Module):

    def __init__(
            self, 
            samplerate: zounds.SampleRate, 
            scale: zounds.FrequencyScale, 
            n_frames: int, 
            resolution: int,
            n_samples: int,
            softmax_func: Any):

        super().__init__()
        self.samplerate = samplerate
        self.scale = scale
        self.n_frames = n_frames
        self.resolution = resolution
        self.n_samples = n_samples
        self.softmax_func = softmax_func

        bank = morlet_filter_bank(
            samplerate, n_samples, scale, 0.1, normalize=False)\
            .real.astype(np.float32)
        
        self.register_buffer('filter_bank', torch.from_numpy(bank)[None, :, :])

        resonances = torch.linspace(0, 0.999, resolution)\
            .view(resolution, 1).repeat(1, n_frames)
        resonances = torch.cumprod(resonances, dim=-1)
        self.register_buffer('resonance', resonances)
    
    @property
    def n_bands(self):
        return self.scale.n_bands
    
    def forward(self, x: torch.Tensor):
        batch, bands, resolution = x.shape
        if bands != self.n_bands or resolution != self.resolution:
            raise ValueError(
                f'Expecting tensor with shape (*, {self.n_bands}, {self.resolution})')
        
        # x = F.gumbel_softmax(x, dim=-1, hard=True)
        # x = F.softmax(x, dim=-1)
        x = self.softmax_func(x)
    
        x = x @ self.resonance
        x = F.interpolate(x, size=self.n_samples, mode='linear')
        x = x * self.filter_bank
        x = torch.mean(x, dim=1, keepdim=True)
        return x


    
