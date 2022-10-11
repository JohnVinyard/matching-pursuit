from typing import Collection
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.distributions.normal import Normal
from matplotlib import pyplot as plt
from functools import reduce
from scipy.signal import stft
from torch.optim import Adam
from modules.transfer import schedule_atoms
from util import device
from data.audioiter import AudioIterator

import matplotlib
matplotlib.use('qt5agg', force=True)
import matplotlib.pyplot as plt

class FFTShifter(torch.autograd.Function):

    def forward(self, items, pos):
        return fft_shift(items, pos)
    
    def backward(self, *grad_outputs):
        a, b = grad_outputs.shape
        return a, b

differentiable_fft_shift = FFTShifter.apply


'''
The main insights here are that sub-20-hz losses
become jagged and unstable near the event due to phase
issues, making them very hard to train.

STFT, which ignores phase issues below 20-hz is smooth
and convex, but begins to fall again *AT THE BOUNDARIES*
since the positioned clip begins to disappear

That said, choosing a domain that avoids boundary issues
and guarantees a convex loss landscape
'''

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


# class ScalarPositioning(torch.autograd.Function):
#     def forward(self, positions: torch.Tensor, n_samples: int):
#         batch, _ = positions.shape
#         self.save_for_backward(positions)
#         positions = (positions.view(batch, 1) * n_samples * 0.9999).long()
#         impulses = torch.zeros(batch, n_samples)
#         impulses = torch.scatter(impulses, -1, positions, torch.ones(positions.shape))
#         return impulses
    
#     def backward(self, *grad_outputs):
#         x, = grad_outputs
#         pos, = self.saved_tensors
#         x = x.view(-1, 1, n_samples)
#         x = F.relu(x)
#         index = torch.argmax(x, dim=-1)
#         new_pos = index / n_samples
#         grad = pos - new_pos
#         return grad, None


# scalar_position = ScalarPositioning.apply



def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.uniform_(-0.02, 0.02)
        except AttributeError:
            pass

        try:
            p.bias.fill_(0)
        except AttributeError:
            pass

def pos_encode_feature(x, domain, n_samples, n_freqs):
    x = torch.clamp(x, -domain, domain)
    output = [x]
    for i in range(n_freqs):
        output.extend([
            torch.sin((2 ** i) * x),
            torch.cos((2 ** i) * x)
        ])

    x = torch.cat(output, dim=-1)
    return x

def n_features_for_freq(n_freqs):
    return n_freqs * 2 + 1


def pos_encoded(batch_size, time_dim, n_freqs, device=None):
    """
    Return positional encodings with shape (batch_size, time_dim, n_features)
    """
    n_features = n_features_for_freq(n_freqs)
    pos = pos_encode_feature(torch.linspace(-1, 1, time_dim).view(-1, 1), 1, time_dim, n_freqs)\
        .view(1, time_dim, n_features)\
        .repeat(batch_size, 1, 1)\
        .view(batch_size, time_dim, n_features)\

    if device:
        pos = pos.to(device)
    return pos



def stft(x, ws=512, step=256, pad=False, log_amplitude=False, log_epsilon=1e-4):
    if pad:
        x = F.pad(x, (0, step))
    
    x = x.unfold(-1, ws, step)
    win = torch.hann_window(ws).to(x.device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x, norm='ortho')
    x = torch.abs(x)
    
    if log_amplitude:
        x = torch.log(x + log_epsilon)
    
    return x


def fft_shift(a, shift):
    n_samples = a.shape[-1]
    a = F.pad(a, (0, n_samples))
    shift_samples = (shift * 0.5) * n_samples
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')
    
    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs) * 2j * np.pi).to(device) / n_coeffs
    shift = torch.exp(-shift * shift_samples)
    spec = spec * shift
    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples



def normalize(x):
    mx, _ = torch.max(torch.abs(x), dim=-1, keepdim=True)
    return x / (mx + 1e-8)




def position(x, clips, n_samples, sum_channels=False):
    
    if len(x.shape) != 2:
        raise ValueError('pos shoud be (batch, n_clips)')

    batch_size, n_clips = x.shape

    n_clips = clips.shape[0]

    # we'd like positions to be (batch, positions)
    x = x.view(-1, n_clips)

    # we'd like clips to be (batch, n_clips, n_samples)
    clips = clips.view(-1, n_clips, n_samples)

    if clips.shape[0] == 1:
        # we're using the same set of stems for every
        # batch
        clips = clips.repeat(batch_size, 1, 1)

    positions = x

    outer = []

    for i in range(batch_size):
        inner = []
        for j in range(n_clips):
            canvas = torch.zeros(n_samples)
            current_index = (x[i, j] * n_samples).long()
            current_stem = clips[i, j]
            duration = n_samples - current_index
            canvas[current_index: current_index + duration] = current_stem[:duration]
            canvas.requires_grad = True
            inner.append(canvas)
        canvas = torch.stack(inner)
        outer.append(canvas)
    
    outer = torch.stack(outer)
    if sum_channels:
        outer = torch.sum(outer, dim=1, keepdim=True)
    
    return outer


        

class STFTLoss(nn.Module):
    def __init__(self, n_samples, log_amp=False):
        super().__init__()
        self.n_samples = n_samples
        self.log_amp = log_amp
    
    def forward(self, input, target):
        batch = input.shape[0]

        input = input.view(batch, -1, self.n_samples)
        target = target.view(batch, -1, self.n_samples)

        input = torch.sum(input, dim=1, keepdim=True)

        input = stft(input.view(-1, self.n_samples), log_amplitude=self.log_amp)
        target = stft(target.view(-1, self.n_samples), log_amplitude=self.log_amp)
        return ((input - target.detach()) ** 2).mean(dim=(1, 2))




class CompositeLoss(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.losses = args
    
    def forward(self, input, target):
        return sum([l(input, target) for l in self.losses])


class Model(nn.Module):
    def __init__(self, n_samples, stems):
        super().__init__()
        self.n_samples = n_samples
        self.embed_spec = nn.Linear(257, 33)
        self.embed = nn.Linear(66, 128)


        self.net = nn.Sequential(
            nn.Conv1d(128, 128, 3, 1, dilation=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, 1, dilation=3, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, 1, dilation=9, padding=9),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, 1, dilation=27, padding=27),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, 1, dilation=1, padding=1),
        )

        self.to_pos = nn.Linear(128, 8)
        self.stems = nn.Parameter(torch.zeros_like(stems).uniform_(-1, 1))


        self.apply(init_weights)
    
    def forward(self, x):
        x = x.view(-1, self.n_samples)
        target = x

        spec = stft(x, 512, 256, pad=True, log_amplitude=True)
        spec = self.embed_spec(spec)
        pos = pos_encoded(x.shape[0], spec.shape[1], 16, device=spec.device)
        x = torch.cat([spec, pos], dim=-1)
        x = self.embed(x)

        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)

        x, _ = torch.max(x, dim=1)
        # x = x[:, -1, :]
        x = self.to_pos(x)
        
        locations = torch.sigmoid(x)
        pos = locations.view(batch_size, 8, 1)

        # x = schedule_atoms(self.stems.view(1, -1, n_samples), locations, target)
        x = differentiable_fft_shift(self.stems.view(1, 8, -1), pos)

        x = torch.sum(x, dim=1, keepdim=True)

        final = x
        return x, locations, final


class DataSet(object):
    def __init__(
        self, 
        batch_size, 
        clip, 
        scheduler, 
        n_samples=2**15, 
        samplerate=22050,
        range_low = 0,
        range_high = 1):
        
        super().__init__()
        self.batch_size = batch_size
        self.clip = clip
        self.scheduler = scheduler
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.range_low = range_low
        self.range_high = range_high
    
    def ordered_batch(self, n_positions):
        locations = torch.linspace(self.range_low, self.range_high, n_positions)
        return self.scheduler.forward(locations.view(-1, 1), self.clip[None, :], None)
    
    def __iter__(self):
        while True:
            locations = torch.zeros(self.batch_size, 1).uniform_(self.range_low, self.range_high)
            examples = self.scheduler.forward(locations, self.clip[None, :], None)
            yield examples, locations




# def match_conv(signal, stem):
#     '''
#     input – input tensor of shape (\text{minibatch} , \text{in\_channels} , iW)(minibatch,in_channels,iW)

#     weight – filters of shape (\text{out\_channels} , \frac{\text{in\_channels}}{\text{groups}} , kW)(out_channels, 
#         groups
#         in_channels
# ​
#         ,kW)
#     '''
#     # signal = torch.zeros(8192)
#     # signal[4096] = 1
#     signal = signal.view(1, 1, -1)

#     # stem = torch.zeros(8192)
#     # stem[0] = 1
#     stem = stem.view(1, 1, -1)

#     padded_signal = F.pad(signal, (0, signal.shape[-1]))
#     a = F.conv1d(padded_signal, stem)

#     print(a.shape)
#     mx = torch.argmax(torch.abs(a), dim=-1)
#     print(mx)
    
#     b = fft_convolve(signal, stem, correlation=True)
#     mx = torch.argmax(torch.abs(b), dim=-1)
#     print(b.shape)
#     print(mx)

#     a = torch.abs(a)
#     b = torch.abs(b)

        # self.net = nn.TransformerEncoder(layer, 3)
#     return a, b

if __name__ == '__main__':

    n_samples = 2**15
    batch_size = 4
    samplerate = 22050
    stem_size = 4096

    audio_iter = AudioIterator(8, stem_size, 22050, normalize=True)
    stems = next(audio_iter.__iter__())
    stems = stems.view(8, 1, stem_size)
    stems = F.pad(stems, (0, n_samples - stem_size))

    # scheduler = RollScheduler(n_samples)

    # TODO: Instead of synthetic examples, these will be
    # normal batches of (batch, 1, n_samples)
    def random_arrangement():
        positions = torch.zeros(1, 8).uniform_(0, 1)
        result = position(positions, stems, n_samples, sum_channels=True)
        return result

    def random_arrangement_batch(batch_size):
        return torch.stack([random_arrangement() for _ in range(batch_size)])
    
    def iter_random_arrangement_batches(batch_size):
        while True:
            yield random_arrangement_batch(batch_size)
    

    # clip = generate_clip(n_samples, samplerate)

    # scheduler = NormalPDFScheduler(n_samples)
    # scheduler = FFTShiftScheduler(n_samples)

    # dataset = DataSet(
    #     batch_size, 
    #     clip, 
    #     scheduler, 
    #     n_samples, 
    #     samplerate, 
    #     range_low=0, 
    #     range_high=1)
    
    model = Model(n_samples, stems)
    optim = Adam(model.parameters(), lr=1e-3, betas=(0, 0.9))

    loss = CompositeLoss(
        STFTLoss(n_samples),
    )
    
    i = 0
    losses = []

    # # train network
        # self.net = nn.TransformerEncoder(layer, 3)
    for item in iter_random_arrangement_batches(batch_size):

        item = item.view(batch_size, 1, n_samples)

        optim.zero_grad()
        recon, fake_loc, final = model.forward(item)

        
        l = loss.forward(recon, item).mean()
        l.backward()
        optim.step()
        print(i, l.item())
        losses.append(l.item())

        i += 1

        if i % 1000 == 0:

            plt.plot(losses)
            plt.show()

            plt.plot(-np.abs(recon[0].data.cpu().numpy().squeeze()))
            plt.plot(np.abs(item[0].data.cpu().numpy().squeeze()))
            plt.show()
