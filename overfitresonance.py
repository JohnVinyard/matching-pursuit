from typing import Union
import torch
from torch import nn
from data.audioiter import AudioIterator
from modules import stft
from modules.fft import fft_convolve
from conjure import LmdbCollection
from modules.iterative import TensorTransform, iterative_loss
from modules.quantize import select_items
from modules.softmax import sparse_softmax
from modules.upsample import interpolate_last_axis, upsample_with_holes
from torch.nn import functional as F
from torch.optim import Adam
from itertools import count
import numpy as np

from util import device

collection = LmdbCollection(path='overfitresonance')

samplerate = 22050
n_samples = 2 ** 16
n_frames = 256
n_events = 32

def fft_shift(a: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    
    # this is here to make the shift value interpretable
    shift = (1 - shift)
    
    n_samples = a.shape[-1]
    
    shift_samples = (shift * n_samples * 0.5)
    
    # a = F.pad(a, (0, n_samples * 2))
    
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    
    shift = torch.exp(shift * shift_samples)

    spec = spec * shift
    
    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    # samples = torch.relu(samples)
    return samples

class Lookup(nn.Module):
    
    def __init__(
            self, 
            n_items: int, 
            n_samples: int,
            initialize: Union[None, TensorTransform] = None):
        
        super().__init__()
        self.n_items = n_items
        self.n_samples = n_samples
        data = torch.zeros(n_items, n_samples)
        initialized = data.uniform_(-1, 1) if initialize is None else initialize(data)
        self.items = nn.Parameter(initialized)
    
    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        return items
    
    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        return items
    
    def forward(self, selections: torch.Tensor) -> torch.Tensor:
        items = self.preprocess_items(self.items)
        selected = select_items(selections, items, type='sparse_softmax')
        processed = self.postprocess_results(selected)
        return processed


def flatten_envelope(x: torch.Tensor, kernel_size: int, step_size: int):
    """
    Given a signal with time-varying amplitude, give it as uniform an amplitude
    over time as possible
    """
    env = torch.abs(x)
    
    normalized = x / (env.max(dim=-1, keepdim=True)[0] + 1e-3)
    env = F.max_pool1d(
        env, 
        kernel_size=kernel_size, 
        stride=step_size, 
        padding=step_size)
    env = 1 / env
    env = interpolate_last_axis(env, desired_size=x.shape[-1])
    result = normalized * env
    return result


class SampleLookup(Lookup):
    def __init__(self, n_items: int, n_samples: int, flatten_kernel_size: int):
        super().__init__(n_items, n_samples)
        self.flatten_kernel_size = flatten_kernel_size

    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have audio-rate samples at a relatively uniform
        amplitude throughout
        """
        return flatten_envelope(
            items, 
            kernel_size=self.flatten_kernel_size, 
            step_size=self.flatten_kernel_size // 2)


class Decays(Lookup):
    def __init__(self, n_items: int, n_samples: int, full_size: int):
        super().__init__(n_items, n_samples)
        self.full_size = full_size
    
    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have all values between 0 and 1
        """
        items = items - items.min()
        items = items / (items.max() + 1e-3)
        return items
    
    def postprocess_results(self, decay: torch.Tensor) -> torch.Tensor:
        """Apply a scan in log-space to end up with exponential decay
        """
        decay = torch.log(decay + 1e-12)
        decay = torch.cumsum(decay, dim=-1)
        amp = interpolate_last_axis(decay, desired_size=self.full_size)
        return amp


class Envelopes(Lookup):
    def __init__(self, n_items: int, n_samples: int, full_size: int):
        super().__init__(n_items, n_samples)
        self.full_size = full_size
    
    def preprocess_items(self, items: torch.Tensor) -> torch.Tensor:
        """Ensure that we have all values between 0 and 1
        """
        items = items - items.min()
        items = items / (items.max() + 1e-3)
        return items
    
    def postprocess_results(self, decay: torch.Tensor) -> torch.Tensor:
        """Scale up to sample rate and multiply with noise
        """
        amp = interpolate_last_axis(decay, desired_size=self.full_size)
        amp = amp * torch.zeros_like(amp).uniform_(-1, 1)
        return amp
        
class Deformations(Lookup):
    
    def __init__(self, n_items: int, channels: int, frames: int, full_size: int):
        super().__init__(n_items, channels * frames)
        self.full_size = full_size
        self.channels = channels
        self.frames = frames
    
    def postprocess_results(self, items: torch.Tensor) -> torch.Tensor:
        """Reshape so that the values are (..., channels, frames).  Apply
        softmax to the channel dimension and then upscale to full samplerate
        """
        
        shape = items.shape[:-1]
        x = items.reshape(*shape, self.channels, self.frames)
        x = torch.softmax(x, dim=-2)
        x = interpolate_last_axis(x, desired_size=self.full_size)
        return x
        

class DiracScheduler(nn.Module):
    def __init__(self, n_events: int, start_size: int, n_samples: int):
        super().__init__()
        self.n_events = n_events
        self.start_size = start_size
        self.n_samples = n_samples
        self.pos = nn.Parameter(
            torch.zeros(1, n_events, start_size).uniform_(-1, 1)
        )
    
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        pos = sparse_softmax(self.pos, normalize=True, dim=-1)
        pos = upsample_with_holes(pos, desired_size=self.n_samples)
        final = fft_convolve(events, pos)
        return final

class FFTShiftScheduler(nn.Module):
    def __init__(self, n_events):
        super().__init__()
        self.n_events = n_events
        self.pos = nn.Parameter(torch.zeros(1, n_events, 1))
    
    def forward(self, events: torch.Tensor) -> torch.Tensor:
        final = fft_shift(events, self.pos)
        return final


class OverfitResonanceModel(nn.Module):
    
    def __init__(
            self, 
            instr_expressivity: int,
            n_events: int, 
            n_resonances: int, 
            n_envelopes: int, 
            n_decays: int, 
            n_deformations: int,
            n_samples: int,
            n_frames: int,
            samplerate: int):
            
        super().__init__()
        
        self.samplerate = samplerate
        self.n_events = n_events
        self.n_samples = n_samples
        
        # choices/selections
        self.resonances = nn.Parameter(
            torch.zeros(1, n_events, instr_expressivity, n_resonances).uniform_(-1, 1))
        self.envelopes = nn.Parameter(
            torch.zeros(1, n_events, n_envelopes).uniform_(-1, 1))
        self.decays = nn.Parameter(
            torch.zeros(1, n_events, n_decays).uniform_(-1, 1))
        self.deformations = nn.Parameter(
            torch.zeros(1, n_events, n_deformations).uniform_(-1, 1))
        self.mixes = nn.Parameter(
            torch.zeros(1, n_events, 2).uniform_(-1, 1))
        self.amplitudes = nn.Parameter(
            torch.zeros(1, n_events, 1).uniform_(0, 0.1))
        

        # dictionaries
        self.r = SampleLookup(n_resonances, n_samples, flatten_kernel_size=512)
        self.e = Envelopes(n_envelopes, n_frames, n_samples)
        self.d = Decays(n_decays, n_frames, n_samples)
        self.warp = Deformations(n_deformations, instr_expressivity, n_frames, n_samples)
        
        # self.scheduler = DiracScheduler(
        #     self.n_events, start_size=1024, n_samples=self.n_samples)
        
        self.scheduler = FFTShiftScheduler(self.n_events)
    
    def forward(self):
        
        # Begin layer ==========================================
        
        # calculate impulses or energy injected into a system
        impulses = self.e.forward(self.envelopes)
        
        # choose a number of resonances to be convolved with
        # those impulses
        resonance = self.r.forward(self.resonances)
        
        # describe how we interpolate between different
        # resonances over time
        deformations = self.warp.forward(self.deformations)
        
        # determine how each resonance decays or leaks energy
        decays = self.d.forward(self.decays)
        decaying_resonance = resonance * decays[:, :, None, :]

        dry = impulses[:, :, None, :]
        
        # convolve the impulse with all the resonances and
        # interpolate between them
        conv = fft_convolve(dry, decaying_resonance)
        with_deformations = conv * deformations
        audio_events = torch.sum(with_deformations, dim=2, keepdim=True)
        
        # mix the dry and wet signals
        mixes = self.mixes[:, :, None, None, :]
        mixes = torch.softmax(mixes, dim=-1)
        
        stacked = torch.stack([dry, audio_events], dim=-1)
        mixed = stacked * mixes
        final = torch.sum(mixed, dim=-1)
        
        final = final.view(-1, self.n_events, self.n_samples)      
        final = final * torch.abs(self.amplitudes)  
        
        # End layer ==========================================
        
        # TODO: This is the final step, after all layers are done 
        # processing, which is _positioning_ the events
        # pos = sparse_softmax(self.pos, normalize=True, dim=-1)
        # pos = upsample_with_holes(pos, desired_size=self.n_samples)
        
        # final = fft_convolve(final, pos)
        final = self.scheduler.forward(final)
        
        return final


def transform(audio: torch.Tensor) -> torch.Tensor:
    return stft(audio, ws=2048, step=256, pad=True)


def train(target: torch.Tensor):
    model = OverfitResonanceModel(
        instr_expressivity=8,
        n_events=n_events,
        n_resonances=16,
        n_envelopes=16,
        n_decays=16,
        n_deformations=16,
        n_samples=n_samples,
        n_frames=n_frames,
        samplerate=samplerate
    ).to(device)
    optim = Adam(model.parameters(), lr=1e-3)
    
    for iteration in count():
        optim.zero_grad()
        recon = model.forward()
        loss = iterative_loss(target, recon, transform)
        loss.backward()
        optim.step()
        print(iteration, loss.item())
        

if __name__ == '__main__':
    ai = AudioIterator(
        batch_size=1, 
        n_samples=n_samples, 
        samplerate=samplerate, 
        normalize=True, 
        overfit=True,)
    target: torch.Tensor = next(iter(ai)).to(device).view(-1, 1, n_samples)
    train(target)
    
    
    
    