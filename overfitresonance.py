from typing import List
import torch
from torch import nn
from data.audioiter import AudioIterator
from modules import stft
from modules.auditory import gammatone_filter_bank
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve, n_fft_coeffs
from modules.instrument import InstrumentStack
from conjure import audio_conjure, serve_conjure, LmdbCollection, bytes_conjure, SupportedContentType, numpy_conjure
from torch.optim import Adam
from modules.normalization import max_norm
from modules.upsample import interpolate_last_axis
from util import device
from io import BytesIO
from soundfile import SoundFile
from torch.nn import functional as F
from matplotlib import pyplot as plt


collection = LmdbCollection(path='overfitresonance')

samplerate = 22050
n_samples = 2 ** 17
n_events = 32

"""
TODO: lookup/select baseclass

lookup for resonance
lookup for envelope
lookup for deformations
"""

# class ResonanceLookup(nn.Module):
#     def __init__(self, n_items: int, n_samples: int):
#         super().__init__()
#         self.n_items = n_items
#         self.n_samples = n_samples
#         self.items = nn.Parameter(torch.zeros(n_items, n_samples).uniform_(-1, 1))

def flatten_envelope(x: torch.Tensor, kernel_size: int, step_size: int):
    env = torch.abs(x)
    
    normalized = x / (env.max(dim=-1)[0] + 1e-3)
    env = F.max_pool1d(
        env, 
        kernel_size=kernel_size, 
        stride=step_size, 
        padding=step_size)
    env = 1 / env
    env = interpolate_last_axis(env, desired_size=x.shape[-1])
    result = normalized * env
    return result

if __name__ == '__main__':
    ai = AudioIterator(
        batch_size=1, 
        n_samples=n_samples, 
        samplerate=samplerate, 
        normalize=True, 
        overfit=True,)
    example: torch.Tensor = next(iter(ai))
    
    
    decay_values = torch.zeros(3, 4, 16).uniform_(0, 1)
    decays = decay(decay_values, 128)
    print(decays.shape)
    
    for i in range(16):
        v = decay_values[0, 0, i].item()
        print(v)
        plt.plot(decays[0, 0, i].data.cpu().numpy())
        plt.show()
    
    