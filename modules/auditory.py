from typing import Literal
from torch import nn
import torch
from modules import stft, unit_norm
from modules.fft import n_fft_coeffs
import numpy as np
from scipy.signal import gammatone

# TODO: multi-resolution STFT, multi-band STFT, Gammatone filter bank, PIF

class STFT(nn.Module):
    def __init__(self, window_size, step_size):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size
    
    @property
    def n_coeffs(self):
        return n_fft_coeffs(self.window_size)


    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        batch, channels, time = audio.shape
        
        spec = stft(
            audio, ws=self.window_size, step=self.step_size, pad=True)
        
        spec = spec\
            .view(batch, channels, -1, self.n_coeffs)\
            .permute(0, 1, 3, 2)
        
        return spec


FrequencySpacingType = Literal['linear', 'geometric']

def gammatone_filter_bank(
        n_filters: int, 
        size: int, 
        min_freq_hz: int, 
        max_freq_hz: int, 
        samplerate: int,
        freq_spacing_type: FrequencySpacingType) -> np.ndarray:
    
    bank = np.zeros((n_filters, size))
    
    if freq_spacing_type == 'linear':
        frequencies = np.linspace(
            min_freq_hz, 
            max_freq_hz, 
            num=n_filters)
    else:
        frequencies = np.geomspace(
            min_freq_hz, 
            max_freq_hz, 
            num=n_filters)
    
    for i, freq in enumerate(frequencies):
        b, a = gammatone(
            freq=freq, 
            ftype='fir', 
            order=4, 
            numtaps=size, 
            fs=samplerate)
        
        bank[i] = b
    
    bank = bank / np.abs(bank).max(axis=-1, keepdims=True)
    return bank