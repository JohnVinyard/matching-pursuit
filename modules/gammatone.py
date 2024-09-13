from typing import Literal

import torch
import numpy as np
from scipy.signal import gammatone

BandSpacing = Literal['linear', 'geometric']

def gammatone_filter_bank(
        n_filters: int,
        size: int, device,
        start_hz: int = 20,
        stop_hz: int = 11000,
        samplerate: int = 22050,
        band_spacing: BandSpacing = 'linear') -> torch.Tensor:


    if band_spacing == 'linear':
        frequencies = np.linspace(
            start_hz,
            stop_hz,
            num=n_filters)
    elif band_spacing == 'geometric':
        frequencies = np.geomspace(
            start_hz, stop_hz, num=n_filters)
    else:
        raise ValueError(
            f'{band_spacing} is not a valid band_spacing value, please choose linear or geometric')

    bank = np.zeros((n_filters, size))

    for i, freq in enumerate(frequencies):
        b, a = gammatone(
            freq=freq,
            ftype='fir',
            order=4,
            numtaps=size,
            fs=samplerate)

        bank[i] = b

    bank = bank / np.abs(bank).max(axis=-1, keepdims=True)
    bank = torch.from_numpy(bank).to(device).float()
    return bank
