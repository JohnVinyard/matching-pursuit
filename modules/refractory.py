'''
total_size = 32

    ret = np.linspace(total_size, 0, total_size - 1) ** 10
    sig = np.ones(total_size) * total_size
    sig[1:] = -ret
'''

import torch

def make_refractory_filter(size: int, power=10, device=None, channels=1):

    filt = torch.ones(size, device=device)
    ret = torch.linspace(1, 0, size - 1, device=device) ** power
    filt[1:] = -ret
    return filt.view(1, 1, size).repeat(1, channels, 1)