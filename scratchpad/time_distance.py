from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from scipy.signal import morlet
from util import device
import zounds
import numpy as np
from torch.optim import Adam


samplerate = zounds.SR22050()
n_samples = 2 ** 15
atom_size = 128


def optimizer(model, lr=1e-4, betas=(0, 0.9), weight_decay=0):
    return Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)


def fft_shift(a, shift):
    n_samples = a.shape[-1]
    shift_samples = shift * n_samples * (1/3)
    a = F.pad(a, (0, n_samples * 2))
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device)
             * 2j * np.pi) / n_coeffs

    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    samples = samples[..., :n_samples]
    return samples


def morlet_filter_bank(
        samplerate,
        kernel_size,
        scale,
        scaling_factor,
        normalize=True):

    basis_size = len(scale)
    basis = np.zeros((basis_size, kernel_size), dtype=np.complex128)

    try:
        if len(scaling_factor) != len(scale):
            raise ValueError('scaling factor must have same length as scale')
    except TypeError:
        scaling_factor = np.repeat(float(scaling_factor), len(scale))

    sr = int(samplerate)

    for i, band in enumerate(scale):
        scaling = scaling_factor[i]
        w = band.center_frequency / (scaling * 2 * sr / kernel_size)
        basis[i] = morlet(
            M=kernel_size,
            w=w,
            s=scaling)

    if normalize:
        basis /= np.linalg.norm(basis, axis=-1, keepdims=True) + 1e-8

    return basis.astype(np.complex64)


class Model(nn.Module):
    def __init__(self, atom, n_samples):
        super().__init__()
        self.register_buffer('atom', F.pad(atom, (0, n_samples - atom.shape[-1])))
        self.shift = nn.Parameter(torch.zeros(1).fill_(0.5))

    def forward(self):
        x = fft_shift(self.atom, self.shift)
        return x, self.shift


def fft_convolve(*args):
    n_samples = args[0].shape[-1]

    # pad to avoid wraparound artifacts
    padded = [F.pad(x, (0, x.shape[-1])) for x in args]
    
    specs = [torch.fft.rfft(x, dim=-1, norm='ortho') for x in padded]
    spec = reduce(lambda accum, current: accum * current, specs[1:], specs[0])
    final = torch.fft.irfft(spec, dim=-1, norm='ortho')

    # remove padding
    return final[..., :n_samples]


if __name__ == '__main__':
    scale = zounds.LinearScale(zounds.FrequencyBand(20, 2000), 128)
    bank = morlet_filter_bank(
        samplerate, atom_size, scale, 0.1, normalize=True)

    atom = torch.from_numpy(bank[20].real).float().view(1, 1, atom_size).to(device)

    model = Model(atom, n_samples).to(device)
    optim = optimizer(model, lr=1e-3)

    actual_shift = torch.zeros(1).fill_(0.9).to(device)
    target = F.pad(atom, (0, n_samples - atom_size))
    target = fft_shift(target, actual_shift)

    while True:
        print('=====================================')
        optim.zero_grad()
        recon, shift = model.forward()

        with torch.no_grad():
            best_fit = F.conv1d(target, model.atom, stride=1, padding=n_samples // 2)[..., :n_samples]
            best_fit = torch.softmax(best_fit, dim=-1)
            values = torch.linspace(0, 1, n_samples, device=best_fit.device)
            bf = best_fit @ values
            print('BF', bf.item())
    
        loss = F.mse_loss(recon, target)
        loss.backward()
        optim.step()
        print(loss.item(), shift.item())
