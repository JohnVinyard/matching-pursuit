import torch
from torch import Tensor

from modules.normalization import unit_norm
from modules.sparse import sparsify
from collections import defaultdict
from torch.nn import functional as F
from util import device
import numpy as np
from scipy.fft import rfft, irfft


# def batch_fft_convolve1d(signal, d, use_gpu=True):
#     """
#     Convolves the dictionary/filterbank `d` with `signal`

#     Parameters
#     ------------
#     signal - A signal of shape `(batch_size, signal_size)`fft_convolve
#     d - A dictionary/filterbank of shape `(n_components, atom_size)`

#     Returns
#     ------------
#     x - the results of convolving the dictionary with the batched signals
#         of dimensions `(batch, n_components, signal_size)`fft_convofft_convolvelve
#     """
#     signal_size = signal.shape[1]
#     atom_size = d.shape[1]
#     diff = signal_size - atom_size

#     half_width = atom_size // 2

#     # TODO: Is it possible to leverage the zero padding and/or
#     # difference in atom and signal size to avoid multiplying
#     # every frequency position
#     px = np.pad(signal, [(0, 0), (half_width, half_width)])
#     py = np.pad(d, [(0, 0), (0, px.shape[1] - atom_size)])

#     if use_gpu:
#         px = torch.from_numpy(px).to(device).float()
#         py = torch.from_numpy(py).to(device).float()

#         fpx = torch.rfft(px, signal_ndim=1)[:, None, ...]

#         fpy = torch.rfft(
#             torch.flip(py, dims=(-1,)), 
#             signal_ndim=1)[None, ...]

#         # print(fpx.shape, fpy.shape)

#         real = (fpx[..., 0] * fpy[..., 0]) - (fpx[..., 1] * fpy[..., 1])
#         imag = (fpx[..., 0] * fpy[..., 1]) + (fpx[..., 1] * fpy[..., 0])

#         # print(real.shape, imag.shape)
#         # TODO: This is wrong!
#         # c = fpx[:, None, ...] * fpy[None, ...]
#         c = torch.cat([real[..., None], imag[..., None]], dim=-1)

#         # raise Exception()
#         new_size = (c.shape[-2] - 1) * 2
#         c = torch.irfft(c, signal_ndim=1, signal_sizes=(new_size,))
#         c = c.data.cpu().numpy()
#     else:
#         fpx = rfft(px, axis=-1)
#         fpy = rfft(py[..., ::-1], axis=-1)
#         c = fpx[:, None, :] * fpy[None, :, :]
#         c = irfft(c, axis=-1)

#     c = np.roll(c, signal_size - diff, axis=-1)
#     c = c[..., atom_size - 1:-1]

#     return c

def torch_conv(signal, atom):
    n_samples = signal.shape[-1]
    n_atoms, atom_size = atom.shape
    padded = F.pad(signal, (0, atom_size))
    fm = F.conv1d(padded, atom.view(n_atoms, 1, atom_size))[..., :n_samples]
    return fm


def fft_convolve(signal, atoms, approx=None):
    batch = signal.shape[0]
    n_samples = signal.shape[-1]
    n_atoms, atom_size = atoms.shape
    diff = n_samples - atom_size
    half_width = atom_size // 2

    signal = F.pad(signal, (0, atom_size))
    padded = F.pad(atoms, (0, signal.shape[-1] - atom_size))

    sig = torch.fft.rfft(signal, dim=-1)
    atom = torch.fft.rfft(torch.flip(padded, dims=(-1,)), dim=-1)

    if approx is not None:
        size = sig.shape[-1]
        n_bins = int(size * approx)
        fm_spec = torch.zeros(batch, n_atoms, sig.shape[-1], device=signal.device, dtype=sig.dtype)
        app = sig[..., :n_bins] * atom[None, :, :n_bins]
        fm_spec[..., :n_bins] = app
    else:
        fm_spec = sig * atom
    
    fm = torch.fft.irfft(fm_spec, dim=-1)
    fm = torch.roll(fm, 1, dims=(-1,))
    return fm[..., :n_samples]


def compare_conv():
    # n_samples = 16
    # atom_size = 4
    # diff = 12
    # half_width = 2
    signal = torch.zeros(1, 1, 16).normal_(0, 1)
    atoms = torch.zeros(8, 4).normal_(0, 1)
    fm_fft = fft_convolve(signal, atoms)
    fm_torch = torch_conv(signal, atoms)
    return fm_fft, fm_torch


def sparse_code(signal: Tensor, d: Tensor, n_steps=100, device=None, approx=None):
    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    d = unit_norm(d, dim=-1)
    residual = signal

    def scatter_segments(x, inst):
        if isinstance(x, tuple):
            x = torch.zeros(*x, device=device)
        target = torch.cat([torch.zeros_like(x), x, torch.zeros_like(x)], dim=-1)
        base = n_samples
        for ai, j, p, a in inst:
            target[j, :, base + p: base + p + atom_size] += a
        return target[..., n_samples: n_samples * 2]


    instances = defaultdict(list)


    for i in range (n_steps):

        # padded = F.pad(residual, (0, atom_size))
        # fm = F.conv1d(padded, d.view(n_atoms, 1, atom_size))[..., :n_samples]
        fm = fft_convolve(residual, d, approx=approx)

        fm = fm.reshape(batch, -1)
        value, mx = torch.max(fm, dim=-1)

        atom_index = mx // n_samples # (Batch,)
        position = mx % n_samples # (Batch,)
        at = d[atom_index] * value[:, None] # (Batch,)

        local_instances = []

        for j in range(batch):
            ai = atom_index[j].item()
            p = position[j][None, ...]
            a = at[j][None, ...]
            local_instances.append((ai, j, p, a))
            instances[ai].append((ai, j, p, a))

        sparse = scatter_segments(residual.shape, local_instances)
        residual -= sparse
    
    print(torch.norm(residual, dim=-1).mean().item())

    return instances, scatter_segments


def dictionary_learning_step(signal: Tensor, d: Tensor, n_steps: int = 100, device=None, approx=None):
    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    d = unit_norm(d, dim=-1)
    residual = signal

    def gather_segments(x, inst):
        source = torch.cat([torch.zeros_like(x), x, torch.zeros_like(x)], dim=-1)
        segments = []
        base = n_samples
        for ai, j, p, a in inst:
            segments.append(source[j, :, base + p: base + p + atom_size])
        segments = torch.cat(segments, dim=0)
        return segments


    instances, scatter_segments = sparse_code(signal, d, n_steps=n_steps, device=device, approx=approx)

    
    for index in instances.keys():
        inst = instances[index]

        # add atoms back to residual
        sparse = scatter_segments(residual.shape, inst)
        residual += sparse

        # take the average of all positions to compute the new atom
        new_segments = gather_segments(residual, inst)
        new_atom = torch.sum(new_segments, dim=0)
        new_atom = unit_norm(new_atom)    
        d[index] = new_atom

        updated = map(lambda x: (x[0], x[1], x[2], new_atom * torch.norm(x[3], dim=-1, keepdim=True)), inst)

        sparse = scatter_segments(residual.shape, updated)
        residual = residual - sparse    


    d = unit_norm(d)

    return d