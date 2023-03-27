import torch
from torch import Tensor

from modules.normalization import unit_norm
from modules.sparse import sparsify
from collections import defaultdict
from torch.nn import functional as F
from util import device
import numpy as np
from scipy.fft import rfft, irfft
from torch.nn import functional as F
from torch import nn


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
    atom = torch.fft.rfft(torch.flip(padded, dims=(-1,)), dim=-1)[None, ...]

    if isinstance(approx, slice):
        slce = approx
        fm_spec = torch.zeros(
            batch, n_atoms, sig.shape[-1], device=signal.device, dtype=sig.dtype)
        app = sig[..., slce] * atom[..., slce]
        fm_spec[..., slce] = app
    elif isinstance(approx, int) and approx < n_samples:
        fm_spec = torch.zeros(
            batch, n_atoms, sig.shape[-1], device=signal.device, dtype=sig.dtype)

        # choose local peaks
        mags = torch.abs(sig)
        # window_size = atom_size // 64 + 1
        # padding = window_size // 2
        # avg = F.avg_pool1d(mags, window_size, 1, padding=padding)
        # mags = mags / avg

        values, indices = torch.topk(mags, k=approx, dim=-1)
        sig = torch.gather(sig, dim=-1, index=indices)

        # TODO: How can I use broadcasting rules to avoid this copy?
        atom = torch.gather(atom.repeat(batch, 1, 1), dim=-1, index=indices)
        sparse = sig * atom
        fm_spec = torch.scatter(fm_spec, dim=-1, index=indices, src=sparse)
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


def build_scatter_segments(n_samples, atom_size):

    def scatter_segments(x, inst):
        if isinstance(x, tuple):
            x = torch.zeros(*x, device=device)
        target = torch.cat(
            [torch.zeros_like(x), x, torch.zeros_like(x)], dim=-1)
        base = n_samples

        for ai, j, p, a in inst:
            p = int(p)
            start = base + p
            end = start + atom_size

            try:
                target[j, :, start: end] += a
            except RuntimeError:
                print(
                    f'Out-of-bounds with start {start - base} and end {end - base}')

        return target[..., n_samples: n_samples * 2]

    return scatter_segments


def flatten_atom_dict(atom_dict):
    all_instances = []
    for k, v in atom_dict.items():
        all_instances.extend(v)
    return all_instances


def sparse_feature_map(
        signal: Tensor,
        d: Tensor,
        n_steps=100,
        device=None,
        approx=None,
        pooling=None):

    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    d = unit_norm(d, dim=-1)

    residual = signal.clone()

    fm = torch.zeros(
        signal.shape[0], d.shape[1], signal.shape[-1], device=device)

    for i in range(n_steps):
        if approx is None:
            padded = F.pad(residual, (0, atom_size))
            f = F.conv1d(
                padded, d.view( n_atoms, 1, atom_size))[..., :n_samples]
        else:
            f = fft_convolve(residual, d, approx=approx)

        values, indices = torch.max(f.reshape(batch, -1), dim=-1)

        atom_indices = indices // n_samples
        positions = indices % n_samples

        for b in range(batch):
            v = values[b]
            ai = atom_indices[b]
            p = positions[b]
            fm[b, ai, p] += v

            start = p
            end = start + atom_size
            slce = residual[b, :, start:end]
            size = slce.shape[-1]
            slce[:] = slce[:] - (d[ai, :size] * v)
            
        # # TODO: Just place the atoms, rather than this wasteful transposed conv
        # orig_f = f = sparsify(f, n_to_keep=1, return_indices=False)
        # f = F.pad(f, (0, 1))
        # sparse_signal = F.conv_transpose1d(f, d.view(n_atoms, 1, atom_size), padding=atom_size // 2)

        # residual = residual - sparse_signal

        # fm = fm + orig_f
    
    if pooling is not None:
        window, step = pooling
        fm = F.max_pool1d(fm, window, step, padding=step)
        print('POOLED', fm.shape)

    return fm


def sparse_coding_loss(
        recon, target, d: Tensor, n_steps=100, device=None, approx=None, pooling=None):

    r_map = sparse_feature_map(recon, d, n_steps, device=device, pooling=pooling)

    with torch.no_grad():
        t_map = sparse_feature_map(target, d, n_steps, device=device, pooling=pooling)

    r_max = r_map.max()
    t_max = t_map.max()

    mx = max(r_max.item(), t_max.item())

    r_map = r_map / mx
    t_map = t_map / mx

    return F.binary_cross_entropy(r_map, t_map)


def sparse_code(
        signal: Tensor,
        d: Tensor,
        n_steps=100,
        device=None,
        approx=None,
        flatten=False):

    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    d = unit_norm(d, dim=-1)
    residual = signal.clone()

    scatter_segments = build_scatter_segments(n_samples, atom_size)

    instances = defaultdict(list)

    for i in range(n_steps):

        if approx is None:
            padded = F.pad(residual, (0, atom_size))
            fm = F.conv1d(padded, d.view(
                n_atoms, 1, atom_size))[..., :n_samples]
        else:
            fm = fft_convolve(residual, d, approx=approx)

        fm = fm.reshape(batch, -1)
        value, mx = torch.max(fm, dim=-1)

        atom_index = mx // n_samples  # (Batch,)
        position = mx % n_samples  # (Batch,)
        at = d[atom_index] * value[:, None]  # (Batch,)

        local_instances = []

        for j in range(batch):
            ai = atom_index[j].item()
            p = position[j][None, ...]
            a = at[j][None, ...]
            local_instances.append((ai, j, p, a))
            instances[ai].append((ai, j, p, a))

        sparse = scatter_segments(residual.shape, local_instances)
        residual -= sparse

    # print('SPARSE CODING', torch.norm(residual, dim=-1).mean().item())

    if not flatten:
        return instances, scatter_segments
    else:
        flattened = flatten_atom_dict(instances) 
        return flattened, scatter_segments


def dictionary_learning_step(
        signal: Tensor,
        d: Tensor,
        n_steps: int = 100,
        device=None,
        approx=None):

    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    d = unit_norm(d, dim=-1)
    residual = signal.clone()

    def gather_segments(x, inst):
        source = torch.cat(
            [torch.zeros_like(x), x, torch.zeros_like(x)], dim=-1)
        segments = []
        base = n_samples
        for ai, j, p, a in inst:
            segments.append(source[j, :, base + p: base + p + atom_size])
        segments = torch.cat(segments, dim=0)
        return segments

    instances, scatter_segments = sparse_code(
        signal, d, n_steps=n_steps, device=device, approx=approx)

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

        updated = map(
            lambda x: (x[0], x[1], x[2], new_atom *
                       torch.norm(x[3], dim=-1, keepdim=True)),
            inst
        )

        sparse = scatter_segments(residual.shape, updated)
        residual = residual - sparse

    d = unit_norm(d)

    return d


class SparseCodingLoss(nn.Module):

    def __init__(
            self,
            n_atoms,
            atom_size,
            n_steps,
            approx,
            learning_steps,
            device=None,
            pooling=None):

        super().__init__()
        self.approx = approx
        self.n_steps = n_steps
        self.learning_steps = learning_steps
        self._steps_executed = 0
        self.d = unit_norm(torch.zeros(n_atoms, atom_size, device=device).uniform_(-1, 1))
        self.pooling = pooling

    def _learning_step(self, signal):
        with torch.no_grad():
            self.d[:] = dictionary_learning_step(
                signal,
                self.d,
                n_steps=self.n_steps,
                device=signal.device,
                approx=self.approx)
            self._steps_executed += 1
            print('LEARNING STEP', self._steps_executed)

    def loss(self, recon: torch.Tensor, target: torch.Tensor):
        if self._steps_executed < self.learning_steps:
            self._learning_step(target)

        return sparse_coding_loss(
            recon,
            target,
            self.d,
            n_steps=self.n_steps,
            device=recon.device,
            pooling=self.pooling)
