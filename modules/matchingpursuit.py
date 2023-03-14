import torch
from torch import Tensor
from modules.fft import fft_convolve

from modules.normalization import unit_norm
from modules.sparse import sparsify
from collections import defaultdict
from torch.nn import functional as F


def sparse_code(signal: Tensor, d: Tensor, n_steps=100, device=None):
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

        padded = F.pad(residual, (0, atom_size))
        fm = F.conv1d(padded, d.view(n_atoms, 1, atom_size))[..., :n_samples]

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
    
    print(torch.norm(residual).item())

    return instances, scatter_segments


def dictionary_learning_step(signal: Tensor, d: Tensor, n_steps: int = 100, device=None):
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


    instances, scatter_segments = sparse_code(signal, d, n_steps=n_steps, device=device)

    
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