import torch
from torch import Tensor

from modules.conv import fft_convolve
from modules.normalization import unit_norm
from modules.pos_encode import pos_encode_feature
from modules.softmax import hard_softmax
from modules.sparse import soft_dirac, sparsify, sparsify_vectors
from collections import defaultdict
from torch.nn import functional as F
from util import device
import numpy as np
from scipy.fft import rfft, irfft
from torch.nn import functional as F
from torch import nn




def build_scatter_segments(n_samples, atom_size):

    def scatter_segments(x, inst):

        channels = 1

        if isinstance(x, tuple):
            # x is the desired _size_ out the target signal
            x = torch.zeros(*x, device=device, requires_grad=True)
            channels = x.shape[1]

        # pad the signal at the beginning and end to avoid any boundary
        # issues    
        target = torch.cat(
            [torch.zeros_like(x), x, torch.zeros_like(x)], dim=-1)
    
        base = n_samples

        counter = defaultdict(int)

        for ai, j, p, a in inst:
            p = int(p)
            start = base + p
            end = start + atom_size

            ch = counter.get(j, 0)

            if channels == 1:
                target[j, :, start: end] += a.view(-1, atom_size)
            else:
                target[j, ch, start: end] = a.view(-1, atom_size)
            
            counter[j] = ch + 1
            

        # remove all the padding
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
        pooling=None,
        return_residual=False):

    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    d = unit_norm(d, dim=-1)

    residual = signal.clone()

    fm = torch.zeros(
        signal.shape[0], d.shape[0], signal.shape[-1], device=device)
    
    for i in range(n_steps):

        if approx is None:
            padded = F.pad(residual, (0, atom_size))
            f = F.conv1d(
                padded, d.view(n_atoms, 1, atom_size))[..., :n_samples]
            # attn = torch.norm(f, dim=1)
            # f, _ = sparsify_vectors(f, attn, n_to_keep=n_steps)
            # return f
        else:
            f = fft_convolve(residual, d, approx=approx)


        hard = soft_dirac(f.reshape(batch, -1)).reshape(fm.shape)
        fm = fm + (hard * f)

        values, indices = torch.max(f.reshape(batch, -1), dim=-1)
        
        atom_indices = indices // n_samples
        positions = indices % n_samples

        for b in range(batch):
            v = values[b]
            ai = atom_indices[b]
            p = positions[b]
            # fm[b, ai, p] += v

            # subtract the atom from the residual
            start = p
            end = start + atom_size

            slce = residual[b, :, start:end]
            size = slce.shape[-1]
            slce[:] = slce[:] - (d[ai, :size] * v)

    if return_residual:
        return fm, residual

    return fm


def sparse_coding_loss(
        recon, target, d: Tensor, n_steps=100, device=None, approx=None, pooling=None):

    r_map = sparse_feature_map(
        recon, d, n_steps, device=device, pooling=pooling)

    with torch.no_grad():
        t_map = sparse_feature_map(
            target, d, n_steps, device=device, pooling=pooling)

    r_max = r_map.max()
    t_max = t_map.max()

    mx = max(r_max.item(), t_max.item())

    r_map = r_map / mx
    t_map = t_map / mx

    return F.binary_cross_entropy(r_map, t_map)


def sparse_code_to_differentiable_key_points(
        signal: Tensor,
        d: Tensor,
        n_steps=100,
        device=None):
    
    # print('-------------------')

    signal = signal.view(signal.shape[0], 1, -1)
    batch, _, n_samples = signal.shape
    n_atoms, atom_size = d.shape
    half_atom = atom_size // 2
    d = unit_norm(d, dim=-1)
    residual = signal.clone()

    scatter_segments = build_scatter_segments(n_samples, atom_size)

    vecs = []

    for i in range(n_steps):

        padded = F.pad(residual, (0, atom_size))
        fm = F.conv1d(padded, d.view(
            n_atoms, 1, atom_size))[..., :n_samples]

        fm = fm.reshape(batch, -1)
        value, mx = torch.max(fm, dim=-1)

        atom_index = mx // n_samples  # (Batch,)
        position = mx % n_samples  # (Batch,)
        at = d[atom_index] * value[:, None]  # (Batch,)

        local_instances = []

        for j in range(batch):
            ai = atom_index[j].item()

            local_fm = fm[j].view(n_atoms, n_samples)

            pos = position[j]
            # vec = local_fm[:, pos]

            # time = soft_dirac(local_fm[ai, :]) @ torch.linspace(0, 1, n_samples, device=d.device, requires_grad=True)
            time = soft_dirac(local_fm.max(dim=0)[0]) @ torch.linspace(0, 1, n_samples, device=d.device, requires_grad=True)
            # time = local_fm[ai, :] @ torch.linspace(0, 1, n_samples, device=d.device, requires_grad=True)
            val = value[j]

            # output the element-wise multiplication of the atom and
            # the signal at that point
            # atom = d[ai]
            rez = residual[j, 0, pos - half_atom: pos + half_atom].clone()
            vec = rez # atom[:rez.shape[0]].clone() * rez
            diff = atom_size - vec.shape[0]
            if diff > 0:
                vec = torch.cat([vec, torch.zeros(diff, device=vec.device)])

            # vec = vec.view(n_atoms)
            # vec = torch.softmax(vec.view(n_atoms), dim=-1)
            # vec = soft_dirac(vec.view(n_atoms))
            # print('VEC MAX', torch.argmax(vec, dim=-1).item())

            x = torch.cat([
                val.view(1), 
                time.view(1) * 100,
                # pos_encode_feature(time.view(1, 1) * np.pi, np.pi, None, 16).view(33), 
                # time.view(1) - 50,
                vec.view(n_atoms)
            ])
            vecs.append(x[None, :])

            p = position[j][None, ...]
            a = at[j][None, ...]
            local_instances.append((ai, j, p, a))

        sparse = scatter_segments(residual.shape, local_instances)
        residual = residual - sparse

    vecs = torch.cat(vecs, dim=0)
    return vecs, torch.norm(residual, dim=-1)

def sparse_code(
        signal: Tensor,
        d: Tensor,
        n_steps=100,
        device=None,
        approx=None,
        flatten=False,
        extract_atom_embedding=None,
        visit_key_point=None,
        return_residual=False,
        local_contrast_norm=False,
        return_sparse_feature_map=False,
        compute_feature_map=None,
        fft_convolution=False):
    
    batch, channels, time = signal.shape

    # TODO: It should be possible to accept signals of different dimension
    signal = signal.view(signal.shape[0], channels, -1)

    batch, _, n_samples = signal.shape
    # n_atoms, atom_size = d.shape
    n_atoms = d.shape[0]
    atom_size = d.shape[-1]

    d = unit_norm(d, dim=-1)

    residual = signal.clone()

    # TODO: scatter segments is hard-coded for 1D signals
    scatter_segments = build_scatter_segments(n_samples, atom_size)

    instances = defaultdict(list)

    if extract_atom_embedding is not None:
        embeddings = []

    if return_sparse_feature_map:
        sfm = torch.zeros(batch, n_atoms, n_samples, device=signal.device)

    for i in range(n_steps):
        
        # print('sparse coding step', i)
        if compute_feature_map is not None:
            fm = compute_feature_map(residual, d)
        elif approx is None:
            padded = F.pad(residual, (0, atom_size))
            fm = F.conv1d(padded, d.view(
                n_atoms, channels, atom_size))[..., :n_samples]
        else:
            # TODO: compare performance of fft implementation and torch
            fm = fft_convolve(residual, d, approx=approx)
        
        if extract_atom_embedding is not None:
            embeddings.append(extract_atom_embedding(fm, d))
        

        if local_contrast_norm:
            # TODO: There might be something to local contrast norm
            feature_map = fm.view(batch, 1, n_atoms, n_samples)
            averages = F.avg_pool2d(feature_map, (9, 9), (1, 1), (4, 4))
            feature_map = feature_map - averages
            feature_map = feature_map.reshape(batch, -1)
            fm = fm.reshape(batch, -1)
            # get the best indices from the normalized feature map
            value, mx = torch.max(feature_map, dim=-1, keepdim=True)
            # get the values from the original feature map
            value = torch.gather(fm, dim=-1, index=mx)
        else:
            fm = fm.reshape(batch, -1)
            value, mx = torch.max(fm, dim=-1, keepdim=True)


        atom_index = mx // n_samples  # (Batch,)
        position = mx % n_samples  # (Batch,)
        if channels == 1:
            at = d[atom_index] * value[:, None]  # (Batch,)
        else:
            at = d[atom_index] * value[:, None, None]

        local_instances = []

        for j in range(batch):
            
            ai = atom_index[j].item()
            p = position[j][None, ...]
            a = at[j][None, ...]

            if return_sparse_feature_map:
                sfm[j, ai, p] += value[j]
            
            local_instances.append((ai, j, p, a))
            instances[ai].append((ai, j, p, a))

            if visit_key_point is not None:
                visit_key_point(fm[j].view(n_atoms, n_samples), ai, p.view(1), a.view(atom_size))

        sparse = scatter_segments(residual.shape, local_instances)

        residual -= sparse

        

    if extract_atom_embedding is not None:
        return embeddings, residual

    if not flatten:
        return instances, scatter_segments
    elif flatten and return_residual:
        flattened = flatten_atom_dict(instances) 
        return flattened, scatter_segments, residual
    elif flatten and return_sparse_feature_map:
        flattened = flatten_atom_dict(instances) 
        return flattened, scatter_segments, sfm
    else:
        flattened = flatten_atom_dict(instances) 
        return flattened, scatter_segments


def dictionary_learning_step(
        signal: Tensor,
        d: Tensor,
        n_steps: int = 100,
        device=None,
        approx=None,
        local_constrast_norm: bool = False,
        compute_feature_map=None,
        fft_convolution=False):

    batch, channels, time = signal.shape
    signal = signal.view(signal.shape[0], channels, -1)
    batch, _, n_samples = signal.shape
    # n_atoms, atom_size = d.shape
    n_atoms = d.shape[0]
    atom_size = d.shape[-1]

    d = unit_norm(d, dim=-1)

    residual = signal.clone()

    def gather_segments(x, inst):
        source = torch.cat(
            [torch.zeros_like(x), x, torch.zeros_like(x)], dim=-1)
        segments = []
        base = n_samples
        for ai, j, p, a in inst:
            seg = source[j, :, base + p: base + p + atom_size]
            segments.append(seg[None, ...])
        segments = torch.cat(segments, dim=0)
        return segments

    instances, scatter_segments = sparse_code(
        signal, 
        d, 
        n_steps=n_steps, 
        device=device, 
        approx=approx, 
        local_contrast_norm=local_constrast_norm, 
        compute_feature_map=compute_feature_map,
        fft_convolution=fft_convolution)


    for index in instances.keys():
        inst = instances[index]

        # add atoms back to residual
        sparse = scatter_segments(residual.shape, inst)
        residual += sparse

        # take the average of all positions to compute the new atom

        new_segments = gather_segments(residual, inst)
        new_atom = torch.sum(new_segments, dim=0)

        new_atom = unit_norm(
            new_atom.view(-1)).view(channels, atom_size)
        
        d[index] = new_atom

        updated = map(
            lambda x: (x[0], x[1], x[2], new_atom *
                       torch.norm(x[3], dim=-1, keepdim=True)),
            inst
        )

        sparse = scatter_segments(residual.shape, updated)
        residual = residual - sparse

    d = unit_norm(d, dim=-1)

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
