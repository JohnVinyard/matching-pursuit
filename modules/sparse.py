import torch
from torch import jit
from torch import nn

from modules.normalization import unit_norm


def sparsify(x, n_to_keep):
    orig_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    values, indices = torch.topk(x, n_to_keep, dim=-1)
    out = torch.zeros_like(x)
    out = torch.scatter(out, dim=-1, index=indices, src=values)
    out = out.reshape(*orig_shape)
    return out


def to_sparse_vectors_with_context(x, n_elements):
    batch, channels, time = x.shape


    one_hot = torch.zeros(batch, n_elements, channels, device=x.device)
    indices = torch.nonzero(x)

    # time positions at which non-zero elements occur
    positions = [idx[-1] for idx in indices]

    for i in range(indices.shape[0]):
        batch = i // n_elements
        el = i % n_elements
        index = indices[i]
        val = x[tuple(index)]
        one_hot[batch, el, index[1]] = val

    context = torch.sum(x, dim=-1, keepdim=True).repeat(1, 1, n_elements).permute(0, 2, 1)

    return one_hot, context, positions


def sparsify_vectors(x, attn, n_to_keep, normalize=True, dense=False):
    batch, channels, time = x.shape

    attn = attn.view(batch, time)
    values, indices = torch.topk(attn, k=n_to_keep, dim=-1)

    if normalize:
        values = values + (1 - values)

    if dense:
        output = torch.zeros_like(x)

    latents = []
    for b in range(batch):
        for i in range(n_to_keep):
            latent = x[b, :, indices[b, i]][None, :]
            v = values[b, i]
            if dense:
                output[b, indices[b, i]] = latent
            else:
                latents.append(latent * v.view(1, 1, 1))

    if dense:
        return output
    else:
        latents = torch.cat(latents, dim=0).view(batch, n_to_keep, channels)
        return latents, indices


class AtomPlacement(jit.ScriptModule):
    def __init__(self, n_samples, n_events, step_size):
        super().__init__()
        self.n_samples = n_samples
        self.n_events = n_events
        self.step_size = step_size
        

    @jit.script_method
    def render(self, x, indices):
        x = x.view(-1, self.n_events, self.n_samples)
        batch = x.shape[0]

        times = indices * self.step_size

        output = torch.zeros(batch, 1, self.n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(self.n_events):
                time = times[b, i]
                output[b, :, time: time + self.n_samples] += x[b, i][None, :]

        output = output[..., :self.n_samples]
        return output


class SparseAudioModel(nn.Module):
    def __init__(self, n_samples, n_atoms, atom_size, to_keep):
        super().__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.to_keep = to_keep

        self.atoms = nn.Parameter(
            torch.zeros(1, n_atoms, atom_size).uniform_(-0.01, 0.01))
        self.placement = AtomPlacement(
            n_samples, n_atoms, atom_size, to_keep)

    def forward(self, x):
        a = self.atoms.repeat(x.shape[0], 1, 1)
        x = self.placement.forward(x, a)
        return x


class ElementwiseSparsity(nn.Module):
    def __init__(self, model_dim, high_dim=2048, keep=64, dropout=None, softmax=False):
        super().__init__()
        self.expand = nn.Conv1d(model_dim, high_dim, 1, 1, 0)
        self.contract = nn.Conv1d(high_dim, model_dim, 1, 1, 0)
        self.keep = keep
        self.dropout = dropout
        self.softmax = softmax

    def forward(self, x):
        if self.dropout is not None:
            x = torch.dropout(x, self.dropout, self.training)
        
        x = self.expand(x)
        
        if self.softmax:
            x = torch.softmax(x, dim=1)
        
        sparse = sparsify(x, self.keep)
        x = self.contract(sparse)
        return x, sparse


class VectorwiseSparsity(nn.Module):
    def __init__(self, model_dim, keep=16, channels_last=True, dense=True, normalize=False, time=128):
        super().__init__()
        self.channels_last = channels_last
        self.attn = nn.Linear(model_dim, 1)
        self.keep = keep
        self.dense = dense
        self.normalize = normalize

    def forward(self, x):
        if self.channels_last:
            x = x.permute(0, 2, 1)

        batch, channels, time = x.shape

        x = x.permute(0, 2, 1)

        attn = self.attn(x)
        attn = attn.view(batch, time)
        # attn = torch.softmax(attn, dim=1)

        x = sparsify_vectors(
            x.permute(0, 2, 1), attn, n_to_keep=self.keep, dense=self.dense, normalize=self.normalize)
        
        if not self.dense:
            return x

        if self.channels_last:
            x = x.permute(0, 2, 1)

        return x
