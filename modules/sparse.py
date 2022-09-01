import torch
from torch import jit
from torch import nn


def sparsify(x, n_to_keep):
    orig_shape = x.shape
    x = x.reshape(x.shape[0], -1)
    values, indices = torch.topk(x, n_to_keep, dim=-1)
    out = torch.zeros_like(x)
    out = torch.scatter(out, dim=-1, index=indices, src=values)
    out = out.reshape(*orig_shape)
    return out


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
    def __init__(self, n_samples, n_atoms, atom_size, to_keep):
        super().__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.to_keep = to_keep

    @jit.script_method
    def forward(self, features, atoms):
        batch = features.shape[0]
        features = features.view(-1, self.n_atoms, self.n_samples)
        atoms = atoms.view(batch, self.n_atoms, self.atom_size)

        features = features.view(batch, -1)
        values, indices = torch.topk(features, self.to_keep, dim=-1)

        output = torch.zeros(
            batch, 1, self.n_samples + self.atom_size).to(features.device)

        for b in range(batch):
            for j in range(self.to_keep):
                index = indices[b, j]

                atom_index = index // self.n_samples
                sample_index = index % self.n_samples

                atom = atoms[b, atom_index]
                factor = values[b, j]

                start = sample_index
                stop = start + self.atom_size

                output[b, 0, start: stop] += atom * factor

        return output[..., :self.n_samples]


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
