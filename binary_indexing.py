import numpy as np
import torch
import zounds
from torch import nn
from modules.linear import LinearOutputStack
from modules.phase import morlet_filter_bank
from modules.pos_encode import pos_encoded
from torch.nn import functional as F

from train.optim import optimizer
from util.weight_init import make_initializer

samplerate = zounds.SR22050()

pow = 12

n_samples = 2 ** pow
atom_size = 512

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)

# z = torch.arange(n_samples)
# y = 2 ** torch.arange(1, pow)

# weights = (2 ** torch.arange(1, pow)).float()
# weights /= weights.max()

# z = (z[None, :] // y[:, None]) #% 2
# z = z * weights[:, None]

# positions = z.T
# positions = positions[:n_samples - atom_size].float()
# positions = positions + torch.zeros_like(positions).uniform_(0, 0.01)

positions = pos_encoded(1, n_samples, 16).view(
    n_samples, 33)[:(n_samples - atom_size), :11]
# positions = positions * weights[None, :]

bank = morlet_filter_bank(samplerate, atom_size, scale, 0.1).real
init_weights = make_initializer(0.1)

n_atoms = 8

atom_indices = np.random.permutation(128)[:n_atoms]
atom_positions = np.random.permutation(n_samples - atom_size)[:n_atoms]

atoms = torch.from_numpy(bank[atom_indices])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.indices = nn.Parameter(torch.zeros(
            1, n_atoms, pow - 1).uniform_(0, 1))

    def forward(self, x, add_noise=False, batch_size=1):

        index = self.indices

        if add_noise:
            index = torch.zeros(batch_size, n_atoms, pow - 1).uniform_(0, 1)

        ind_norms = torch.norm(index, dim=-1, keepdim=True)
        pos_norms = torch.norm(positions, dim=-1, keepdim=True)
        ind = index / ind_norms
        pos = positions / pos_norms
        sim = (ind @ pos.T)
        int_index = torch.argmax(sim, dim=-1).view(batch_size, n_atoms)

        output = torch.zeros(batch_size, n_samples)

        for b in range(batch_size):
            for i in range(n_atoms):
                ind = int_index[b, i].view(1)
                atom = atoms[i]
                output[b, ind: ind + atom_size] += atom

        return int_index, output, index


class SyntheticGradient(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = LinearOutputStack(128, 2, in_channels=88)
        self.to_loss = LinearOutputStack(128, 2, out_channels=1)
        self.apply(init_weights)

    def forward(self, orig_signal, indices, signals):
        indices = indices.view(indices.shape[0], -1)
        a = self.embedding(indices).view(indices.shape[0], 128)
        l = self.to_loss(a)
        return l ** 2


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    target = torch.zeros(n_samples)
    for i in range(n_atoms):
        atom = atoms[i]
        pos = atom_positions[i]
        target[pos: pos + atom_size] += atom
    target = target.view(1, n_samples)

    real = target.data.cpu().numpy().squeeze()

    model = Model()
    optim = optimizer(model, lr=1e-4)

    grad = SyntheticGradient()
    grad_optim = optimizer(grad, lr=1e-4)

    i = 0

    while True:
        i += 1

        # train grad
        grad_optim.zero_grad()
        rand_indices, o, soft_index = model.forward(
            None, add_noise=True, batch_size=32)
        real_loss = torch.abs(o - target).sum()
        pred_loss = grad.forward(target, soft_index, atoms)
        grad_loss = torch.abs(real_loss - pred_loss).sum()
        grad_loss.backward()
        grad_optim.step()

        # train model
        optim.zero_grad()
        indices, o, soft_index = model.forward(None)
        fake = o.data.cpu().numpy()
        loss = grad.forward(target, soft_index, atoms)
        loss.backward()
        optim.step()

        if i % 10 == 0:
            print(loss.item(), grad_loss.item())
            print(atom_positions)
            print(indices.data.cpu().numpy())
