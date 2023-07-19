
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.linear import LinearOutputStack
from modules.matchingpursuit import sparse_code
from modules.normalization import unit_norm
from modules.reverb import ReverbGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


# PIF will be (batch, bands, time, periodicity)

features_per_band = 8
n_atoms = 512
atom_shape = (3, 3, 3)
padding = tuple(x // 2 for x in atom_shape)
sparse_coding_iterations = 128

d = torch.zeros(n_atoms, *atom_shape, device=device).uniform_(-1, 1)
atom_dict = unit_norm(d.view(n_atoms, -1), dim=-1).reshape(n_atoms, *atom_shape)



def to_original_dim(flat_indices, shape):
    fi = flat_indices
    sh = (list(shape) + [1])
    results = []
    for i in range(len(sh) - 1, 0, -1):

        raw_sh = sh[i:]
        prd = np.product(raw_sh)
        mod = sh[i - 1]

        # print(f'(indices // {"*".join([str(x) for x in raw_sh])}) % {mod}')
        nxt = (fi // prd) % mod
        results.append(nxt)

    return results[::-1]


def best_fit(signal, atom_dict):

    # prepare shapes and whatnot
    orig_shape = signal.shape
    batch, bands, time, periodicity = orig_shape
    signal = signal.view(batch, 1, bands, time, periodicity)
    atom_dict = atom_dict.view(n_atoms, 1, *atom_shape)

    
    # convolve dictionary with signal
    signal = F.pad(signal, (0, 2, 0, 2, 0, 2))
    sparse_signal = torch.zeros_like(signal)

    fm = F.conv3d(signal, atom_dict, stride=1)

    # find the best spot in the feature map
    flat = fm.reshape(batch, -1)
    values, indices = torch.max(flat, dim=-1)
    

    for i in range(batch):
        f = fm[i]
        index = indices[i]
        val = values[i]
        unflat = to_original_dim(index, f.shape)

        atom_index, b, c, d = unflat
        x, y, z = atom_shape

        atom = atom_dict[atom_index, 0]
        atom = atom * val

        sig = signal[i, 0]
        sparse = sparse_signal[i, 0]

        b_stop = min(b + x, sig.shape[0])
        b_size = b_stop - b

        c_stop = min(c + y, sig.shape[1])
        c_size = c_stop - c

        d_stop = min(d + z, sig.shape[2])
        d_size = d_stop - d

        sig[b: b_stop, c: c_stop, d: d_stop] -= atom[:b_size, :c_size, :d_size]
        sparse[b: b_stop, c: c_stop, d: d_stop] += atom[:b_size, :c_size, :d_size]

    # KLUDGE: don't hard-code this
    return \
        signal[:, :, :128, :128, :8].view(orig_shape), \
        sparse_signal[:, :, :128, :128, :8].view(orig_shape)


def sparse_code(signal, d):
    """
    After a single round of sparse coding, for each atom:
        - add the atom back to the appropriate location in each signal
        - make the atom the average of all these positions
        - subtract the new atom (with the appropriate norm) from the signal
    """
    residual = signal.clone()

    sparse = torch.zeros_like(signal)

    for i in range(sparse_coding_iterations):
        residual, sp = best_fit(residual, atom_dict)
        # print(torch.norm(residual))
        sparse = sparse + sp
    
    return sparse
    

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode='nearest')



class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(257, features_per_band)


        self.to_verb_context = LinearOutputStack(1024, 3, out_channels=1024, norm=nn.LayerNorm((1024,)))

        channels = 1024

        self.up = nn.Sequential(

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            UpsampleBlock(512),

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            UpsampleBlock(256),

            nn.Conv1d(256, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            UpsampleBlock(128),

            nn.Conv1d(128, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            UpsampleBlock(64),

            nn.Conv1d(64, 1, 7, 1, 3),
        )        

        self.verb = ReverbGenerator(1024, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm(1024,))

        self.apply(lambda x: exp.init_weights(x))
    
    def embed_features(self, x, iteration):

        # torch.Size([16, 128, 128, 257])
        batch, channels, time, period = x.shape
        x = self.embed(x) # (batch, channels, time, 8)

        sparse_iter = 100


        if iteration == sparse_iter:
            print('initializing atoms')
            # randomly initialize the dictionary on the 100th iteration
            for i in range(n_atoms):
                b = np.random.randint(0, batch)
                q = np.random.randint(0, 128 - 3)
                y = np.random.randint(0, 128 - 3)
                z = np.random.randint(0, 8 - 3)
                new_atom = x[b, q: q + 3, y: y + 3, z: z + 3]
                atom_dict[i] = unit_norm(new_atom.reshape(-1)).reshape(3, 3, 3)
        elif iteration > sparse_iter:
            with torch.no_grad():
                print('sparse coding')
                # start using the dictionary to sparsify the signal
                x = sparse_code(x, atom_dict)


        x = x.permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)
        return x

    def generate(self, x):
        x = self.up(x)
        return x
    
    def forward(self, x, iteration):

        # torch.Size([16, 128, 128, 257])
        encoded = self.embed_features(x, iteration)
        ctx = torch.sum(encoded, dim=-1)
        ctx = self.to_verb_context(ctx)

        x = self.generate(encoded)

        x = self.verb.forward(ctx, x)
        return x

model = Model().to(device)
optim = optimizer(model, lr=1e-3)



def train(batch, i):
    optim.zero_grad()

    with torch.no_grad():
        spec = exp.perceptual_feature(batch)

    recon = model.forward(spec, i)
    recon_spec = exp.perceptual_feature(recon)

    audio_loss = F.mse_loss(recon_spec, spec)
    loss = audio_loss
    loss.backward()
    optim.step()
    return loss, recon



@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    
            

                
    
    