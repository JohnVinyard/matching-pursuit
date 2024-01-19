
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.activation import unit_sine
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from modules.normalization import unit_norm
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encode_feature, pos_encoded
from modules.sparse import sparsify, sparsify_vectors
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

# scale = MelScale()
# codec = AudioCodec(scale)

d_size = 512
kernel_size = 512
iterations = 512

local_constrast_norm = True
dict_learning_steps = 50
approx = None


atom_dict = torch.zeros(d_size, kernel_size, device=device).uniform_(-1, 1)
atom_dict = unit_norm(atom_dict)

# TODO: Move this into util somewhere
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


# TODO: extract some generic code for adding/subtracting arbitrary-sized atoms
# at arbitrary dimensions

# class SpectrogramSparseCode(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.n_atoms = 256
#         self.kernel_shape = (5, 15)
#         self.padding = tuple(x // 2 for x in self.kernel_shape)

#         self.time_dim = 128
#         self.freq_dim = 256

#         self._filters = nn.Parameter(torch.zeros(self.n_atoms, 2, *self.kernel_shape).uniform_(-1, 1))
#         self.apply(lambda x: exp.init_weights(x))
    
#     @property
#     def filters(self):
#         f = self._filters.reshape(self.n_atoms, -1)
#         f = unit_norm(f, dim=-1)
#         f = f.reshape(self.n_atoms, 2, *self.kernel_shape)
#         return f
    
#     def forward(self, x, iterations=128):

#         batch, channels, time, freq = x.shape

#         residual = x.clone()
#         recon = torch.zeros_like(residual)

#         print('=====================================')

#         # print('NORMS', torch.norm(self.filters.reshape(self.n_atoms, -1), dim=-1))

#         for i in range(iterations):
#             residual = F.pad(residual, (0, 5, 0, 5))

#             fm = F.conv2d(residual, self.filters, stride=1)
            
#             shape = fm.shape

#             # K-sparse, K=1
#             fm = fm.reshape(batch, -1)
#             values, indices = torch.max(fm, dim=-1)

#             dims = to_original_dim(indices, shape[1:])
#             # print(dims)

#             for j in range(batch):
#                 val = values[j]
#                 index = [x[j] for x in dims]

#                 atom_index, t, f = index

#                 atom = self.filters[atom_index] * val
#                 # print(atom.shape)

#                 t_start = t
#                 t_stop = min(self.time_dim, t_start + self.kernel_shape[0])
#                 t_size = t_stop - t_start

#                 f_start = f
#                 f_stop = min(self.freq_dim, f_start + self.kernel_shape[1])
#                 f_size = f_stop - f_start


#                 next_res = residual.clone()
#                 next_res[j, :, t_start: t_stop, f_start: f_stop] -=  atom[:, :t_size, :f_size]
#                 residual = next_res

#                 next_recon = recon.clone()
#                 next_recon[j, :, t_start: t_stop, f_start: f_stop] += atom[:, :t_size, :f_size]
#                 recon = next_recon

#                 # print(torch.norm(residual).item())
        

#         return recon



def encode_events(events):
    batch_size = max(x[1] for x in events) + 1
    encoding_dim = d_size + 2
    encoded = torch.zeros(batch_size, iterations, encoding_dim, device=device)

    by_batch = defaultdict(list)
    for ai, j, p, a in events:
        by_batch[j].append((ai, p / exp.n_samples, torch.norm(a)))
    
    for i in range(batch_size):

        # ensure that events are ordered 
        # according to ascending time
        ordered = sorted(by_batch[i], key=lambda x: x[1])
        j = 0
        for ai, p, amp in ordered:
            oh = torch.zeros(d_size, device=encoded.device)
            oh[ai] = 1
            amp = torch.norm(a)
            vec = torch.cat([oh, p.view(1), amp.view(1)])
            encoded[i, j, :] = vec
            j += 1

    
    # pos = encoded[:, :, d_size: d_size + 1]
    # pos = torch.diff(pos, n=1, dim=-1, prepend=torch.zeros(batch_size, iterations, 1, device=encoded.device))
    # encoded[:, :, d_size: d_size + 1] = pos
    return encoded


def decode_events(events):

    batch_size, n_events, _ = events.shape

    atoms = events[:, :, :d_size]
    pos = events[:, :, d_size: d_size + 1]
    # pos = torch.cumsum(pos, dim=-1)
    pos = torch.clamp(pos, 0, 1)

    amp = events[:, :, d_size + 1: d_size + 2]


    indices = torch.argmax(atoms, dim=-1)

    atoms = atom_dict[indices.view(-1)].view(batch_size, n_events, kernel_size)
    atoms = atoms * amp
    pos = (pos * exp.n_samples).long()

    events = []

    for i in range(batch_size):
        for j in range(n_events):
            ai = indices[i, j]
            batch = i
            p = pos[i, j]
            at = atoms[i, j]
            events.append((ai, batch, p, at))
    
    return events


class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.embed_atoms = nn.Linear(d_size, self.channels)
        self.embed_points = nn.Linear(2, channels)

        self.down = nn.Linear(channels * 2, channels)

        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 6, norm=nn.LayerNorm((iterations, channels,)))

        self.to_atoms = nn.Linear(channels, d_size)
        self.to_points = nn.Linear(channels, 2)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        atoms = x[:, :, :d_size]
        points = x[:, :, d_size:]

        atoms = self.embed_atoms(atoms)
        points = self.embed_points(points)

        x = torch.cat([atoms, points], dim=-1)
        x = self.down(x)

        x = self.encoder(x)

        atoms = self.to_atoms(x)

        points = self.to_points(x)
        pos = points[:, :, :1]
        pos = torch.sigmoid(pos)

        amp = points[:, :, 1:]
        amp = torch.abs(amp)
        points = torch.cat([pos, amp], dim=-1)

        x = torch.cat([atoms, points], dim=-1)
        return x

model = Model(512).to(device)
optim = optimizer(model, lr=1e-3)


def exp_loss(recon, target):
    recon_indices = recon[:, :, :d_size]
    target_indices = torch.argmax(target[:, :, :d_size], dim=-1)
    atom_loss = F.cross_entropy(recon_indices.view(-1, d_size), target_indices.view(-1))
    recon_points = recon[:, :, d_size:]
    target_points = target[:, :, d_size:]
    point_loss = F.mse_loss(recon_points, target_points)
    return (atom_loss) + point_loss * 100


def train(batch, i):
    batch_size = batch.shape[0]
    optim.zero_grad()

    with torch.no_grad():
        # sparse code signal
        events, scatter = sparse_code(
                batch, 
                atom_dict, 
                n_steps=iterations, 
                device=device, 
                flatten=True, 
                local_contrast_norm=local_constrast_norm,
                approx=approx)
    
        encoded = encode_events(events)

        if i < dict_learning_steps:
            # update sparse dictionary    
            d = dictionary_learning_step(batch, atom_dict, n_steps=iterations, device=device, approx=approx)
            atom_dict.data[:] = d
        
    
    real = scatter(batch.shape, events)

    recon_events = model.forward(encoded)

    events = decode_events(recon_events)
    recon = scatter(batch.shape, events)

    loss = exp_loss(recon_events, encoded)
    loss.backward()
    optim.step()

    return loss, recon.view(batch_size, 1, exp.n_samples), real


@readme
class MatchingPursuitPlayground(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            l, f, r = train(item.view(-1, 1, exp.n_samples), i)
            self.fake = f
            self.real = r
            print(i, l.item())
            self.after_training_iteration(l)
    