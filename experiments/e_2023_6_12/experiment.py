
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.matchingpursuit import dictionary_learning_step, flatten_atom_dict, sparse_code
from modules.pointcloud import encode_events
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
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

sparse_coding_iterations = 64
do_sparse_coding = False
sparse_coding_phase = True

def encode_events(inst, batch_size, n_frames, n_points, device=None):
    """
    TODO: Sort by batch, then time, both ascending

    Then, "pack" into atom indices and 2D points for 
    time and amplitude

    ai, j, p, a 
    """
    srt = sorted(inst, key=lambda x: (x[1], x[2]))
    packed_indices = torch.zeros(batch_size * n_points, dtype=torch.long, device=device)
    packed_points = torch.zeros(batch_size * n_points, 2, dtype=torch.float, device=device)

    for i, tup in enumerate(srt):
        ai, j, p, a = tup
        packed_indices[i] = ai
        amp = torch.norm(a)
        pos = p / n_frames
        packed_points[i, :] = torch.cat([pos.view(1), amp.view(1)])
    
    indices = packed_indices.view(batch_size, n_points)
    points = packed_points.view(batch_size, n_points, 2)

    return indices, points


def decode_events(events: torch.Tensor):
    """
    Convert back to a shape usable by the scatter function returned
    by sparse_code
    """
    pass


class SparseCode(nn.Module):
    def __init__(self, n_atoms, atom_size, channels):
        super().__init__()

        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.channels = channels

        self.d = nn.Parameter(torch.zeros((n_atoms, channels, atom_size)).uniform_(-1, 1))
    
    def learning_step(self, x, n_iterations):
        d = dictionary_learning_step(x, self.d, n_steps=n_iterations, device=x.device, d_normalization_dims=(-1, -2))
        self.d.data[:] = d
    
    def do_sparse_code(self, x, n_iterations):
        batch, channels, frames = x.shape

        inst, scatter = sparse_code(
            x, 
            self.d, 
            n_steps=n_iterations, 
            device=x.device, 
            d_normalization_dims=(-1, -2), 
            flatten=True)

        # indices, points = encode_events(
        #     inst, batch, frames, n_iterations, device=x.device)
        
        # print('=======================')
        # print(indices)
        # print(points)

        recon = scatter(x.shape, inst)
        return recon
    
    def forward(self, x, n_iterations=32):
        return self.sparse_code(x, n_iterations=n_iterations)
    

class UpsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch, channels, time = x.shape
        return F.interpolate(x, scale_factor=4, mode='nearest')

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(257, 8)

        self.sparse = SparseCode(1024, 32, 1024)

        self.up = nn.Sequential(

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(256, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(128, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            UpsampleBlock(),

            nn.Conv1d(64, 1, 7, 1, 3),
        )        
        self.apply(lambda x: exp.init_weights(x))
    
    def embed_features(self, x):
        # torch.Size([16, 128, 128, 257])
        batch, channels, time, period = x.shape
        x = self.embed(x).permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)
        return x

    def generate(self, x):
        x = self.up(x)
        return x
    
    def forward(self, x):
        # torch.Size([16, 128, 128, 257])
        x = self.embed_features(x)

        orig = x

        if do_sparse_coding:
            res, rec = self.sparse.sparse_code(x, n_iterations=32)
        else:
            res, rec = None, orig

        x = self.generate(x)

        return x, res, rec, orig

try:
    model = Model().to(device)
    model.load_state_dict(torch.load('model.dat'))
    print('Loaded model')
except IOError:
    pass
optim = optimizer(model, lr=1e-3)


try:
    sparse_model = SparseCode(4096, 32, 1024).to(device)
    sparse_model.load_state_dict(torch.load('sparse_model.dat'))
    print('Loaded sparse model')
except IOError:
    pass


def train(batch, i):
    optim.zero_grad()
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)

    recon, residual, latent_recon, orig_recon = model.forward(spec)
    recon_spec = exp.perceptual_feature(recon)

    audio_loss = F.mse_loss(recon_spec, spec)

    if do_sparse_coding:
        latent_loss = F.mse_loss(latent_recon, orig_recon.detach())
        loss = audio_loss + latent_loss
    else:
        loss = audio_loss
    
    loss.backward()
    optim.step()
    return loss, recon


def train_sparse_coding(batch, i):
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)
        embedded = model.embed_features(spec)
        sparse_model.learning_step(embedded, n_iterations=sparse_coding_iterations)
        recon = sparse_model.do_sparse_code(embedded, n_iterations=sparse_coding_iterations)
        # loss = torch.abs(recon - embedded).sum()
        loss = F.mse_loss(recon, embedded)
    
    with torch.no_grad():
        recon = model.generate(recon)

    return loss, recon
    

@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train_sparse_coding if sparse_coding_phase else train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r = self.train(item, i)
            self.fake = r
            print(i, l.item())
            self.after_training_iteration(l)

            if i >= 6500 and not sparse_coding_phase:
                torch.save(model.state_dict(), 'model.dat')
                break

            if i > 500 and sparse_coding_phase:
                torch.save(sparse_model.state_dict(), 'sparse_model.dat')
                break
    
    