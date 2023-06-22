
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.softmax import hard_softmax
from modules.sparse import sparsify_vectors
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

d_size = 256
kernel_size = 256
sparse_coding_iterations = 16

band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, d_size)
d = morlet_filter_bank(exp.samplerate, kernel_size, scale, np.linspace(0.25, 0.01, d_size)).real
d = torch.from_numpy(d).float().to(device)
d.requires_grad = True


def generate(batch_size):
    total_events = batch_size * sparse_coding_iterations
    amps = torch.zeros(total_events, device=device).uniform_(0.9, 1)
    positions = torch.zeros(total_events, device=device).uniform_(0, 1)
    atom_indices = (torch.zeros(total_events).uniform_(0, 1) * d_size).long()
    output = _inner_generate(
        batch_size, total_events, amps, positions, atom_indices)
    return output

    
def _inner_generate(batch_size, total_events, amps, positions, atom_indices):
    output = torch.zeros(total_events, exp.n_samples, device=device)
    for i in range(total_events):
        index = atom_indices[i]
        pos = positions[i]
        amp = amps[i]
        signal = torch.zeros(exp.n_samples, device=device)
        signal[:kernel_size] = unit_norm(d[index]) * amp
        signal = fft_shift(signal, pos)[..., :exp.n_samples]
        output[i] = signal
    
    output = output.view(batch_size, sparse_coding_iterations, exp.n_samples)
    output = torch.sum(output, dim=1, keepdim=True)
    return output


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(128 + 33, 128)
        encoder = nn.TransformerEncoderLayer(128, 4, 128, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 6, norm=None)
        self.attn = nn.Conv1d(128, 1, 1, 1, 0)
        self.to_amp = nn.Linear(128, 1)
        self.to_pos = nn.Linear(128, 512)
        self.to_atom = nn.Linear(128, d_size)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = x.view(-1, 1, exp.n_samples)
        x = exp.pooled_filter_bank(x).permute(0, 2, 1)

        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device)

        x = torch.cat([x, pos], dim=-1)
        x = self.embed(x)

        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        attn = self.attn(x)
        x, indices = sparsify_vectors(x, attn, n_to_keep=sparse_coding_iterations)

        amp = torch.relu(self.to_amp(x))

        pos = self.to_pos(x)
        pos = F.interpolate(pos, size=exp.n_samples)
        # sched = torch.softmax(pos, dim=-1)
        sched = hard_softmax(pos, dim=-1, invert=True)

        # atom = torch.softmax(self.to_atom(x), dim=-1)
        atom = self.to_atom(x)
        atom = hard_softmax(atom, dim=-1, invert=True)
        atoms = (atom @ d) * amp
        

        atoms = F.pad(atoms, (0, exp.n_samples - kernel_size))
        
        output = fft_convolve(sched, atoms)
        output = torch.sum(output, dim=1, keepdim=True)

        return output



model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class SchedulingExperiment(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):

        g = generate(self.batch_size)        

        for i, item in enumerate(self.iter_items()):
            self.real = g
            l, r = train(g.clone().detach(), i)
            self.fake = r
            print('ITER', i, l.item())
            self.after_training_iteration(l)
            g = generate(self.batch_size)

    