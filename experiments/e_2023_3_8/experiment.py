import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.dilated import DilatedStack
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from modules.normalization import ExampleNorm, unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from modules.phase import morlet_filter_bank
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import Tensor, nn
from util import device, playable
from torch.nn import functional as F
import torch

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)



n_atoms = 512
atom_size = 512

d = torch.zeros(n_atoms, atom_size, requires_grad=False).uniform_(-1, 1).to(device)
d = unit_norm(d, dim=-1)

def train():
    pass

@readme
class BasicMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None
    
    @property
    def recon(self):
        instances, scatter = sparse_code(self.real, d, n_steps=100, device=device)
        recon = scatter(self.real.shape, instances)
        return playable(recon, exp.samplerate)
    
    @property
    def view_dict(self):
        return np.fft.rfft(d.data.cpu().numpy(), axis=-1, norm='ortho')
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            with torch.no_grad():
                new_d = dictionary_learning_step(item, d, n_steps=100, device=device)
                d[:] = new_d
    