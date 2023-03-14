import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.dilated import DilatedStack
from modules.matchingpursuit import dictionary_learning_step, sparse_code, compare_conv
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



n_atoms = 1024
atom_size = 2048

d = torch.zeros(n_atoms, atom_size, requires_grad=False).uniform_(-1, 1).to(device)
d = unit_norm(d, dim=-1)

approx = 0.1

ex1, ex2 = compare_conv()

def train():
    pass

@readme
class BasicMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None
        self.ex1 = ex1.data.cpu().numpy().squeeze()
        self.ex2 = ex2.data.cpu().numpy().squeeze()

        print(self.ex1.shape, self.ex2.shape)
        print(self.ex1.max(), self.ex2.max())
    
    def recon(self, steps=256):
        instances, scatter = sparse_code(self.real[:1, ...], d, n_steps=steps, device=device, approx=approx)
        all_instances = []
        for k, v in instances.items():
            all_instances.extend(v)
        
        recon = scatter(self.real[:1, ...].shape, all_instances)
        return playable(recon, exp.samplerate)
    
    def view_dict(self):
        return np.rot90(np.abs(np.fft.rfft(d.data.cpu().numpy(), axis=-1, norm='ortho')))
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            with torch.no_grad():
                new_d = dictionary_learning_step(item, d, n_steps=100, device=device, approx=approx)
                d[:] = new_d
            
            
            
