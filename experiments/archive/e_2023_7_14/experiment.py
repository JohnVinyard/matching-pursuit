
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.linear import LinearOutputStack
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
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

d_size = 1024
kernel_size = 512
n_steps = 512

d = torch.zeros(d_size, kernel_size, device=device).uniform_(-1, 1)

def train(batch, i):
    with torch.no_grad():
        batch = batch.view(-1, 1, exp.n_samples)

        encoded, scatter = sparse_code(batch, d, device=device, flatten=True, n_steps=n_steps)
        recon = scatter(batch.shape, encoded)
        l = F.mse_loss(recon, batch)

        new_d = dictionary_learning_step(batch, d, n_steps=n_steps)
        d[:] = new_d
        
        return l, recon

@readme
class ComplexMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    