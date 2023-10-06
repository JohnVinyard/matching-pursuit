
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from angle import windowed_audio
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.activation import unit_sine
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.softmax import hard_softmax, sparse_softmax
from modules.stft import stft
from modules.transfer import ImpulseGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        n_events = 128
        n_atoms = 128
        self.dense_positioning = True

        self.n_events = n_events
        self.n_atoms = n_atoms

        self.n_factors = 128
        
        self.positions = nn.Parameter(torch.zeros(128, 512).uniform_(0, 1))
        self.gen = ImpulseGenerator(exp.n_samples)

        self.sparse_positions = nn.Parameter(torch.zeros(128, self.n_factors).uniform_(0, 1))

        self.register_buffer('factors', torch.softmax(torch.linspace(100, 0.01, steps=self.n_factors), dim=-1))

    
    def forward(self, x):
        windows = windowed_audio(x, 512, 256)
        atoms = windows.view(-1, 128, 512)
        atoms = F.pad(atoms, (0, exp.n_samples - 512))

        if self.dense_positioning:
            impulses = self.gen.forward(self.positions).view(1, 128, -1)
            atoms = fft_convolve(atoms, impulses)[..., :exp.n_samples]
        else:
            pos = self.sparse_positions @ self.factors
            # pos = torch.sigmoid(pos)
            atoms = fft_shift(atoms, pos.view(-1, 1))[..., :exp.n_samples]
        
        atoms = atoms.view(-1, self.n_atoms, exp.n_samples).sum(dim=1, keepdim=True)
        return atoms

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)


    # fake_spec = stft(recon, 512, 256, pad=True)
    # real_spec = stft(batch, 512, 256, pad=True)

    # fake_sim = torch.cdist(fake_spec, fake_spec)
    # real_sim = torch.cdist(real_spec, real_spec)

    # loss = F.mse_loss(fake_sim, real_sim)

    # fake_spec = torch.fft.rfft(recon)
    # real_spec = torch.fft.rfft(batch)

    # loss = F.mse_loss(torch.abs(fake_spec), torch.abs(real_spec)) + F.mse_loss(torch.angle(fake_spec), torch.angle(real_spec.imag))
    # loss = exp.perceptual_loss(recon, batch)
    loss = F.mse_loss(recon, batch)

    loss.backward()
    optim.step()

    return loss, recon

@readme
class PointCloud(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    