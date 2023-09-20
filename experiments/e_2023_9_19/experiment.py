
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from modules.normalization import unit_norm
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


iterations = 32


def init_atoms(n_atoms, atom_size):
    raw = torch.zeros(n_atoms, atom_size, device=device).uniform_(-1, 1)
    return unit_norm(raw)

atom_dict = {
    512: init_atoms(1024, 64),
    1024: init_atoms(512, 128),
    2048: init_atoms(256, 256),
    4096: init_atoms(128, 512),
    8192: init_atoms(64, 1024),
    16384: init_atoms(32, 2048),
    32768: init_atoms(16, 4096),
}


def learn(batch):
    bands = fft_frequency_decompose(batch, 512)
    for size, band in bands.items():
        print(f'learning band {size} with shape {band.shape}')
        new_d = dictionary_learning_step(band, atom_dict[size], iterations, device=device)
        atom_dict[size][:] = new_d

def code(batch):
    batch_size = batch.shape[0]

    bands = fft_frequency_decompose(batch, 512)
    coded = {size: sparse_code(band, atom_dict[size], iterations, device=device, flatten=True) for size, band in bands.items()}

    recon_bands = {}
    for size, encoded in coded.items():
        print(f'coding band {size}')
        events, scatter = encoded
        recon = scatter((batch_size, 1, bands[size].shape[-1]), events)
        recon_bands[size] = recon
    
    recon = fft_frequency_recompose(recon_bands, exp.n_samples)    
    return recon


def train(batch, i):
    with torch.no_grad():
        recon = code(batch)
        learn(batch)
        loss = F.mse_loss(recon, batch)
        return loss, recon

@readme
class MatchingPursuitV3(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    