
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack
from modules.fft import fft_convolve
from modules.normalization import unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from torch.nn.utils.weight_norm import weight_norm


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

class Model(nn.Module):
    def __init__(self, channels, n_atoms, atom_size):
        super().__init__()
        self.channels = channels
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.to_channels = weight_norm(nn.Conv1d(1, channels, 7, 1, 3))
        self.encoder = AntiCausalStack(channels, 2, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1])
        self.d = nn.Parameter(torch.zeros(n_atoms, atom_size).uniform_(-0.01, 0.01))
        self.to_atoms = weight_norm(nn.Conv1d(channels, n_atoms, 7, 1, 3))
        
        self.verb = ReverbGenerator(
            channels, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm(channels,), hard_choice=True)
        
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        batch, _, samples = x.shape
        # x = exp.apply_filter_bank(x)
        x = self.to_channels(x)
        encoded = self.encoder(x)
        
        summary = torch.mean(encoded, dim=-1)
        x = self.to_atoms(encoded)
        
        x = sparsify(x, n_to_keep=512)
        x = torch.relu(x)
        sparse_features = x
        
        x = x.view(batch, self.n_atoms, samples)
        
        atoms = unit_norm(self.d).view(1, self.n_atoms, self.atom_size)
        atoms = F.pad(atoms, (0, samples - self.atom_size))
        
        fm = fft_convolve(x, atoms)
        
        final = torch.sum(fm, dim=1, keepdim=True)[..., :samples]
        final = self.verb.forward(summary, final)
        
        return final, sparse_features
        

model = Model(64, 1024, 512).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon, sparse_features = model.forward(batch)
    
    nz = sparse_features > 0
    nz = torch.sum(nz, dim=(1, 2)).float().mean()
    
    print(f'AVERAGE OF {nz.item()} non-zero elements')
    
    sparsity_loss = torch.abs(sparse_features).sum() * 1e-4
    
    loss = exp.perceptual_loss(recon, batch) + sparsity_loss
    loss.backward()
    optim.step()
    return loss, recon


@readme
class RelaxedSparsity(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    