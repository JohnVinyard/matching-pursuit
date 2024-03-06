
import torch
from torch import nn
from config.experiment import Experiment
from modules.atoms import unit_norm
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_coeffs = exp.n_samples // 2 + 1
total_dict_size = n_coeffs * 2
n_atoms = 512
n_iterations = 64

def to_freq_domain(x: torch.Tensor):
    spec = torch.fft.rfft(x, dim=-1)
    return spec

def to_real(x: torch.Tensor):
    return torch.cat([x.real, x.imag], dim=-1)

def time_domain_to_real(x: torch.Tensor):
    x = to_freq_domain(x)
    x = to_real(x)
    return x


def from_real(x: torch.Tensor):
    size = x.shape[-1]
    size = size // 2
    return torch.complex(x[..., :size], x[..., size:])

def from_freq_domain(x: torch.Tensor):
    return torch.fft.irfft(x, dim=-1)

def real_to_time_domain(x: torch.Tensor):
    x = from_real(x)
    x = from_freq_domain(x)
    return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        atoms_time_domain = torch.zeros(1, n_atoms, exp.n_samples).uniform_(-0.01, 0.01) * (torch.linspace(1, 0, exp.n_samples)[None, None, :] ** 12)
        atoms = time_domain_to_real(atoms_time_domain)
        self.atoms = nn.Parameter(atoms)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        x = time_domain_to_real(x)
        
        residual = x.clone()
        
        reconstruction = torch.zeros_like(residual)
        
        # TODO: analyze and choose subset of atoms
        atoms = unit_norm(self.atoms, axis=-1)
        
        
        for i in range(n_iterations):
            
            # TODO: autoencoder to move this into a _much_ smaller dimension
            # before correlation step
            corr = atoms * residual
            norms = torch.sum(corr, dim=-1, keepdim=True)
            # TODO: I have to subtract the correlation, not just the atom
            # norms = atoms @ residual.permute(0, 2, 1)
            values, indices = torch.max(norms, dim=1, keepdim=True)
            best_atoms = torch.take_along_dim(corr, indices, dim=1)
            
            # scaled_atoms = best_atoms * values
            residual = residual - best_atoms
            reconstruction = reconstruction + best_atoms
        
        reconstruction = real_to_time_domain(reconstruction)
        
        return reconstruction, residual
            

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    print('=======================')
    
    optim.zero_grad()
    recon, residual = model.forward(batch)
    
    # loss = exp.perceptual_loss(recon, batch)
    loss = torch.norm(residual, dim=-1).sum()
    loss.backward()
    optim.step()
    return loss, recon


@readme
class ComplexDomainMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    