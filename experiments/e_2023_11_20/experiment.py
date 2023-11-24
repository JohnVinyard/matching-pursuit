
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
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

def complex_to_tensor(x: torch.Tensor):
    return torch.cat([x.real, x.imag], dim=-1)

def tensor_to_complex(x: torch.Tensor):
    last_dim = x.shape[-1] // 2
    real, imag = x[..., :last_dim], x[..., last_dim:]
    return torch.complex(real, imag)


class Model(nn.Module):
    def __init__(self, n_atoms: int, kernel_size: int, bottleneck: int):
        super().__init__()
        self.n_atoms = n_atoms
        self.kernel_size = kernel_size
        # self.n_coeffs = kernel_size // 2 + 1
        self.n_coeffs = exp.n_samples // 2 + 1
        self.bottleneck = bottleneck
        self.encode = nn.Linear(self.n_coeffs * 2, bottleneck)
        self.decode = nn.Linear(bottleneck, self.n_coeffs * 2)
        self.atoms = nn.Parameter(torch.zeros(n_atoms, kernel_size).uniform_(-1, 1))
    
    def encode(self, x: torch.Tensor):
        coeffs = torch.fft.rfft(x, dim=-1)
        t = complex_to_tensor(coeffs)
        encoded = self.encode(t)
        return encoded, t
    
    def decode(self, x: torch.Tensor):
        decoded = self.decode(x)
        c = tensor_to_complex(decoded)
        atoms = torch.fft.rfft(c)
        return atoms, decoded
    
    def approx_conv(self, a: torch.Tensor, b: torch.Tensor):
        a, _ = self.encode(a)
        b, b_coeffs = self.encode(b)
        check = self.decode(b)
        conv = a * b
        decoded = self.decode(conv)
    
    def forward(self, x):
        normed_atoms = unit_norm(self.atoms)
        normed_atoms = F.pad(normed_atoms, (0, exp.n_samples - self.kernel_size))
        
        

def train(batch, i):
    pass

@readme
class SparseGraphRepresentation(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    