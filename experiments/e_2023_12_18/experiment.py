
import torch
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.matchingpursuit import dictionary_learning_step, sparse_code
from scratchpad.time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
import numpy as np


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

d_size = 2048
atom_size = 16384
sparse_coding_steps = 32

latent_dim = 32
n_coeffs = exp.n_samples // 2 + 1
total_coeffs = n_coeffs * 2



class LinearAutoencoder(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.encoder = nn.Linear(in_size, out_size)
        self.decoder = nn.Linear(out_size, in_size)
        
        self.apply(lambda x: exp.init_weights(x))
    
    def encode(self, x):
        e = self.encoder(x)
        return e
    
    def decode(self, x):
        d = self.decoder(x)
        return d
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded

model = LinearAutoencoder(total_coeffs, latent_dim).to(device)
optim = optimizer(model, lr=1e-3)

d = torch.zeros(d_size, atom_size, device=device).uniform_(-1, 1)
d = unit_norm(d, axis=-1)

def view_as_real(x: torch.Tensor):
    return torch.cat([x.real, x.imag], dim=-1)

def view_as_imag(x: torch.Tensor):
    size = x.shape[-1]
    real, imag = x[..., :size // 2], x[..., size // 2:]
    return torch.complex(real, imag)

def encode_coeffs(x: torch.Tensor):
    x = view_as_real(x)
    encoded = model.encode(x)
    return encoded

def decode_latent(x: torch.Tensor):
    x = model.decode(x)
    x = view_as_imag(x)
    return x


def compute_feature_map(residual: torch.Tensor, d: torch.Tensor):
    residual_spec = torch.fft.rfft(residual, dim=-1)
    residual_spec_encoded = encode_coeffs(residual_spec)
    
    diff = residual.shape[-1] - d.shape[-1]
    
    d = F.pad(d, (0, diff))
    d_spec = torch.fft.rfft(d, dim=-1)
    d_spec_encoded = encode_coeffs(d_spec)
    
    conv = residual_spec_encoded * d_spec_encoded
    
    fm_coeffs = decode_latent(conv)
    
    fm = torch.fft.irfft(fm_coeffs, dim=-1)
    
    return fm    


def train(batch, i):
    
    optim.zero_grad()
    
    with torch.no_grad():
        events, scatter = sparse_code(
            batch, 
            d, 
            n_steps=sparse_coding_steps, 
            flatten=True, 
            device=device,
            compute_feature_map=compute_feature_map)
        
        recon = scatter(batch.shape, events)
        
        new_d = dictionary_learning_step(
            batch, 
            d, 
            n_steps=sparse_coding_steps, 
            device=device, 
            compute_feature_map=compute_feature_map)
        
        d[:] = (new_d * 0.1) + (d * 0.9)
        
    
    batch_coeffs = torch.fft.rfft(batch, dim=-1)
    batch_coeffs = view_as_real(batch_coeffs)
    recon_coeffs = model.forward(batch_coeffs)
    loss = F.mse_loss(recon_coeffs, batch_coeffs)
    loss.backward()
    optim.step()
    
    return loss, recon

@readme
class ApproxMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    