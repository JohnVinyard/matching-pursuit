
from typing import Tuple
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType

from scipy.signal import square, sawtooth
from torch import nn
from torch.nn import functional as F
from config.experiment import Experiment
from modules.anticausal import AntiCausalStack
from modules.decompose import fft_frequency_decompose

from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax, step_func
from modules.sparse import sparsify, sparsify2, sparsify_vectors
from modules.stft import stft
from modules.upsample import ConvUpsample
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.music import musical_scale_hz
from util.readmedocs import readme
from torch.nn.utils.weight_norm import weight_norm
from torch.distributions import Normal

exp = Experiment(
    samplerate=22050,
    n_samples=2 ** 15,
    weight_init=0.02,
    model_dim=256,
    kernel_size=512)


# class MatchingPursuitBlock(nn.Module):
    
#     def __init__(self, channels, analysis_channels):
#         super().__init__()
#         self.channels = channels
#         self.analysis_channels = analysis_channels
#         self.analysis = weight_norm(nn.Conv1d(channels, analysis_channels, 7, 1, 3))
#         self.synthesis = weight_norm(nn.Conv1d(analysis_channels, channels, 7, 1, 3))
    
#     def forward(self, x):
#         a = self.analysis(x)
#         s = self.synthesis(a)
#         s = unit_norm(s, dim=(-1, -2))
#         corr = (s * x).sum(dim=(1, 2), keepdim=True)
#         scaled = s * corr
#         new_residual = x - scaled
#         return scaled, new_residual


# class MatchingPursuitStack(nn.Module):
#     def __init__(self, channels, analysis_channels, layers):
#         super().__init__()
#         self.layers = nn.ModuleList([MatchingPursuitBlock(channels, analysis_channels) for _ in range(layers)])
    
#     def forward(self, x):
#         scaled_atoms = []
        
#         residual = x.clone().detach()
        
#         for layer in self.layers:
#             residual = x.clone().detach()
#             sa, residual = layer.forward(residual)
            
#             scaled_atoms.append(sa[:, None, :, :])
        
#         scaled_atoms = torch.cat(scaled_atoms, dim=1)
#         return scaled_atoms, residual



class MatchingPursuitBlock(nn.Module):
    def __init__(self, latent_channels, n_atoms, n_spec_coeffs):
        super().__init__()
        self.latent_channels = latent_channels
        self.n_atoms = n_atoms
        self.n_spec_coeffs = n_spec_coeffs
        
        self.to_atom_space = nn.Conv1d(latent_channels, n_atoms, 7, 1, padding=0)
        
        self.atoms = nn.Parameter(torch.zeros(n_atoms, n_spec_coeffs, 128).uniform_(-0.01, 0.01))
    
    def forward(self, x):
        
        start_spec = x
        
        input_spec_shape = x.shape
        
        batch_size = x.shape[0]
        
        x = F.pad(x, (0, 6))
        x = self.to_atom_space(x)
        x, indices, values = sparsify(x, n_to_keep=1, return_indices=True)
        
        atom_index = indices // 128
        position = indices % 128
        
        pallette = torch.zeros(*input_spec_shape, device=x.device)
        
        for i in range(batch_size):
            ai = atom_index[i].item()
            pos = position[i].item()
            segment = pallette[i, :, pos:]
            size = segment.shape[-1]
            pallette[i, :, pos:pos + size] += self.atoms[ai, :, :size]
        
        normed = unit_norm(pallette, dim=(-1, -2))
        corr = (start_spec * normed).sum(dim=(-1, -2), keepdim=True)
        
        scaled = normed * corr
        
        residual = start_spec - scaled
        return scaled, residual
            

class Model(nn.Module):
    def __init__(self, n_atoms=1024, n_steps=16):
        super().__init__()
        self.n_steps = n_steps
        self.mp_block = MatchingPursuitBlock(1025, n_atoms, 1025)
        
        
    def forward(self, x):
        x = spec = stft(x, 2048, 256, pad=True).view(-1, 128, 1025).permute(0, 2, 1).view(-1, 1025, 128)
        
        residual = x.clone().detach()
        
        atoms = []
        
        for i in range(self.n_steps):
            residual = residual.clone().detach()
            scaled, residual = self.mp_block(residual)
            atoms.append(scaled[:, None, :, :])
        
        
        atoms = torch.cat(atoms, dim=1)
        
        
        return atoms, residual, spec
        
        
model = Model(n_atoms=512, n_steps=16).to(device)
optim = optimizer(model, lr=1e-3)        


def train(batch, i):
    optim.zero_grad()
    atoms, residual, spec = model.forward(batch)
    full_spec = torch.sum(atoms, dim=1)
    loss = F.mse_loss(full_spec, spec)
    loss.backward()
    optim.step()
    return loss, None, full_spec
    


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x.data.cpu().numpy()[0]
        return x

    return (encoded,)



@readme
class IterativeDecompositionAgain2(BaseExperimentRunner):
    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None, load_weights=True, save_weights=False, model=model):
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            load_weights=load_weights, 
            save_weights=save_weights, 
            model=model)

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)

            self.real = item
            self.fake = torch.zeros_like(item)
            self.encoded = e
            
            print(i, l.item())
            self.after_training_iteration(l, i)


    