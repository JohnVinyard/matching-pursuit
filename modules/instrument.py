from typing import List
from torch import nn
import torch
from modules.fft import fft_convolve, fft_shift
from modules.hypernetwork import HyperNetworkLayer
from torch.nn import functional as F
import numpy as np

from modules.transfer import freq_domain_transfer_function_to_resonance, make_waves

def exponential_decay(
        decay_values: torch.Tensor, 
        n_atoms: int, 
        n_frames: int, 
        base_resonance: float,
        n_samples: int):
    
    # decay_values = torch.sigmoid(decay_values.view(-1, n_atoms, 1).repeat(1, 1, n_frames))
    decay_values = decay_values.view(-1, n_atoms, 1).repeat(1, 1, n_frames)
    resonance_factor = (1 - base_resonance) * 0.99
    decay = base_resonance + (decay_values * resonance_factor)
    decay = torch.log(decay + 1e-12)
    decay = torch.cumsum(decay, dim=-1)
    decay = torch.exp(decay).view(-1, n_atoms, n_frames)
    
    if n_samples != n_frames:    
        decay = F.interpolate(decay, size=n_samples, mode='linear')
    
    return decay

class InstrumentLayer(nn.Module):
    
    def __init__(
            self, 
            encoding_channels: int, 
            channels: int, 
            n_frames: int, 
            n_samples: int,
            shape_channels: int,
            learnable_resonances: bool = False):
        
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.shape_channels = shape_channels
        self.learnable_resonances = learnable_resonances
        
        self.base_shape = nn.Parameter(torch.zeros(shape_channels,))
        self.deformability = nn.Parameter(torch.zeros(1).fill_(0.1))
        
        self.hyper = HyperNetworkLayer(
            shape_channels, 64, channels, encoding_channels)
        
        self.energy_hyper = HyperNetworkLayer(
            shape_channels, 16, channels, channels
        )
        
        offsets = torch.zeros(1, 1, self.encoding_channels, 1).uniform_(-np.pi, np.pi)
        self.register_buffer('offsets', offsets)
        
        if self.learnable_resonances:
            filters = torch.zeros(1, 1, self.encoding_channels, 512).uniform_(-1, 1)
            self.filters = nn.Parameter(torch.cat([filters, torch.zeros(1, 1, self.encoding_channels, self.n_samples - 512)], dim=-1))
            
            w = make_waves(
                self.n_samples, 
                np.linspace(20, 4000, num=self.encoding_channels // 4).tolist(), 
                samplerate=22050)
            self.register_buffer('res', w.view(1, 1, self.encoding_channels, self.n_samples))
            
            # self.res = nn.Parameter(torch.zeros(1, 1, self.encoding_channels, self.n_samples).uniform_(-1, 1))
            # self.res = nn.Parameter(torch.zeros(1, 1, self.encoding_channels, 257).uniform_(0, 1))
        
    def _pos_encoding(self, n_samples: int, device: torch.device):
        """Returns a filterbank with periodic functions
        """
        
        if self.learnable_resonances:
            # shifts = torch.zeros(1, 1, self.encoding_channels, 1, device=device).uniform_(-1, 1)
            # res = torch.roll(self.res, shifts=np.random.randint(self.n_samples), dims=-1)
            # res = fft_shift(self.res, shifts)
            # print(self.res.shape, self.filters.shape)
            res = fft_convolve(self.res, self.filters)
            return res
            # res = freq_domain_transfer_function_to_resonance(512, torch.clamp(self.res, 0, 1), self.n_frames)
            # res = res.view(1, 1, self.encoding_channels, -1)
            return self.res
        
        freqs = torch.linspace(0.00001, 0.49, steps=self.encoding_channels, device=device)
        t = torch.linspace(0, n_samples, steps=n_samples, device=device)
        p = torch.sin(t[None, :] * freqs[:, None] * np.pi)
        p = p.view(1, 1, self.encoding_channels, self.n_samples)
        
        
        # p = fft_shift(p, self.offsets)
        
        return p

    def forward(
            self, 
            energy: torch.Tensor,
            transforms: torch.Tensor,
            decays: torch.Tensor):
        
        """
        
        energy: a tensor describing how energy is injected into the system
        transforms: how the system is deformed over time
        decays: how quickly does energy "leak" from each channel of the input plane?
        
        Returns:
            _type_: _description_
        """
        
        batch, n_events, cp, frames = energy.shape
        
        pos = self._pos_encoding(self.n_samples, device=energy.device)
        
        envelopes = exponential_decay(
            decay_values=decays,
            n_atoms=n_events,
            n_frames=frames,
            base_resonance=0.5,
            n_samples=frames
        )
        envelopes = envelopes.view(batch, n_events, cp, frames)
        
        energy = fft_convolve(energy, envelopes)
        # energy = torch.tanh(energy)
        # orig_energy = energy
        
        energy = energy.permute(0, 1, 3, 2)
        

        # the shape describes how the control plane translates into
        # a mixture of resonators        
        _, _, shape_shape, shape_frames = transforms.shape
        
        # here, we're determining how energy is transformed into a mxiture of resonators
        transforms = transforms + (self.deformability * self.base_shape[None, None, :, None])
        
        transforms = transforms.view(batch * n_events, shape_shape, shape_frames)
        transforms = F.interpolate(transforms, size=self.n_frames, mode='linear')
        transforms = transforms.view(batch, n_events, shape_shape, frames)
        transforms = transforms.permute(0, 1, 3, 2)
        w, fwd = self.hyper.forward(transforms)
        
        # this defines how energy is passed on to the next control plane
        _, energy_fwd = self.energy_hyper.forward(transforms)
        
        energy = energy.reshape(-1, self.channels)
        
        transformed = fwd(energy)
        transformed = transformed.view(batch, n_events, frames, self.encoding_channels)
        transformed = transformed.permute(0, 1, 3, 2).view(batch * n_events, self.encoding_channels, self.n_frames)
        transformed = F.interpolate(transformed, size=self.n_samples, mode='linear')
        transformed = transformed.view(batch, n_events, self.encoding_channels, self.n_samples)
        
        orig_energy = energy_fwd(energy)
        orig_energy = orig_energy.view(batch, n_events, frames, self.channels)
        orig_energy = orig_energy.permute(0, 1, 3, 2)
        
        final = pos * torch.relu(transformed)
        final = torch.sum(final, dim=2)
        
        return final, orig_energy
        
class InstrumentStack(nn.Module):
    def __init__(
            self, 
            encoding_channels: int, 
            channels: int, 
            n_frames: int, 
            n_samples: int,
            shape_channels: int,
            n_layers: int,
            learnable_resonances: bool = False):
        
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.shape_channels = shape_channels
        self.n_layers = n_layers
        self.learnable_resonances=  learnable_resonances
        
        self.layers = nn.ModuleList([
            InstrumentLayer(
                encoding_channels, 
                channels, 
                n_frames, 
                n_samples, 
                shape_channels,
                learnable_resonances=learnable_resonances
            ) 
            for _ in range(self.n_layers)
        ])
    
    def forward(
            self, 
            energy: torch.Tensor, 
            transforms: List[torch.Tensor],
            decays: List[torch.Tensor],
            mix: torch.Tensor):
        
        batch, n_events, layers = mix.shape
        
        batch, n_events, channels, frames = energy.shape
        
        e = energy
        output = torch.zeros(batch, n_events, self.n_layers, self.n_samples, device=energy.device)
        
        for i, layer in enumerate(self.layers):
            audio, e = layer.forward(e, transforms[i], decays[i])
            output[:, :, i, :] = audio
        
        mx = torch.softmax(mix, dim=-1)
        
        output = output * mx[:, :, :, None]
        output = torch.sum(output, dim=2)
        return output