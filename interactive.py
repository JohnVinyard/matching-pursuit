from typing import Optional, Tuple
import torch
from torch import nn
from util import device

from modules.hypernetwork import HyperNetworkLayer
from modules.transfer import fft_convolve
from modules.upsample import ensure_last_axis_length, interpolate_last_axis
from util.playable import listen_to_sound, playable


@torch.jit.script
def sequential(forces: torch.Tensor, damping: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(forces)
    for i in range(forces.shape[-1]):
        if i == 0:
            output[..., i] = forces[..., i]
        else:
            output[..., i] = (forces[..., i] + output[..., i - 1]) * damping[..., i]
    return output

class Damping(nn.Module):
    
    def __init__(
        self, 
        control_plane_dim: int, 
        base_resonance: float, 
        n_frames: int):
        
        super().__init__()
        self.n_frames = n_frames
        self.control_plane_dim = control_plane_dim
        self.base_resonance = base_resonance
        self.max_resonance = 0.9999
        self.diff = self.max_resonance - self.base_resonance
        self.damping = nn.Parameter(torch.zeros(1, control_plane_dim, 1))
    
    def forward(self, forces: torch.Tensor, modifier: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        damping = self.base_resonance + (torch.sigmoid(self.damping) * self.diff)
        
        damping = damping.repeat(1, 1, self.n_frames)
        
        if modifier is not None:
            damping = torch.clamp(damping - torch.abs(modifier), 0, 1)
        
        damped = sequential(forces, damping)
        return damped

class Layer(nn.Module):
    
    def __init__(
        self, 
        control_plane_dim: int, 
        control_rate: int, 
        n_samples: int, 
        base_resonance: float,
        n_resonances: int,
        filter_size: int):
        
        super().__init__()
        self.control_plane_dim = control_plane_dim
        self.control_rate = control_rate
        self.n_samples = n_samples
        self.n_frames = n_samples // control_rate
        self.n_resonances = n_resonances
        self.filter_size = filter_size
        
        self.damping = Damping(control_plane_dim, base_resonance, self.n_frames)
        self.routing = nn.Parameter(torch.zeros(1, 1, control_plane_dim, n_resonances).uniform_(-0.01, 0.01))
        
        self.routing_modifier = HyperNetworkLayer(control_plane_dim, 16, control_plane_dim, n_resonances)
        
        self.filters = nn.Parameter(torch.zeros(1, self.n_resonances, self.filter_size).uniform_(-0.01, 0.01))
        
        self.to_control = nn.Parameter(torch.zeros(self.n_resonances, self.control_plane_dim).uniform_(-0.01, 0.01))
    
    def forward(
        self, 
        forces: torch.Tensor, 
        damping_modifier: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch = forces.shape[0]
        
        damped = self.damping.forward(forces, damping_modifier)
        
        w, func = self.routing_modifier(damped.permute(0, 2, 1))
        
        w = w.view(batch, self.n_frames, self.control_plane_dim, self.n_resonances)
        w = w + self.routing
        
        
        routed = torch.einsum('abc,acbd->adc', damped, w)
        
        to_control = torch.einsum('brf,rc->bcf', routed, self.to_control)
        
        
        upsampled = interpolate_last_axis(routed, desired_size=self.n_samples, mode='linear')
        noise = torch.zeros_like(upsampled).uniform_(-0.01, 0.01)
        
        energy = upsampled * noise
        
        filters = ensure_last_axis_length(self.filters, desired_size=self.n_samples)
        
        with_resonance = fft_convolve(energy, filters)
        
        return to_control, with_resonance



if __name__ == '__main__':
    
    batch_size = 3
    control_plane_dim = 16
    n_samples = 2 ** 16
    control_rate = 128
    filter_size = 1024
    n_frames = n_samples // control_rate
    
    l = Layer(
        control_plane_dim=control_plane_dim, 
        control_rate=control_rate, 
        n_samples=n_samples, 
        base_resonance=0.02, 
        n_resonances=32,
        filter_size=filter_size).to(device)
    
    control = torch.zeros(batch_size, control_plane_dim, n_frames).bernoulli_(p=0.005).to(device)
    print(control.shape, control.sum())
    
    damping_mod = torch.zeros(batch_size, control_plane_dim, n_frames).to(device)
    
    ctl, samples = l.forward(control, damping_mod)
    samples = torch.sum(samples, dim=1, keepdim=True)
    print(samples.shape)
    
    print(control)
    
    
    p = playable(samples, 22050, normalize=True)
    listen_to_sound(p, wait_for_user_input=True)