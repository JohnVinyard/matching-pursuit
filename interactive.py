from typing import Optional, Tuple
import torch
from torch import nn
from modules import stft
from util import device

from modules.hypernetwork import HyperNetworkLayer
from modules.transfer import fft_convolve
from modules.upsample import ensure_last_axis_length, interpolate_last_axis
from util.overfit import overfit_model
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



class Performance(nn.Module):
    
    def __init__(
            self,
            control_plane_dim: int,
            control_rate: int,
            n_samples: int,
            base_resonance: float,
            n_resonances: int,
            filter_size: int,
            n_layers: int
        ):
        super().__init__()
    
    def forward(self):
        raise NotImplementedError('implement this')

class Instrument(nn.Module):
    
    def __init__(
            self,
            control_plane_dim: int,
            control_rate: int,
            n_samples: int,
            base_resonance: float,
            n_resonances: int,
            filter_size: int,
            n_layers: int
        ):
        
        super().__init__()
        
        self.n_layers = n_layers
        
        self.layers = nn.ModuleList([
            Layer(
                control_plane_dim, 
                control_rate, 
                n_samples, 
                base_resonance, 
                n_resonances, 
                filter_size
            )
            for _ in range(n_layers)
        ])
        
        self.mix = nn.Parameter(torch.zeros(self.n_layers).uniform_(-0.01, 0.01))
    
    def forward(
        self, 
        forces: torch.Tensor, 
        deformations: torch.Tensor, 
        damping_modifier: torch.Tensor) -> torch.Tensor:
        
        batch = forces.shape[0]
        outputs = []
        
        for i, layer in enumerate(self.layers):
            if i == 0:
                control, res = layer.forward(forces, deformations, damping_modifier)
            else:
                control, res = layer.forward(control)
            
            outputs.append(torch.sum(res, dim=1, keepdim=True))
        
        stacked = torch.stack(outputs, dim=-1)
        
        mixed = torch.einsum('bisc,c->bis', stacked, self.mix)
        return mixed
        
        

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
        self.deform = HyperNetworkLayer(control_plane_dim, 16, control_plane_dim, n_resonances)
        
        self.filters = nn.Parameter(torch.zeros(1, self.n_resonances, self.filter_size).uniform_(-0.01, 0.01))
        
        self.to_control = nn.Parameter(torch.zeros(self.n_resonances, self.control_plane_dim).uniform_(-0.01, 0.01))
    
    def forward(
        self, 
        forces: torch.Tensor,
        deformations: Optional[torch.Tensor] = None,
        damping_modifier: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch = forces.shape[0]
        
        damped = self.damping.forward(forces, damping_modifier)
        
        w, func = self.routing_modifier(damped.permute(0, 2, 1))
        
        w = w.view(batch, self.n_frames, self.control_plane_dim, self.n_resonances)
        w = w + self.routing
        
        if deformations is not None:
            dw, dfunc = self.deform(deformations.permute(0, 2, 1))
            dw = dw.view(batch, self.n_frames, self.control_plane_dim, self.n_resonances)
            w = w + dw
        
        
        routed = torch.einsum('abc,acbd->adc', damped, w)
        
        to_control = torch.einsum('brf,rc->bcf', routed, self.to_control)
        
        
        upsampled = interpolate_last_axis(routed, desired_size=self.n_samples, mode='linear')
        noise = torch.zeros_like(upsampled).uniform_(-0.01, 0.01)
        
        energy = upsampled * noise
        
        filters = ensure_last_axis_length(self.filters, desired_size=self.n_samples)
        
        with_resonance = fft_convolve(energy, filters)
        
        return to_control, with_resonance


def compute_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = stft(a, 2048, 256, pad=True)
    b = stft(b, 2048, 256, pad=True)
    loss = torch.abs(a - b).sum()

if __name__ == '__main__':
    
    batch_size = 3
    control_plane_dim = 16
    n_samples = 2 ** 16
    control_rate = 128
    filter_size = 1024
    n_frames = n_samples // control_rate
    
    instr = Instrument(
        control_plane_dim=control_plane_dim, 
        control_rate=control_rate, 
        n_samples=n_samples, 
        base_resonance=0.02, 
        n_resonances=32, 
        filter_size=filter_size, 
        n_layers=3).to(device)
    
    # l = Layer(
    #     control_plane_dim=control_plane_dim, 
    #     control_rate=control_rate, 
    #     n_samples=n_samples, 
    #     base_resonance=0.02, 
    #     n_resonances=32,
    #     filter_size=filter_size).to(device)
    
    control = torch.zeros(batch_size, control_plane_dim, n_frames).bernoulli_(p=0.005).to(device)
    deformations = torch.zeros_like(control)
    
    
    damping_mod = torch.zeros(batch_size, control_plane_dim, n_frames).to(device)
    
    result = instr.forward(control, deformations, damping_mod)
    
    print(result.shape)
    
    # ctl, samples = l.forward(control, deformations, damping_mod)
    # samples = torch.sum(samples, dim=1, keepdim=True)
    # print(samples.shape)
    
    # print(control)
    
    
    # p = playable(samples, 22050, normalize=True)-*
    # listen_to_sound(p, wait_for_user_input=True)
    
    def training_loop_hook(iteration: int, loggers: List[conjure.Conjure], model: nn.Module):
        t, = loggers
        # TODO: unable to tell if time is moving forward or backward here
        times = max_norm(model.times.view(n_oscillators, -1))
        t(times)

    overfit_model(
        n_samples=n_samples,
        model=instr,
        loss_func=compute_loss,
        collection_name='dho',
        logger_factory=add_loggers,
        training_loop_hook=training_loop_hook,
        learning_rate=1e-3
    )