from typing import List, Optional, Tuple
import conjure
import torch
from torch import nn
from modules import stft
from modules.decompose import fft_resample
from modules.fft import randomize_phase
from modules.normalization import max_norm, unit_norm
from modules.sparse import sparsify
from spiking import SpikingModel
from util import device

from modules.hypernetwork import HyperNetworkLayer
from modules.transfer import fft_convolve
from modules.upsample import ensure_last_axis_length, interpolate_last_axis
from util.overfit import overfit_model
from util.playable import encode_audio, listen_to_sound, playable


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
        self.damping = nn.Parameter(torch.zeros(1, control_plane_dim, 1).uniform_(1e-8, 0.9999))
    
    def forward(self, forces: torch.Tensor, modifier: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        damping = self.base_resonance + (torch.clamp(self.damping, 0, 1) * self.diff)
        
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
        
        if control_plane_dim != n_resonances:
            raise ValueError(
                f'Control plane dim and resonances must be \
                the same, but were {control_plane_dim} and {n_resonances} respectively')
        
        n_frames = n_samples // control_rate
        
        self.instr = Instrument(
            control_plane_dim, 
            control_rate, 
            n_samples, 
            base_resonance, 
            n_resonances, 
            filter_size, 
            n_layers)
        
        self.control = nn.Parameter(torch.zeros(1, control_plane_dim, n_frames).uniform_(-0.01, 0.01))
        self.deformations = nn.Parameter(torch.zeros_like(self.control))
        self.damping_mod = nn.Parameter(torch.zeros(1, control_plane_dim, n_frames))
    
    def random(self) -> torch.Tensor:
        ctl = torch.zeros_like(self.control).uniform_(-0.1, 1)
        result = self.instr.forward(
            sparsify(ctl, n_to_keep=128), 
            torch.zeros_like(self.deformations), 
            torch.zeros_like(self.damping_mod)
        )
        return result
    
    def forward(self) -> torch.Tensor:
        ctl = sparsify(self.control - self.control.mean(), n_to_keep=128)
        
        # TODO: reinstate deformations and damping
        result = self.instr.forward(
            ctl, 
            torch.zeros_like(self.deformations), 
            torch.zeros_like(self.damping_mod))
        return result

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
        
        mixed = torch.einsum('bisc,c->bis', stacked, torch.softmax(self.mix, dim=-1))
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
        
        # self.to_control = nn.Parameter(torch.zeros(self.n_resonances, self.control_plane_dim).uniform_(-0.01, 0.01))
    
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
            dw, dfunc = self.deform(sparsify(deformations.permute(0, 2, 1), n_to_keep=64))
            dw = dw.view(batch, self.n_frames, self.control_plane_dim, self.n_resonances)
            w = w + dw
        
        routed = torch.einsum('abc,acbd->adc', damped, w)
        
        # to_control = torch.einsum('brf,rc->bcf', routed, self.to_control)
        to_control = routed
        
        
        upsampled = interpolate_last_axis(routed, desired_size=self.n_samples, mode='linear')
        # upsampled = fft_resample(routed, desired_size=self.n_samples, is_lowest_band=True)
        
        noise = torch.zeros_like(upsampled).uniform_(-0.01, 0.01)
        energy = upsampled * noise
        
        filters = randomize_phase(self.filters)
        filters = ensure_last_axis_length(filters, desired_size=self.n_samples)
        filters = unit_norm(filters, dim=-1)
        
        with_resonance = fft_convolve(energy, filters)
        
        return to_control, with_resonance

spiking = SpikingModel(64, 64, 64, 64, 64).to(device)

def compute_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    sl = spiking.compute_multiband_loss(a, b, hard=True)
    a = stft(a, 2048, 256, pad=True)
    b = stft(b, 2048, 256, pad=True)
    loss = torch.abs(a - b).sum()
    return loss + (sl * 1e-5)

if __name__ == '__main__':
    
    n_samples = 2 ** 16
    control_rate = 128
    filter_size = 4096
    
    control_plane_dim = 64
    n_resonances = 64
    
    n_frames = n_samples // control_rate
    
    instr = Performance(
        control_plane_dim=control_plane_dim, 
        control_rate=control_rate, 
        n_samples=n_samples, 
        base_resonance=0.02, 
        n_resonances=n_resonances, 
        filter_size=filter_size, 
        n_layers=3).to(device)
    
    def add_loggers(collection: conjure.LmdbCollection):
        other_loggers = conjure.loggers(
            ['rnd'],
            conjure.SupportedContentType.Audio.value,
            encode_audio,
            collection)
        
        return other_loggers
    
    def training_loop_hook(
        iteration: int, 
        loggers: List[conjure.Conjure], 
        model: Performance):
        
        # TODO: Again, these should be indexed by
        # name.  Relying on remembering the position
        # is not great
        rnd = loggers[0]
        
        with torch.no_grad():
            result = model.random()
            rnd(max_norm(result).view(-1))

    overfit_model(
        n_samples=n_samples,
        model=instr,
        logger_factory=add_loggers,
        loss_func=compute_loss,
        training_loop_hook=training_loop_hook,
        collection_name='interactive',
        learning_rate=1e-3
    )