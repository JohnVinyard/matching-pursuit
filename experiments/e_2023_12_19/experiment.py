
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.overlap_add import overlap_add
from modules.angle import windowed_audio
from modules.atoms import unit_norm
from modules.linear import LinearOutputStack
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class NormPreservingNetwork(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers
        self.net = LinearOutputStack(
            channels, layers, out_channels=channels, norm=nn.LayerNorm((channels,)))
        
    
    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True)
        transformed = self.net(x)
        transformed = torch.relu(transformed)
        transformed = unit_norm(transformed)
        transformed = transformed * norms
        return transformed
        

class PhysicalModelNetwork(nn.Module):
    
    def __init__(self, window_size, n_frames):
        super().__init__()
        self.window_size = window_size
        self.step_size = window_size // 2
        self.coeffs = window_size // 2 + 1
        self.embed = NormPreservingNetwork(self.coeffs, layers=3)
        self.transform = NormPreservingNetwork(self.coeffs, layers=3)
        self.embed_shape = NormPreservingNetwork(self.coeffs, layers=3)
        self.leakage = LinearOutputStack(
            self.coeffs, layers=3, out_channels=self.coeffs, norm=nn.LayerNorm((self.coeffs,)))
        self.n_frames = n_frames
        
        self.to_spec = nn.Linear(self.coeffs, self.coeffs * 2, bias=False)   
        
        self.register_buffer('group_delay', torch.linspace(0, np.pi, self.coeffs))
        
        self.max_leakage = 0.5
        
        self.apply(lambda x: exp.init_weights(x))
             
        
    def signal_to_latent(self, x):
        batch = x.shape[0]
        windowed = windowed_audio(x, self.window_size, self.step_size)
        spec = torch.fft.rfft(windowed, dim=-1)
        mag = torch.abs(spec)
        embedded = self.embed(mag)
        return embedded.view(batch, -1, self.coeffs)
    
    def forward(self, control_signal, shape):
        batch_size = control_signal.shape[0]
        
        control = self.signal_to_latent(control_signal)
        n_frames = control.shape[-2]

        hidden_state = torch.zeros(batch_size, self.coeffs, device=control.device)
        
        frames = []
        
        for i in range(n_frames):
            hidden_state = hidden_state + control[:, i, :]
            hidden_state = self.transform(hidden_state)
            shape_latent = self.embed_shape(shape[:, i, :])
            
            leakage_ratio = self.leakage(hidden_state + shape_latent)
            leakage_ratio = torch.sigmoid(leakage_ratio) * self.max_leakage
            
            leaked = hidden_state * leakage_ratio
            spec = self.to_spec(leaked)
            real, imag = spec[:, :self.coeffs], spec[:, self.coeffs:]
            spec = torch.complex(real, imag)
            samples = torch.fft.irfft(spec)
            
            frames.append(samples[:, None, :])

            hidden_state = hidden_state - leaked
        
        frames = torch.cat(frames, dim=1)
        audio = overlap_add(frames[:, None, :, :], apply_window=True)[..., :exp.n_samples]
        audio = audio.view(batch_size, 1, exp.n_samples)
        return audio


model = PhysicalModelNetwork(512, 128).to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    with torch.no_grad():
        control_signal = torch.zeros(1, 1, 1024, device=device).uniform_(-1, 1)
        control_signal = F.pad(control_signal, (0, exp.n_samples - 1024))
        
        shape = torch.zeros(1, 1, 257, device=device).uniform_(-1, 1).repeat(1, 128, 1)
        output = model.forward(control_signal, shape)
        
        loss = torch.zeros(1, device=device)
        return loss, output


@readme
class PhysicalModel(BaseExperimentRunner):
    def __init__(
        self, 
        stream, 
        port=None, 
        save_weights=False, 
        load_weights=False):
        
        super().__init__(
            stream, 
            train, 
            exp, 
            port=port, 
            save_weights=save_weights, 
            load_weights=load_weights)
    