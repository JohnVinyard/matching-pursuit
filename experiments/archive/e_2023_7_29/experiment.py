
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.activation import unit_sine
from modules.fft import fft_convolve
from modules.floodfill import flood_fill_loss
from modules.linear import LinearOutputStack
from modules.sparse import soft_dirac
from modules.transfer import ImpulseGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_atoms = 1024
kernel_size = 1024
n_events = 512

band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, n_atoms)
fb = morlet_filter_bank(exp.samplerate, kernel_size, scale, np.linspace(0.1, 0.9, len(scale)), normalize=True).real
fb = torch.from_numpy(fb).float().to(device)

def training_softmax(x, dim=-1):
    """
    Produce a random mixture of the soft and hard functions, such
    that softmax cannot be replied upon.  This _should_ cause
    the model to gravitate to the areas where the soft and hard functions
    are near equivalent
    """
    mixture = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device).uniform_(0, 1)
    sm = torch.softmax(x, dim=dim)
    d = soft_dirac(x)
    return (d * mixture) + (sm * (1 - mixture))

class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.atoms = nn.Parameter(torch.zeros(n_atoms, kernel_size).uniform_(-1, 1))
        self.atoms.data[:] = fb
        self.to_atom = LinearOutputStack(channels, 3, out_channels=len(scale), norm=nn.LayerNorm((channels,)))
        self.to_amps = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        batch = x.shape[0]
        atoms = soft_dirac(self.to_atom(x))
        amps = self.to_amps(x)
        atoms = atoms @ self.atoms
        atoms = atoms * amps
        return atoms.view(batch, -1, self.kernel_size)
        

class Scheduler(nn.Module):
    def __init__(self, start_size, final_size, generator, channels, factor=8):
        super().__init__()
        self.generator = Generator(channels)
        self.start_size = start_size
        self.channels = channels
        self.final_size = final_size
        self.factor = factor

        self.expand = ConvUpsample(
            channels, channels, start_size, end_size=n_events, mode='learned', batch_norm=True, out_channels=channels)
        
        self.to_location = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))

        self.amps = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
        self.sched = ImpulseGenerator(final_size, softmax=soft_dirac)
    
    def forward(self, x):
        batch = x.shape[0]

        x = self.expand(x).view(batch, self.channels, n_events).permute(0, 2, 1) # batch, events, channels
        sig = self.generator(x)

        sched = torch.sin(self.to_location(x))

        amp = torch.abs(self.amps(x))

        sig = F.pad(sig, (0, exp.n_samples - sig.shape[-1]))
        sig = sig * amp


        if sched.shape[-1] > 1:
            impulses = self.sched.forward(sched.view(-1, self.start_size)).view(batch, -1, exp.n_samples)
            final = fft_convolve(sig, impulses[:, None, :])
        else:
            final = fft_shift(sig, sched)[..., :exp.n_samples]
        
        return final


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.latent = nn.Parameter(torch.zeros((1, 16)).uniform_(-1, 1))
#         self.net = ConvUpsample(16, 8, 8, exp.n_samples, mode='nearest', out_channels=1, batch_norm=True)
#         self.apply(lambda x: exp.init_weights(x))
    
#     def forward(self, x):
#         x = self.net(self.latent)
#         return x

# model = Model().to(device)
# optim = optimizer(model, lr=1e-3)

# model2 = Model().to(device)
# optim2 = optimizer(model2, lr=1e-3)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.latents = nn.Parameter(torch.zeros(1, 128).uniform_(-1, 1))

        self.stack = Scheduler(512, exp.n_samples, Generator(128), 128)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        batch = x.shape[0]
        x = self.stack(self.latents)
        x = x.view(batch, n_events, exp.n_samples)
        x = torch.sum(x, dim=1, keepdim=True)
        return x

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)

    loss = exp.perceptual_loss(recon, batch)

    # loss = 0

    # residual = batch.clone()
    # for i in range(n_events):
    #     start_norm = torch.norm(residual, dim=-1)
    #     residual = residual - recon[:, i: i + 1, :]
    #     end_norm = torch.norm(residual, dim=-1)
    #     # maximize the change in norm for each atom individually
    #     diff = (start_norm - end_norm).mean()
    #     loss = loss - diff
        
    # loss = F.mse_loss(recon, batch)

    loss.backward()
    optim.step()
    return loss, torch.sum(recon, dim=1, keepdim=True)



@readme
class HierarchicalScheduling(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
        self.filled = None
    

    
            
    