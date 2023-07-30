
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
    model_dim=512,
    kernel_size=1024)

n_atoms = 1024
kernel_size = 2048
n_events = 1024

band = zounds.FrequencyBand(20, exp.samplerate.nyquist)
scale = zounds.MelScale(band, n_atoms)
fb = morlet_filter_bank(exp.samplerate, kernel_size, scale, 0.5, normalize=True).real
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
        self.to_atom = LinearOutputStack(channels, 3, out_channels=len(scale), norm=nn.LayerNorm((channels,)))
        self.to_amps = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
    
    def forward(self, x):
        batch = x.shape[0]
        atoms = training_softmax(self.to_atom(x))
        amps = self.to_amps(x)
        atoms = atoms @ fb
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
        self.sched = ImpulseGenerator(final_size, softmax=training_softmax)
    
    def forward(self, x):
        batch = x.shape[0]

        x = self.expand(x).view(batch, self.channels, n_events).permute(0, 2, 1) # batch, events, channels
        sig = self.generator(x)

        sched = self.to_location(x)

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
    loss.backward()
    optim.step()
    return loss, recon

def compute_feature(x):
    x = exp.pooled_filter_bank(x)
    x = x[:, None, :, :]
    # avg = F.avg_pool2d(x, (7, 7), (1, 1), (3, 3))
    # x = x - avg
    return x

@readme
class HierarchicalScheduling(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
        self.filled = None
    

    # def real_spec(self):
    #     x = self.filled.data.cpu().numpy()[0].squeeze()
    #     return x
    
    # def fake_spec(self):
    #     return np.abs(zounds.spectral.stft(self.orig()))
    
    # def run(self):
    #     for i, item in enumerate(self.iter_items()):
    #         item = item.view(-1, 1, exp.n_samples)
    #         optim.zero_grad()
    #         optim2.zero_grad()

    #         # Traditional Loss
    #         recon_1 = model.forward(item)
    #         loss_1 = exp.perceptual_loss(recon_1, item)
    #         loss_1.backward()
    #         optim.step()

    #         # Flood fill loss
    #         recon_2 = model2.forward(item)
    #         fake = compute_feature(recon_2)
    #         batch = compute_feature(item)

    #         loss, batched_shapes = flood_fill_loss(fake, batch, threshold=1, return_shapes=True)
    #         loss.backward()
    #         optim2.step()

    #         self.fake = recon_2
    #         self.real = recon_1

    #         # show flood-fill results
    #         filled = torch.zeros_like(batch)
    #         for b in range(item.shape[0]):
    #             for val, coords in batched_shapes[b]:
    #                 # choose a color for each shape
    #                 # val = np.random.uniform(0, 1)
    #                 for x, y in coords:
    #                     filled[b, :, x, y] = val
                    
    #         self.filled = filled

    #         print(i, loss.item())
    #         self.after_training_iteration(loss)

            
    