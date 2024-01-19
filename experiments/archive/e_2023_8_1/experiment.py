
import torch
from torch import nn
import zounds
from torch.nn import functional as F
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack

from modules.pos_encode import pos_encoded
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
    kernel_size=512,
    a_weighting=False)

model_dim = exp.model_dim
n_samples = exp.n_samples

n_atoms = 1024
kernel_size = 1024
n_events = 512


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.atoms = nn.Parameter(torch.zeros(n_atoms, kernel_size).uniform_(-1, 1))
        self.to_atom = LinearOutputStack(channels, 3, out_channels=n_atoms, norm=nn.LayerNorm((channels,)))
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
        self.channels = channels
        self.final_size = final_size
        self.factor = factor

        self.expand = ConvUpsample(
            channels, channels, 8, end_size=n_events, mode='learned', batch_norm=True, out_channels=channels)
        
        self.to_location = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))

        self.amps = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))
        self.sched = ImpulseGenerator(final_size, softmax=soft_dirac)
    
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



class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, model_dim)))
        self.down = nn.Linear(33 + model_dim, model_dim)
        self.pos = nn.Parameter(torch.zeros(1, 128, 33).uniform_(-1, 1))
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        pos = self.pos.repeat(x.shape[0], 1, 1)
        x = x.permute(0, 2, 1)
        x = torch.cat([x, pos], dim=-1)
        x = self.down(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Scheduler(512, exp.n_samples, Generator(exp.model_dim), exp.model_dim)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = torch.sum(decoded, dim=1, keepdim=True)
        return decoded


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train_model(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class NerfContinuation(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train_model, exp, port=port)
    