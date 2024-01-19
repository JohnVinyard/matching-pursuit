
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import unit_norm
from modules.reverb import ReverbGenerator
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


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, channels, encoding_channels, atom_size):
        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.atom_size = atom_size

        self.stack = nn.Sequential(
            nn.Conv1d(exp.n_bands, channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),
            DilatedBlock(channels, 1),    
            DilatedBlock(channels, 3),    
            DilatedBlock(channels, 9),    
            DilatedBlock(channels, 27),    
            DilatedBlock(channels, 81),    
            DilatedBlock(channels, 243),    
            DilatedBlock(channels, 1),    
        )

        self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)

        self.latent = nn.Parameter(torch.zeros(1, encoding_channels).uniform_(-1, 1))

        self.embed_latent = LinearOutputStack(
            channels, layers=3, in_channels=encoding_channels, out_channels=encoding_channels, norm=nn.LayerNorm((channels,)))
        
        self.to_atoms = ConvUpsample(
            encoding_channels, channels, 8, end_size=atom_size, out_channels=1, mode='nearest', batch_norm=True)

        self.atoms = nn.Parameter(torch.zeros(encoding_channels, 1, atom_size).uniform_(-1, 1))
        self.verb = ReverbGenerator(encoding_channels, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm(encoding_channels,))
        self.apply(lambda x: exp.init_weights(x))
    
    @property
    def unit_norm_atoms(self):
        normed = unit_norm(self.atoms, dim=-1)
        return normed
        
    def forward(self, x):
        x = exp.fb.forward(x, normalize=False)
        x = self.stack(x)

        x = self.up(x)
        x = F.dropout(x, 0.01)

        # TODO: lateral competition
        encoding = x = torch.relu(x)

        ctxt = torch.sum(encoding, dim=-1)
        latent = self.embed_latent(ctxt)
        latent = self.latent + latent

        # TODO: resonance
        atoms = self.to_atoms(latent)
        padded = F.pad(atoms, (0, exp.n_samples - self.atom_size))
        x = fft_convolve(padded, encoding)[..., :exp.n_samples]

        x = torch.sum(x, dim=1, keepdim=True)
        x = self.verb.forward(ctxt, x)
        return x, encoding

model = Model(256, 1024, 1024).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    batch_size = batch.shape[0]

    optim.zero_grad()
    recon, encoding = model.forward(batch)

    encoding = encoding.view(batch_size, -1)
    non_zero = (encoding > 0).sum()
    sparsity = non_zero / encoding.nelement()
    print('sparsity', sparsity.item(), 'n_elements', (non_zero / batch_size).item())


    sparsity_loss = torch.abs(encoding).sum() * 0.0005

    loss = exp.perceptual_loss(recon, batch) + sparsity_loss

    loss.backward()
    optim.step()
    return loss, recon

@readme
class SparsityPenalty(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    