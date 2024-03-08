
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules import stft
from modules.anticausal import AntiCausalStack
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.normalization import max_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify2
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 256
atom_size = 1024

class Model(nn.Module):
    def __init__(self, channels, n_atoms):
        super().__init__()
        self.channels = channels
        self.n_atoms = n_atoms
        
        self.atoms = nn.Parameter(torch.zeros(1, n_atoms, atom_size).uniform_(-0.01, 0.01))
        self.embed = nn.Conv1d(1, channels, 7, 1, 3)
        self.analyze = AntiCausalStack(channels, 2, [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2])
        self.up = nn.Conv1d(channels, self.n_atoms, 1, 1, 0)
        self.verb = ReverbGenerator(channels, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((channels,)))
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = self.embed(x)
        x = self.analyze(x)
        x = torch.relu(x)
        
        z = torch.mean(x, dim=-1)
        
        x = F.pad(x, (0, 512))
        pooled = F.avg_pool1d(x, 512, 1, padding=0)
        x = x[..., :exp.n_samples] - pooled[..., :exp.n_samples]
        
        x = self.up(x)
        
        sparse, packed, one_hot = sparsify2(x, n_to_keep=n_events)
        selected = one_hot @ self.atoms
        selected = F.pad(selected, (0, exp.n_samples - selected.shape[-1]))
        positioned = fft_convolve(selected, packed)
        
        positioned = self.verb.forward(z, positioned)
        return positioned

model = Model(32, 4096).to(device)
optim = optimizer(model, lr=1e-3)

def transform(x: torch.Tensor):
    batch_size, channels, _ = x.shape
    bands = multiband_transform(x)
    return torch.cat([b.view(batch_size, channels, -1) for b in bands.values()], dim=-1)

        
def multiband_transform(x: torch.Tensor):
    bands = fft_frequency_decompose(x, 512)
    # TODO: each band should have 256 frequency bins and also 256 time bins
    # this requires a window size of (n_samples // 256) * 2
    # and a window size of 512, 256
    
    d1 = {f'{k}_long': stft(v, 128, 64, pad=True) for k, v in bands.items()}
    d3 = {f'{k}_short': stft(v, 64, 32, pad=True) for k, v in bands.items()}
    d4 = {f'{k}_xs': stft(v, 16, 8, pad=True) for k, v in bands.items()}
    return dict(**d1, **d3, **d4)



def single_channel_loss_3(target: torch.Tensor, recon: torch.Tensor):
    
    target = transform(target).view(target.shape[0], -1)
    
    # full = torch.sum(recon, dim=1, keepdim=True)
    # full = transform(full).view(*target.shape)
    
    channels = transform(recon)
    
    residual = target
    
    # Try L1 norm instead of L@
    # Try choosing based on loudest patch/segment
    
    # sort channels from loudest to softest
    diff = torch.norm(channels, dim=(-1), p = 1)
    indices = torch.argsort(diff, dim=-1, descending=True)
    
    srt = torch.take_along_dim(channels, indices[:, :, None], dim=1)
    
    loss = 0
    for i in range(n_events):
        current = srt[:, i, :]
        start_norm = torch.norm(residual, dim=-1, p=1)
        # TODO: should the residual be cloned and detached each time,
        # so channels are optimized independently?
        residual = residual - current
        end_norm = torch.norm(residual, dim=-1, p=1)
        diff = -(start_norm - end_norm)
        loss = loss + diff.sum()
        
    
    return loss
    
def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    # loss = exp.perceptual_loss(recon, batch)
    loss = single_channel_loss_3(batch, recon)
    loss.backward()
    optim.step()
    
    recon = torch.sum(recon, dim=1, keepdim=True)
    recon = max_norm(recon)

    return loss, recon

@readme
class Factorization(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    