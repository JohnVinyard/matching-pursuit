
from typing import Callable, Dict
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.phase import morlet_filter_bank
from modules.sparse import sparsify
from modules.stft import stft
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


class SparseCodingModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.atoms = nn.Parameter(torch.zeros(1, 1024, 2048).uniform_(-0.01, 0.01))     
    
    def forward(self, x):
        
        padded = F.pad(self.atoms, (0, exp.n_samples - self.atoms.shape[-1]))
        normed = padded
        
        fm = fft_convolve(x, normed)
        fm = sparsify(fm, n_to_keep=1024)
        
        recon = fft_convolve(fm, padded)
        recon = torch.sum(recon, dim=1, keepdim=True)
        return recon


import zounds
band = zounds.FrequencyBand(40, 22050 // 2)
scale = zounds.LinearScale(band, 128)
sr = zounds.SR22050()

filter_bank = morlet_filter_bank(sr, 128, scale, 0.1, normalize=True).real
filter_bank = torch.from_numpy(filter_bank).to(device).float()


def multiband_pif(x: torch.Tensor):
    
    bands = fft_frequency_decompose(x, 512)
    d = {}
    for size, band in bands.items():
        spec = F.conv1d(band, filter_bank.view(128, 1, 128), stride=1, padding=64)
        spec = torch.relu(spec)
        
        spec = F.pad(spec, (0, 64))
        windowed = spec.unfold(-1, 64, 32)
        windowed = torch.fft.rfft(windowed, dim=-1)
        windowed = torch.abs(windowed)
        d[size] = windowed
    return d


def dict_op(
        a: Dict[int, torch.Tensor],
        b: Dict[int, torch.Tensor],
        op: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> Dict[int, torch.Tensor]:
    return {k: op(v, b[k]) for k, v in a.items()}



def multiband_pif_loss(a: torch.Tensor, b: torch.Tensor):
    batch_size = a.shape[0]
    a = multiband_pif(a)
    b = multiband_pif(b)
    diff = dict_op(a, b, lambda a, b: a - b)
    loss = sum([torch.abs(y).sum() for y in diff.values()])
    return loss / batch_size

model = SparseCodingModel().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = multiband_pif_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class SparseLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    