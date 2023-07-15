
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.matchingpursuit import sparse_feature_map
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


d_size = 512
kernel_size = 512

band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, d_size)


d = morlet_filter_bank(
    exp.samplerate, 
    kernel_size, 
    scale, 
    # np.linspace(0.25, 0.01, d_size), 
    0.1,
    normalize=False).real

d = torch.from_numpy(d).float().to(device)


def l1_loss(a, b):
    return torch.abs(a - b).sum()

def extract_feature(x):
    fm = sparse_feature_map(x, d, n_steps=128, device=device)
    return fm

def matching_pursuit_loss(a, b):
    a = extract_feature(a)
    b = extract_feature(b)
    return l1_loss(a, b)

def sample_loss(a, b):
    return l1_loss(a, b)

def pif_loss(a, b):
    return exp.perceptual_loss(a, b, norm='l1')

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent = nn.Parameter(torch.zeros(1, 128).uniform_(-1, 1))
        self.net = ConvUpsample(
            128, 
            16, 
            start_size=4, 
            end_size=exp.n_samples, 
            mode='nearest', 
            out_channels=1, 
            batch_norm=True)
    
    def forward(self, x):
        return self.net(self.latent)

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)

    loss = matching_pursuit_loss(recon, batch)

    loss.backward()
    optim.step()
    return loss, recon

@readme
class MatchingPursuitLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    