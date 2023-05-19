import numpy as np
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.normalization import max_norm
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DilatedStack(exp.model_dim, [1, 3, 9, 27, 81, 1])
        self.decoder = ConvUpsample(128, 32, 8, end_size=exp.n_samples, mode='learned', out_channels=1)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        spec = exp.pooled_filter_bank(x)
        encoded = self.encoder.forward(spec)
        encoded, _ = encoded.max(dim=-1)
        decoded = self.decoder.forward(encoded)
        decoded = max_norm(decoded)
        return decoded


ae = AutoEncoder().to(device)
optim = optimizer(ae, lr=1e-3)


def train_ae(batch):
    optim.zero_grad()
    recon = ae.forward(batch)
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


def train(batch, i):
    return train_ae(batch)


@readme
class PointcloudAutoencoder(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
        self.fake = None
        self.vec = None
        self.encoded = None
        self.model = ae
        
    
    