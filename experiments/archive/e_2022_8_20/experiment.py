from lib2to3.pytree import Leaf
import numpy as np
import zounds
import torch
from torch import nn
from modules.normal_pdf import pdf
from modules.psychoacoustic import PsychoacousticFeature
from modules.sparse import sparsify
from train.optim import optimizer
from torch.nn import functional as F

from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2 ** 15
samplerate = zounds.SR22050()

n_atoms = 512
encoding_keep = 64

model_dim = 64

n_bands = 64
kernel_size = 256

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


pif = PsychoacousticFeature([128] * 6).to(device)

init_weights = make_initializer(0.1)

def perceptual_feature(x):
    bands = pif.compute_feature_dict(x)
    return torch.cat(bands, dim=-2)

def perceptual_loss(a, b):
    return F.mse_loss(a, b)



class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
    def forward(self, x):
        batch = x.shape[0]
        skip = x
        scale = self.scale(x)
        gate = self.gate(x)
        x = torch.tanh(scale) * F.sigmoid(gate)
        out = self.out(x)
        next = self.next(x) + skip
        return next, out


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encode = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 1),
        )

        self.up = nn.Conv1d(model_dim, n_atoms, 1, 1, 0)
        self.down = nn.Conv1d(n_atoms, model_dim, 1, 1, 0)

        self.decode = nn.Sequential(
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 1),
        )


        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]
        n = x = fb.forward(x, normalize=False)

        outputs = torch.zeros(batch, model_dim, x.shape[-1], device=x.device)
        for layer in self.encode:
            n, o = layer.forward(n)
            outputs = outputs + o
        e = outputs

        e = self.up(e)
        e = F.dropout(e, 0.05)
        n = e = sparsify(e, encoding_keep)
        n = self.down(n)

        outputs = torch.zeros(batch, model_dim, x.shape[-1], device=x.device)
        for layer in self.decode:
            n, o = layer.forward(n)
            outputs = outputs + o
        
        x = outputs

        x = torch.sum(x, dim=1, keepdim=True)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class SparseAblations(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            loss, self.fake = train(item)
            print(loss.item())
