from lib2to3.pytree import Leaf
import numpy as np
import zounds
import torch
from torch import nn
from modules.normal_pdf import pdf
from modules.sparse import sparsify
from train.optim import optimizer
from torch.nn import functional as F

from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2 ** 15
samplerate = zounds.SR22050()

encoding_keep = 256
encoding_channels = 1024


model_dim = 128
n_bands = 128
kernel_size = 512

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


init_weights = make_initializer(0.05)

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        return x


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

        self.decode = nn.Sequential(
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 1),
        )

        self.to_means = nn.Conv1d(model_dim, model_dim, 1, 1, 0)
        self.to_stds = nn.Conv1d(model_dim, model_dim, 1, 1, 0)

        self.to_samples = nn.Conv1d(model_dim, model_dim, 1, 1, 0)

        self.apply(init_weights)

    def forward(self, x):
        x = fb.forward(x, normalize=False)
        e = self.encode(x)
        e = F.dropout(e, 0.05)
        e = sparsify(e, encoding_keep)

        d = self.decode(e)

        g = torch.mean(d, dim=-1, keepdim=True)

        m = torch.sigmoid(self.to_means(g)) * n_samples
        s = 512 + (torch.sigmoid(self.to_stds(g)) * 4096)

        x = torch.arange(0, n_samples, 1) #[None, None, :].repeat(x.shape[0], model_dim, 1)

        p = pdf(x, m, s)
        
        s = self.to_samples(d)

        x = s.sum(dim=1, keepdim=True)
        return e, x, d, p


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    encoded, recon, d, p = model.forward(batch)
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, encoded, recon, d, p


@readme
class SparseAgainExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.encoding = None
        self.decoding = None
        self.env = None

    def listen(self):
        return playable(self.fake, samplerate)

    def e(self):
        return self.encoding.data.cpu().numpy()[0]
    
    def d(self):
        return self.decoding.data.cpu().numpy()[0]
    
    def p(self):
        return self.env.data.cpu().numpy()[0]
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            loss, self.encoding, self.fake, self.decoding, self.env = train(item)
            print(loss.item())
