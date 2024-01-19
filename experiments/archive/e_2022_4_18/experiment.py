import numpy as np
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from torch.nn import functional as F
import torch
from torch import nn, norm
import zounds

n_samples = 2**14
samplerate = zounds.SR22050()

def self_similarity(x, ws=None, ss=None, dim=-1):
    batch = x.shape[0]

    x = x.view(-1, n_samples)

    elements = int((ws * (ws - 1)) / 2)
    x = F.pad(x, (0, ss))
    x = x.unfold(dim, ws, ss)
    window = torch.hamming_window(ws).to(x.device)
    x = x * window
    x = x[..., None, :] * x[..., :, None]


    batch_shape = x.shape[:-2]
    sim_shape = x.shape[-2:]
    x = x.reshape(np.prod(batch_shape), *sim_shape)
    row, col = torch.triu_indices(ws, ws, 1)
    x = x[:, row, col]
    a = x = x.reshape(*batch_shape, elements)
    b = x @ x.permute(0, 2, 1)

    row, col = torch.triu_indices(b.shape[-1], b.shape[-1], 1)
    b = b[:, row, col]
    
    x = torch.cat([
        a.view(batch, -1), 
        # b.view(batch, -1)
    ], dim=1)
    return x



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.audio = nn.Parameter(torch.zeros(1, 1, n_samples).uniform_(-0.01, 0.01))
    
    def forward(self, x):
        return self.audio


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train_model(batch):
    optim.zero_grad()
    recon = model.forward(None)

    real_feat = self_similarity(batch, 512, 256)
    fake_feat = self_similarity(recon, 512, 256)

    sim_loss = F.mse_loss(fake_feat, real_feat)

    norm_loss = torch.abs(torch.norm(batch, dim=-1) - torch.norm(recon, dim=-1)).mean()

    loss = sim_loss #+ (norm_loss * 0.00001)

    loss.backward()
    optim.step()
    print(loss.item())
    return recon


@readme
class MultiScaleSelfSimilarity(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.orig = None
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def real(self):
        return playable(self.orig, samplerate)
    
    def run(self):
        for item in self.stream:
            self.orig = item
            x = item.view(-1, 1, n_samples)
            self.fake = train_model(x)

