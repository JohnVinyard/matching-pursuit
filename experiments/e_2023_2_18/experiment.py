from config.experiment import Experiment
from modules.dilated import DilatedStack
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import nn
from util import device, playable
from torch.nn import functional as F
import torch
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.02,
    model_dim=64,
    kernel_size=512)

n_events = 8


class Isolator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            DilatedStack(channels, [1, 3, 9, 27, 81, 1]),
            nn.Conv1d(channels, 1, 1, 1, 0)
        )
    
    def forward(self, x):
        x = exp.fb.forward(x, normalize=False)
        x = self.net(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            DilatedStack(channels, [1, 3, 9, 27, 81, 1]),
            nn.Conv1d(channels, 1, 1, 1, 0)
        )
    
    def forward(self, x):
        x = exp.fb.forward(x, normalize=False)
        x = self.net(x)
        x = torch.sigmoid(x)
        return x

class Model(nn.Module):
    def __init__(self, channels, n_events):
        super().__init__()
        self.channels = channels
        self.n_events = n_events
        self.isolator = Isolator(channels)

    def forward(self, x):
        events = []

        for i in range(self.n_events):
            event = self.isolator.forward(x)
            events.append(event[:, None, :, :])
            x = x - event
        
        events = torch.cat(events, dim=1)
        recon = torch.sum(events, dim=1)
        residual = x

        return events, recon, residual        


model = Model(exp.model_dim, n_events).to(device)
optim = optimizer(model, lr=1e-3)

disc = Discriminator(exp.model_dim).to(device)
disc_optim = optimizer(disc, lr=1e-3)

def train_isolator(batch):
    optim.zero_grad()
    events, recon, residual = model.forward(batch)
    a = torch.cat([events, recon[:, None, :, :]], dim=1).view(-1, 1, exp.n_samples)
    indices = torch.randperm(a.shape[0])[:8]
    a = a[indices]
    j = disc.forward(a)
    adv_loss = torch.abs(1 - j).mean()
    recon_loss = F.mse_loss(recon, batch)
    loss = adv_loss + recon_loss
    loss.backward()
    optim.step()
    return loss, recon, events


def train_discriminator(batch):
    disc_optim.zero_grad()
    rj = disc.forward(batch)

    with torch.no_grad():
        events, recon, residual = model.forward(batch)
        a = torch.cat([events, recon[:, None, :, :]], dim=1).view(-1, 1, exp.n_samples)
        indices = torch.randperm(a.shape[0])[:8]
        a = a[indices]
    
    j = disc.forward(a)
    adv_loss = torch.abs(0 - j).mean() + torch.abs(1 - rj).mean()
    adv_loss.backward()
    disc_optim.step()
    return adv_loss



@readme
class Incremental(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = None
        self.real = None
        self.fake = None
        self.events = None
        self.stream = stream
    
    def listen(self):
        return playable(self.fake, exp.samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def orig(self):
        return playable(self.real, exp.samplerate)
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))
    
    def event(self, index):
        return playable(self.events[:, index, :, :], exp.samplerate)
    
    def event_spec(self, index):
        return np.abs(zounds.spectral.stft(self.event(index)))
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            if i % 2 == 0:
                d_loss = train_discriminator(item)
                print('D', d_loss.item())
            else:
                l, r, e = train_isolator(item)
                self.events = e
                self.fake = r
                print('G', l.item())
