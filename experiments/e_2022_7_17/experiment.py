import torch
from torch import nn
import zounds
from torch.nn import functional as F

from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2 ** 14
samplerate = zounds.SR22050()
model_dim = 128
batch_size = 4

positions = pos_encoded(batch_size, n_samples, 16, device=device)

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    samplerate, 512, scale, 0.1, normalize_filters=True, a_weighting=True).to(device)


init_weights = make_initializer(0.1)

class Modulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(model_dim, model_dim)
        self.bias = nn.Linear(model_dim, model_dim)
    
    def forward(self, x, cond):
        w = self.weight(cond)
        w = F.leaky_relu(w, 0.2)[:, None, :]
        b = self.bias(cond)
        b = F.leaky_relu(b, 0.2)[:, None, :]

        return (x + b) * w

class NerfLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Linear(in_channels, out_channels)
    
    def forward(self, x):
        x = self.net(x)
        x = F.leaky_relu(x, 0.2)
        return x


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.nerf = NerfLayer(in_channels, out_channels)
        self.mod = Modulator()
    
    def forward(self, x, cond):
        x = self.nerf(x)
        x = self.mod.forward(x, cond)
        return x


class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, 1, 4, 4, 0),
        )

        self.apply(init_weights)
    
    def forward(self, x):
        x = fb.forward(x, normalize=False)
        x = self.net(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Layer(33, model_dim),
            *[Layer(model_dim, model_dim) for _ in range(3)],
        )

        self.to_samples = nn.Linear(model_dim, 1)
        self.apply(init_weights)
    
    def forward(self, x, cond):
        for layer in self.net:
            x = layer(x, cond)
        
        x = self.to_samples(x)
        x = x.view(-1, 1, n_samples)
        m = torch.mean(x, dim=-1, keepdim=True)
        x = x - m
        # x = torch.sin(x * 30)
        # x = torch.tanh(x)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Disc().to(device)
disc_optim = optimizer(disc, lr=1e-3)

def train_model(batch):
    optim.zero_grad()
    cond = torch.zeros(batch_size, model_dim).normal_(0, 1).to(device)
    recon = model.forward(positions, cond)
    j = disc.forward(recon)
    loss = -torch.mean(j)
    loss.backward()
    optim.step()
    return recon, loss

def train_disc(batch):
    disc_optim.zero_grad()
    cond = torch.zeros(batch_size, model_dim).normal_(0, 1).to(device)
    recon = model.forward(positions, cond)
    fj = disc.forward(recon)
    rj = disc.forward(batch)
    loss = -(torch.mean(rj) - torch.mean(fj))
    loss.backward()
    disc_optim.step()

    for p in disc.parameters():
        p.data.clamp_(-0.01, 0.01)
    
    return loss



@readme
class NerfAgain(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def orig(self):
        return playable(self.real, samplerate)
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item

            if i % 2 == 0:
                self.fake, loss = train_model(item)
            else:
                disc_loss = train_disc(item)

            if i > 0 and i % 10 == 0:
                print('G', loss.item())
                print('D', disc_loss.item())