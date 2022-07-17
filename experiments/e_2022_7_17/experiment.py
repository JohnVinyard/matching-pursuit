from itertools import chain
import numpy as np
import torch
from torch import nn
import zounds
from torch.nn import functional as F
from modules.latent_loss import latent_loss
from modules.linear import LinearOutputStack

from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
from torch.optim import Adam
from loss import least_squares_generator_loss, least_squares_disc_loss

n_samples = 2 ** 14
samplerate = zounds.SR22050()
model_dim = 128
batch_size = 4

positions = pos_encoded(batch_size, n_samples, 16, device=device)

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    samplerate, 512, scale, 0.1, normalize_filters=True, a_weighting=False).to(device)


def activation(x):
    # return torch.sin(x * 3)
    return F.leaky_relu(x, 0.2)


init_weights = make_initializer(0.1)


class Modulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Linear(model_dim, model_dim)
        self.bias = nn.Linear(model_dim, model_dim)

    def forward(self, x, cond):
        cond = cond.view(-1, model_dim)

        w = self.weight(cond)
        w = torch.sigmoid(w)[:, None, :]
        
        b = self.bias(cond)
        b = b[:, None, :]

        return (x * w) + b


class NerfLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        x = self.net(x)
        x = activation(x)
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
    def __init__(self, encoder=False):
        super().__init__()
        self.encoder = encoder
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 7, 4, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 7, 4, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 7, 4, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 7, 4, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 7, 4, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 7, 4, 3),
                nn.LeakyReLU(0.2),
            ),

            nn.Conv1d(model_dim, model_dim, 4, 4, 0),
        )

        self.cond = LinearOutputStack(model_dim, 3)
        if not self.encoder:
            self.judge = nn.Linear(model_dim, 1)

        self.apply(init_weights)

    def forward(self, x, cond):
        x = fb.forward(x, normalize=False)
        features = [x]

        for layer in self.net:
            x = layer(x)
            features.append(x)

        if not self.encoder:
            cond = self.cond(cond.view(-1, model_dim))
            # combine embedding and conditioning
            x = cond + x.view(-1, model_dim)
            x = self.judge(x)
        else:
            # return the embedding
            pass

        return x, features[:1]


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
        return x


model = Model().to(device)
encoder = Disc(encoder=True)
optim = Adam(chain(model.parameters(), encoder.parameters()),
             lr=1e-4, betas=(0, 0.9))

disc = Disc().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    # cond = torch.zeros(batch_size, model_dim).normal_(0, 1).to(device)
    cond, _ = encoder.forward(batch, None)
    recon = model.forward(positions, cond)
    j, fake_feat = disc.forward(recon, cond)
    _, real_feat = disc.forward(batch, cond)

    recon_loss = 0
    for i in range(len(fake_feat)):
        recon_loss = recon_loss + F.mse_loss(fake_feat[i], real_feat[i])
    
    adv_loss = least_squares_generator_loss(j)
    loss = adv_loss + latent_loss(cond.view(-1, model_dim)) + recon_loss
    loss.backward()
    optim.step()
    return recon, loss, cond


def train_disc(batch):
    disc_optim.zero_grad()
    # cond = torch.zeros(batch_size, model_dim).normal_(0, 1).to(device)
    cond, _ = encoder.forward(batch, None)
    recon = model.forward(positions, cond)

    fj, _ = disc.forward(recon, cond)
    rj, _ = disc.forward(batch, cond)
    # loss = -(torch.mean(rj) - torch.mean(fj))

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()

    # for p in disc.parameters():
        # p.data.clamp_(-0.01, 0.01)

    return loss


@readme
class NerfAgain(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.cond = None

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))

    def orig(self):
        return playable(self.real, samplerate)

    def z(self):
        return self.cond.data.cpu().numpy().squeeze()

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item

            if i % 2 == 0:
                self.fake, loss, self.cond = train_model(item)
            else:
                disc_loss = train_disc(item)

            if i > 0 and i % 10 == 0:
                print('G', loss.item())
                print('D', disc_loss.item())
