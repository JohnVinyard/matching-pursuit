
import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from torch.distributions import Normal


from util import device, playable, readmedocs

import numpy as np

from util.weight_init import make_initializer

init_weights = make_initializer(0.1)

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

n_events = 128

class Encoder(nn.Module):
    def __init__(self, n_events):
        super().__init__()
        self.n_events = n_events
        encoder = nn.TransformerEncoderLayer(exp.model_dim, 4, exp.model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(encoder, 4)
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

    def forward(self, x):
        batch = x.shape[0]

        x = exp.fb.forward(x, normalize=False)
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        x = self.norm(x)
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        x = x.permute(0, 2, 1)
        x = self.context(x)
        x = self.norm(x)
        return x


class Decoder(nn.Module):
    def __init__(self, n_events):
        super().__init__()
        self.n_events = n_events

        self.to_location = LinearOutputStack(exp.model_dim, 3, out_channels=1)
        self.to_variance = LinearOutputStack(exp.model_dim, 3, out_channels=1)
        self.to_freq = LinearOutputStack(exp.model_dim, 3, out_channels=1)
        self.to_amp = LinearOutputStack(exp.model_dim, 3, out_channels=1)

    def forward(self, x):
        x = x.view(-1, self.n_events, exp.model_dim)

        loc = torch.sigmoid(self.to_location(x))
        var = (1e-8 + torch.sigmoid(self.to_variance(x)))

        freq = torch.sigmoid(self.to_freq(x)) ** 2
        amp = torch.sigmoid(self.to_amp(x))

        canvas = torch.zeros(x.shape[0], self.n_events, exp.n_samples, device=x.device)
        canvas[:] = freq

        sines = torch.sin(torch.cumsum(canvas * np.pi, dim=-1)) * amp

        dist = Normal(loc, var)
        rng = torch.linspace(0, 1, exp.n_samples, device=x.device)
        bumps = torch.exp(dist.log_prob(rng))

        final = sines * bumps

        x = torch.sum(final, dim=1, keepdim=True)
        x = max_norm(x)
        return x


class Model(nn.Module):
    def __init__(self, n_events):
        super().__init__()
        self.n_events = n_events
        self.events = nn.Parameter(
            torch.zeros(1, self.n_events, exp.model_dim).normal_(0, 1))
        self.encoder = Encoder(n_events)
        self.decoder = Decoder(n_events)
        self.apply(init_weights)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder.forward(self.events)
        return x


model = Model(n_events).to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return recon, loss


@readmedocs.readme
class MatchingPursuit(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None
        self.orig = None

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def real(self):
        return playable(self.orig, exp.samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.orig = item
            r, l = train(item)
            self.fake = r
            print(l.item())
