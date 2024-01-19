import numpy as np
from torch import nn
import torch
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from torch.nn import functional as F
import zounds

from util.weight_init import make_initializer

n_samples = 2 ** 14
samplerate = zounds.SR22050()
model_dim = 64

n_steps = 25

positions = pos_encoded(1, n_steps, 16).view(n_steps, 33).to(device)

stds = np.linspace(0.01, 0.99, n_steps) ** 2


def forward_process(audio, n_steps):
    degraded = audio.clone()
    n = []

    for i in range(audio.shape[0]):
        for s in range(n_steps[i]):
            noise = torch.zeros_like(degraded[i]).normal_(0, stds[s]).to(device)
            degraded[i] = degraded[i] + noise
        
        n.append(noise[None, ...])
    
    noise = torch.cat(n, dim=0)

    return audio, degraded, noise

    # for a, s in zip(audio, n_steps):
    #     for i in range(s):
    #         noise = torch.zeros_like(a).normal_(0, stds[i]).to(device)
    #         degraded = degraded + noise
    
    # return audio, degraded, noise

def reverse_process(model):
    degraded = torch.zeros(1, 1, n_samples).normal_(0, 1.5).to(device)
    for i in range(n_steps - 1, -1, -1):
        pred_noise = model.forward(degraded, positions[i, :])
        degraded = degraded - pred_noise
    return degraded

init_weights = make_initializer(0.1)


def activation(x):
    return F.leaky_relu(x, 0.2)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return activation(x)



class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)

        self.down = nn.Conv1d(channels * 2, channels, 1, 1, 0)

        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
    def forward(self, x, step, cond=None):
        batch = x.shape[0]

        skip = x

        step = step.view(batch, self.channels, 1).repeat(1, 1, x.shape[-1])

        x = torch.cat([x, step], dim=1)
        x = self.down(x)

        scale = self.scale(x + cond)
        gate = self.gate(x + cond)

        x = torch.tanh(scale) * F.sigmoid(gate)

        out = self.out(x)
        next = self.next(x) + skip

        return next, out

class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        c = channels

        self.initial = nn.Conv1d(1, channels, 7, 1, 3)

        self.embed_step = nn.Conv1d(33, channels, 1, 1, 0)

        self.stack = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 27),
            DilatedBlock(c, 81),
            DilatedBlock(c, 243),
            DilatedBlock(c, 1),
        )

        self.final = nn.Sequential(
            nn.Conv1d(c, c, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, 1, 1, 1, 0),
        )
    
        self.apply(init_weights)
    
    def forward(self, x, step, cond=None):
        batch = x.shape[0]

        n = F.leaky_relu(self.initial(x), 0.2)

        if cond is None:
            cond = torch.zeros(1, self.channels, 1, device=x.device)

        step = step.view(batch, 33, 1)
        step = self.embed_step(step)

        outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)

        for layer in self.stack:
            n, o = layer.forward(n, step, cond)
            outputs = outputs + o
        
        x = self.final(outputs)

        means = torch.mean(x, dim=-1, keepdim=True)
        x = x - means
        return x

model = Model(model_dim).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch):
    optim.zero_grad()

    if batch.shape[0] == 1:
        batch = batch.repeat(4, 1, 1)

    pos = np.random.randint(1, n_steps, (batch.shape[0],))
    p = positions[pos].view(batch.shape[0], 33, 1)

    orig, degraded, noise = forward_process(batch, pos)
    pred = model.forward(degraded, p)
    loss = F.mse_loss(pred, noise)
    loss.backward()
    optim.step()
    return loss

@readme
class DiffusionExperimentTwo(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.denoise()))

    def check_degraded(self, step=n_steps - 1):
        with torch.no_grad():
            audio, degraded, noise = forward_process(self.real[:1], (step,))
            return playable(degraded, samplerate)

    def denoise(self):
        with torch.no_grad():
            result = reverse_process(model)
            return playable(result, samplerate)
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss = train(item)
            print(loss.item())