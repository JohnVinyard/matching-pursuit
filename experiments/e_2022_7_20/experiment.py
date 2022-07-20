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
model_dim = 128

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
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = activation(x)
        return x

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, 1, 3)
        self.context = nn.Sequential(
            DilatedBlock(out_channels, 1),
            DilatedBlock(out_channels, 3),
        )
        self.conv2 = nn.Conv1d(out_channels + 33, out_channels, 7, 4, 3)
        

    def forward(self, x, step):
        x = self.conv1(x)
        x = activation(x)

        x = self.context(x)
        
        step = step.view(x.shape[0], 33, 1).repeat(1, 1, x.shape[-1])
        x = torch.cat([x, step], dim=1)
        x = self.conv2(x)
        x = activation(x)


        return x


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels + 33, in_channels, 7, 1, 3)
        self.context = nn.Sequential(
            DilatedBlock(in_channels, 1),
            DilatedBlock(in_channels, 3),
        )
        self.up = nn.ConvTranspose1d(in_channels, in_channels, 12, 4, 4)

        self.conv2 = nn.Conv1d(in_channels, out_channels, 7, 1, 3)


    def forward(self, x, d, step):

        x = x + d

        step = step.view(x.shape[0], 33, 1).repeat(1, 1, x.shape[-1])
        x = torch.cat([x, step], dim=1)
        x = self.conv1(x)
        x = activation(x)

        x = self.context(x)

        # x = F.upsample(x, scale_factor=4, mode='nearest')
        x = self.up(x)
        
        x = activation(x)

        x = self.conv2(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            DownsamplingBlock(1, 16),
            DownsamplingBlock(16, 32),
            DownsamplingBlock(32, 64),
            DownsamplingBlock(64, 128),
            DownsamplingBlock(128, 256),
        )

        self.up = nn.Sequential(
            UpsamplingBlock(256, 128),
            UpsamplingBlock(128, 64),
            UpsamplingBlock(64, 32),
            UpsamplingBlock(32, 16),
            UpsamplingBlock(16, 1),
        )


        self.apply(init_weights)

    def forward(self, audio, pos_embedding):
        x = audio

        d = {}

        # initial shape assertions
        for layer in self.down:
            x = layer.forward(x, pos_embedding)
            d[x.shape[-1]] = x

        for layer in self.up:
            z = d[x.shape[-1]]
            x = layer.forward(x, z, pos_embedding)
        
        return x

# class DilatedBlock(nn.Module):
#     def __init__(self, channels, dilation):
#         super().__init__()
#         self.channels = channels
#         self.out = nn.Conv1d(channels, channels, 1, 1, 0)
#         self.next = nn.Conv1d(channels, channels, 1, 1, 0)

#         self.down = nn.Conv1d(channels * 2, channels, 1, 1, 0)

#         self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
#         self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
#     def forward(self, x, step, cond=None):
#         batch = x.shape[0]

#         skip = x

#         step = step.view(batch, self.channels, 1).repeat(1, 1, x.shape[-1])

#         x = torch.cat([x, step], dim=1)
#         x = self.down(x)

#         scale = self.scale(x + cond)
#         gate = self.gate(x + cond)

#         x = torch.tanh(scale) * F.sigmoid(gate)

#         out = self.out(x)
#         next = self.next(x) + skip

#         return next, out

# class Model(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.channels = channels

#         c = channels

#         self.initial = nn.Conv1d(1, channels, 7, 1, 3)

#         self.embed_step = nn.Conv1d(33, channels, 1, 1, 0)

#         self.stack = nn.Sequential(
#             DilatedBlock(c, 1),
#             DilatedBlock(c, 3),
#             DilatedBlock(c, 9),
#             DilatedBlock(c, 27),
#             DilatedBlock(c, 81),
#             DilatedBlock(c, 243),
#             DilatedBlock(c, 1),
#         )

#         self.final = nn.Sequential(
#             nn.Conv1d(c, c, 1, 1, 0),
#             nn.LeakyReLU(0.2),
#             nn.Conv1d(c, 1, 1, 1, 0),
#         )
    
#         self.apply(init_weights)
    
#     def forward(self, x, step, cond=None):
#         batch = x.shape[0]

#         n = F.leaky_relu(self.initial(x), 0.2)

#         if cond is None:
#             cond = torch.zeros(1, self.channels, 1, device=x.device)

#         step = step.view(batch, 33, 1)
#         step = self.embed_step(step)

#         outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)

#         for layer in self.stack:
#             n, o = layer.forward(n, step, cond)
#             outputs = outputs + o
        
#         x = self.final(outputs)
#         return x

# model = Model(model_dim).to(device)
model = Generator().to(device)
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