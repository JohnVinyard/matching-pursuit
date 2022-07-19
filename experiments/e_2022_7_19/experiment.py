
from loss.least_squares import least_squares_disc_loss
from util.readmedocs import readme

from modules.ddsp import NoiseModel, OscillatorBank
from util.readmedocs import readme

import numpy as np
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from sklearn.cluster import MiniBatchKMeans
from modules.phase import MelScale, AudioCodec
import zounds
import torch
from torch import nn
from torch.nn import functional as F

from util.weight_init import make_initializer

n_clusters = 512
n_samples = 2 ** 14
samplerate = zounds.SR22050()
batch_size = 4

n_steps = 25
means = [0] * n_steps
stds = [0.5] * n_steps


pos_embeddings = pos_encoded(batch_size, n_steps, 16, device=device)


def forward_process(audio, n_steps):
    degraded = audio
    for i in range(n_steps):
        noise = torch.zeros_like(audio).normal_(means[i], stds[i]).to(device)
        degraded = degraded + noise
    return audio, degraded, noise


def reverse_process(model, indices, norms):
    degraded = torch.zeros(
        indices.shape[0], 1, n_samples).normal_(0, 1.6).to(device)
    for i in range(n_steps - 1, -1, -1):
        pred_noise = model.forward(
            degraded, indices, norms, pos_embeddings[:indices.shape[0], i, :])
        degraded = degraded - pred_noise
    return degraded


init_weights = make_initializer(0.05)


def activation(x): return F.leaky_relu(x, 0.2)


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return activation(x)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv1d(in_channels, out_channels, 7, 1, 3)
        self.conv2 = nn.Conv1d(out_channels + 33, out_channels, 3, 1, 1)

    def forward(self, x, step):
        x = self.conv1(x)
        x = activation(x)
        step = step.view(-1, 33, 1).repeat(1, 1, x.shape[-1])
        x = torch.cat([x, step], dim=1)
        x = self.conv2(x)
        x = F.max_pool1d(x, 7, 4, 3)
        x = activation(x)
        return x



class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.down = nn.Sequential(
            DownsamplingBlock(1, 16),
            DownsamplingBlock(16, 32),
            DownsamplingBlock(32, 64),
            DownsamplingBlock(64, 128),
        )
        self.apply(init_weights)

    def forward(self, audio, pos_embedding):
        # initial shape assertions
        for layer in self.down:
            audio = layer.forward(audio, pos_embedding)
        raise NotImplementedError()


gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-3)


def train_gen(batch):
    gen_optim.zero_grad()
    step = np.random.randint(1, n_steps)
    pos = pos_embeddings[:, step, :]
    orig, degraded, noise = forward_process(batch, step)
    pred_noise = gen.forward(degraded, pos)
    loss = torch.abs(pred_noise - noise).sum()
    loss.backward()
    gen_optim.step()
    return loss


@readme
class DiffusionWithUNet(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None
        self.real = None
        self.gen = gen

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.denoise()))

    def check_degraded(self):
        with torch.no_grad():
            audio, degraded, noise = forward_process(self.real, n_steps)
            return playable(degraded, samplerate)

    def denoise(self):
        with torch.no_grad():
            result = reverse_process(gen, self.indices[:1], self.norms[:1])
            return playable(result, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            gen_loss = train_gen(item)
            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
