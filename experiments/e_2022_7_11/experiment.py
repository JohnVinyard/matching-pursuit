import zounds
import torch
from torch import nn

from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from torch.nn import functional as F

from util.readmedocs import readme
from util.weight_init import make_initializer
import numpy as np


n_samples = 2**14
samplerate = zounds.SR22050()
model_dim = 128

n_frames = n_samples // 256
n_noise_frames = n_frames * 4
n_rooms = 8

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, model_dim)
fb = zounds.learn.FilterBank(
    samplerate, 
    512, 
    scale, 
    0.1, 
    normalize_filters=True, 
    a_weighting=False).to(device)


pif = PsychoacousticFeature([128] * 6).to(device)
aim = AuditoryImage(
    512, n_frames, do_windowing=True, check_cola=True).to(device)


init_weights = make_initializer(0.1)


def perceptual_feature(x):
    return pif.compute_feature_dict(x, 512, 64)


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    loss = 0
    for k, v in a.items():
        loss = loss + F.mse_loss(v, b[k])
    return loss


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
        return x


class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim,
            model_dim,
            n_samples,
            constrain=True,
            lowest_freq=25 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)

        self.noise = NoiseModel(
            model_dim,
            n_frames,
            n_noise_frames,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)

        self.verb = NeuralReverb(n_samples, n_rooms)

        self.to_rooms = LinearOutputStack(model_dim, 3, out_channels=n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, model_dim, n_frames)

        agg = x.mean(dim=-1)
        room = torch.softmax(self.to_rooms(agg), dim=-1)
        mix = torch.sigmoid(self.to_mix(agg)).view(-1, 1, 1)

        harm = self.osc.forward(x)
        noise = self.noise(x)


        dry = harm + noise
        wet = self.verb(dry, room)
        signal = (dry * mix) + (wet * (1 - mix))
        return signal


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.audio = AudioModel(n_samples)

        self.reduce = LinearOutputStack(
            model_dim, 2, in_channels=257, out_channels=8)
        self.reduce_again = nn.Conv1d(8 * model_dim, model_dim, 3, 1, 1)

        self.net = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 1),
        )

    def forward(self, x):
        x = fb.forward(x, normalize=False)
        x = aim.forward(x)

        x = self.reduce(x)
        x = x.permute(0, 3, 1, 2).reshape(-1, 8 * model_dim, n_frames)
        x = self.reduce_again(x)


        encoded = x = self.net(x)


        audio = self.audio(encoded)
        return encoded, audio


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    encoded, decoded = model.forward(batch)


    loss = perceptual_loss(decoded, batch)
    loss.backward()
    optim.step()
    return loss, encoded, decoded


@readme
class EasyExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.encoded = None
        self.model = model

    def listen(self):
        return playable(self.fake, samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def orig(self):
        return playable(self.real, samplerate)
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def encoding(self):
        return self.encoded.data.cpu().numpy()[0].T

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item

            loss, self.encoded, self.fake = train(item)

            if i % 10 == 0:
                print(loss.item())
