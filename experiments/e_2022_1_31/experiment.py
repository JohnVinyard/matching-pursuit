from turtle import forward
from torch import nn
from torch.optim import Adam
from config.dotenv import Config
import torch
from itertools import chain
import numpy as np
from modules.ddsp import NoiseModel, OscillatorBank
from modules.erb import scaled_erb
from modules.psychoacoustic import PsychoacousticFeature
from modules.stft import geom_basis, short_time_transform
from util import device
from torch.nn import functional as F
import zounds

from data.datastore import batch_stream
from util.readmedocs import readme
from util.weight_init import make_initializer



n_samples = 2**14
sr = zounds.SR22050()
mel_basis = torch.from_numpy(geom_basis(512, sr)).float().to(device)

min_frequency = 20 / sr.nyquist
max_frequency = 1

init_weights = make_initializer(0.1)

feature = PsychoacousticFeature().to(device)


def frequency_transform(signal):
    freq = short_time_transform(signal, mel_basis, pad=True)
    freq = torch.abs(freq)
    return freq

# def loss_func(inp, t, f_params):

#     inp = inp.view(-1, 1, n_samples)
#     t = t.view(-1, 1, n_samples)

#     inp = frequency_transform(inp)
#     t = frequency_transform(t)
#     return F.mse_loss(inp, t)


def loss_func(inp, t, f_params):
    inp = feature.compute_feature_dict(inp)
    t = feature.compute_feature_dict(t)

    loss = 0
    for k, v in inp.items():
        loss = loss + F.mse_loss(v, t[k])
    return loss


network_channels = 128


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        layers = int(np.log2(64))

        self.initial = nn.Conv1d(257, self.channels, 1, 1, 0)

        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(self.channels, self.channels, 3, 2, 1),
                nn.LeakyReLU(0.2)
        ) for _ in range(layers)])

        self.final = nn.Conv1d(self.channels, self.channels, 1, 1, 0)
        self.apply(init_weights)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)
        x = frequency_transform(x)
        x = x.view(batch_size, 64, 257)
        x = x.permute(0, 2, 1)
        x = self.initial(x)
        x = self.net(x)
        x = self.final(x)
        x = x.view(batch_size, self.channels)
        return x


class Decoder(nn.Module):
    def __init__(self, channels, n_audio_samples):
        super().__init__()
        self.channels = channels
        self.n_audio_samples = n_audio_samples

        self.osc = OscillatorBank(
            self.channels, 
            128, 
            n_audio_samples, 
            activation=lambda x: torch.clamp(x, -1, 1) ** 2, 
            return_params=True,
            constrain=True,
            log_frequency=False)
        
        self.noise = NoiseModel(channels, 32, 512, n_audio_samples, self.channels)

        layers = int(np.log2(32) - np.log2(4))
        self.initial = nn.Linear(channels, channels * 4)
        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.LeakyReLU(0.2)
            ) 
            for _ in range(layers)])
        
        self.final = nn.Conv1d(channels, channels, 3, 1, 1)

        self.apply(init_weights)
        

    def forward(self, x):
        x = self.initial(x)
        x = x.reshape(-1, self.channels, 4)
        x = self.net(x)
        x = self.final(x)
        harm, f_params = self.osc(x)
        noise = self.noise(x)
        x = harm + noise
        return x, f_params


encoder = Encoder(network_channels).to(device)
decoder = Decoder(network_channels, n_samples).to(device)
optim = Adam(chain(encoder.parameters(), decoder.parameters()),
             lr=1e-4, betas=(0, 0.9))


@readme
class SineAndNoiseDecoder2(object):
    def __init__(self, overfit=False, batch_size=8):
        super().__init__()

        self.n_samples = n_samples
        self.batch_size = 16
        self.overfit = overfit
        self.batch_size = batch_size

        self.batch = None
        self.decoded = None
        self.encoded = None
        self.f_params = None
    
    def real(self):
        return zounds.AudioSamples(self.batch[0].data.cpu().numpy().squeeze(), sr).pad_with_silence()
    
    def real_spec(self):
        b = self.batch[:1, ...]
        ft = frequency_transform(b)
        return ft.data.cpu().numpy().squeeze()
    
    def fake(self):
        return zounds.AudioSamples(self.decoded[0].data.cpu().numpy().squeeze(), sr).pad_with_silence()
    
    def fake_spec(self):
        b = self.decoded[:1, ...]
        ft = frequency_transform(b)
        return ft.data.cpu().numpy().squeeze()
    
    def latent(self):
        return self.encoded.data.cpu().numpy().squeeze()
    
    def f(self):
        return self.f_params.data.cpu().numpy()[0]

    def run(self):
        stream = batch_stream(
            Config.audio_path(),
            '*.wav',
            self.batch_size,
            self.n_samples,
            self.overfit)

        for batch in stream:

            batch /= (batch.max(axis=-1, keepdims=True) + 1e-8)

            optim.zero_grad()
            b = torch.from_numpy(batch).float().to(device)
            self.batch = b
            encoded = encoder.forward(b)
            self.encoded = encoded
            decoded, f_params = decoder(encoded)
            self.f_params = f_params
            self.decoded = decoded
            loss = loss_func(b, decoded, f_params)
            loss.backward()
            optim.step()
            print(loss.item())
