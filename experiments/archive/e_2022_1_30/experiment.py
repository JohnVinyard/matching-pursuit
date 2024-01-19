from turtle import forward
from torch import nn
from torch.optim import Adam
from config.dotenv import Config
import torch
from itertools import chain
from modules import stft, stft_relative_phase
import numpy as np
from modules.ddsp import NoiseModel, OscillatorBank
from modules.erb import scaled_erb
from util import device
from torch.nn import functional as F
import zounds

from data.datastore import batch_stream
from util.readmedocs import readme
from util.weight_init import make_initializer



n_samples = 2**14
sr = zounds.SR22050()
min_frequency = 20 / sr.nyquist
max_frequency = 1

init_weights = make_initializer(0.1)

def loss_func(inp, t, f_params):

    inp = inp.view(-1, 1, n_samples)
    t = t.view(-1, 1, n_samples)

    inp = stft(inp, pad=True)
    t = stft(t, pad=True)
    return F.mse_loss(inp, t)

    inp_mag, inp_phase = stft_relative_phase(inp, pad=True)
    t_mag, t_phase = stft_relative_phase(t, pad=True)

    mag_loss = F.mse_loss(inp_mag, t_mag)
    phase_loss = F.mse_loss(inp_phase, t_phase) * 0.01

    return mag_loss + phase_loss


    # inp = stft(inp, pad=True)
    # t = stft(t, pad=True)

    # loss = F.mse_loss(inp, t)

    # return loss

    # discourage big jumps in a frequency channel
    delta = torch.abs(torch.diff(f_params, axis=-1)).mean()

    # encourage frequencies to be at least an ERB apart
    # batch, n_osc, time = f_params.shape
    # f_params = f_params.permute(0, 2, 1).reshape(batch * time, n_osc)
    # erbs = scaled_erb(f_params, sr) # (batch * time, n_osc)
    # diff = f_params[:, None, :] - f_params[:, :, None]
    # diff = diff - erbs[:, :, None]
    # diff = torch.clamp(diff, -np.inf, 0)
    # erb_loss = -diff.mean()

    # encourage frequencies to push apart
    batch, n_osc, time = f_params.shape
    f_params = f_params.permute(0, 2, 1).reshape(batch * time, n_osc)
    # erbs = scaled_erb(f_params, sr) # (batch * time, n_osc)
    diff = f_params[:, None, :] - f_params[:, :, None]
    diff = torch.triu(diff)
    diff = -torch.clamp(diff, 0, 1).mean()

    # encourage frequencies to be in correct range
    overage = torch.clamp(f_params - 1, 0, np.inf)
    underage = -torch.clamp(f_params - min_frequency, -np.inf, 0)

    return loss

    # return (loss * 10) + (mean_freq * 10) + (delta * 10.0) + (diff * 2.0) #+ overage.mean() + underage.mean()


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
        x = stft(x, 512, 256, pad=True)
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
            64, 
            n_audio_samples, 
            activation=torch.sigmoid, 
            return_params=True,
            constrain=True,
            log_frequency=True)
        
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
class SineAndNoiseDecoder(object):
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
        return np.log(0.001 + np.abs(zounds.spectral.stft(self.real())))
    
    def fake(self):
        return zounds.AudioSamples(self.decoded[0].data.cpu().numpy().squeeze(), sr).pad_with_silence()
    
    def fake_spec(self):
        return np.log(0.001 + np.abs(zounds.spectral.stft(self.fake())))
    
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
