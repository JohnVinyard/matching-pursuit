import numpy as np
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss

from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
from train.optim import optimizer
from upsample import FFTUpsampleBlock
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2 ** 14
samplerate = zounds.SR22050()
model_dim = 128
n_frames = n_samples // 256
n_noise_frames = n_frames * 8
n_rooms = 8

batch_size = 8

positions = pos_encoded(batch_size, n_frames, 16, device)

init_weights = make_initializer(0.1)


band = zounds.FrequencyBand(30, samplerate.nyquist)
scale = zounds.MelScale(band, model_dim)
fb = zounds.learn.FilterBank(
    samplerate, 
    512, 
    scale, 
    0.1, 
    normalize_filters=True, 
    a_weighting=True).to(device)


class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim,
            model_dim,
            n_samples,
            constrain=True,
            lowest_freq=40 / samplerate.nyquist,
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


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.transform = LinearOutputStack(
            model_dim, 3, in_channels=33 + model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True)
        self.net = nn.TransformerEncoder(
            layer, 4, norm=nn.LayerNorm((n_frames, model_dim)))
        self.audio = AudioModel(n_samples)
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, model_dim).repeat(1, n_frames, 1)
        x = torch.cat([x, positions], dim=-1)
        x = self.transform(x)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        x = self.audio(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.apply(init_weights)

        self.transform = LinearOutputStack(model_dim, 3, in_channels=33 + model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True)
        self.net = nn.TransformerEncoder(
            layer, 4, norm=nn.LayerNorm((n_frames, model_dim)))
        self.judge = LinearOutputStack(model_dim, 3, out_channels=1)

    def forward(self, x):
        x = torch.abs(fb.convolve(x))
        x = F.avg_pool1d(x, 512, 256, 256)[..., :n_frames]
        x = x.permute(0, 2, 1)
        x = torch.cat([x, positions], dim=-1)
        x = self.transform(x)
        x = self.net(x)
        x = self.judge(x)
        return x


gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-4)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(batch):
    gen_optim.zero_grad()
    z = torch.zeros(batch_size, model_dim).to(device).normal_(0, 1)
    fake = gen.forward(z)
    j = disc.forward(fake)
    loss = -torch.mean(j)
    loss.backward()
    gen_optim.step()
    return fake, loss


def train_disc(batch):
    disc_optim.zero_grad()
    z = torch.zeros(batch_size, model_dim).to(device).normal_(0, 1)
    fake = gen.forward(z)
    fj = disc.forward(fake)
    rj = disc.forward(batch)
    loss = -(torch.mean(rj) - torch.mean(fj))
    loss.backward()
    disc_optim.step()
    for p in disc.parameters():
        p.data.clamp_(-0.1, 0.1)
    return loss


@readme
class BackAroundAgainGanExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
        
            if i % 2 == 0:
                self.fake, gen_loss = train_gen(item)
            else:
                disc_loss = train_disc(item)
            

            if i > 0 and i % 10 == 0:
                print('===============================')
                print('G', gen_loss.item())
                print('D', disc_loss.item())
            
