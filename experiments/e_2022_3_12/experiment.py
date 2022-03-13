from torch import nn
import torch

from config.dotenv import Config
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.metaformer import MetaFormer, PoolMixer
from modules.mixer import Mixer
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded
import zounds
from modules.transformer import Transformer
from train.gan import get_latent
from util import device
from torch.optim import Adam
from train import gan_cycle
from util.readmedocs import readme
import numpy as np

from util.weight_init import make_initializer

n_samples = 2 ** 14

samplerate = zounds.SR22050()

init_weights = make_initializer(0.05)


class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.up = ExpandUsingPosEncodings(128, 64, 16, 128, concat=True)
        self.t = LinearOutputStack(128, 7, activation=torch.sin)
        self.harm_transform = LinearOutputStack(128, 2, activation=torch.sin)
        self.noise_transform = LinearOutputStack(128, 2, activation=torch.sin)

        self.n_samples = n_samples

        self.osc = OscillatorBank(
            input_channels=128,
            n_osc=128,
            n_audio_samples=n_samples,
            activation=torch.sigmoid,
            amp_activation=torch.abs,
            return_params=False,
            constrain=True,
            log_frequency=False,
            lowest_freq=40 / samplerate.nyquist,
            sharpen=False,
            compete=False)

        self.noise = NoiseModel(
            input_channels=128,
            input_size=64,
            n_noise_frames=64,
            n_audio_samples=n_samples,
            channels=128,
            activation=lambda x: x,
            squared=False,
            mask_after=1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.up(x[:, None, :])
        
        x = self.t(x)

        h = self.harm_transform(x)
        n = self.noise_transform(x)

        h = h.permute(0, 2, 1)
        n = n.permute(0, 2, 1)

        h = self.osc(h)
        n = self.noise(n)
        return h, n


class BandEncoder(nn.Module):
    def __init__(
            self,
            channels,
            periodicity_feature_size):

        super().__init__()
        self.channels = channels
        self.periodicity_feature_size = periodicity_feature_size

        self.down = nn.Sequential(
            # (128, 32, 257)

            nn.Conv3d(1, 16, (4, 1, 4), (4, 1, 4)), # (32, 32, 64)
            nn.LeakyReLU(0.2),

            nn.Conv3d(16, 32, (4, 1, 4), (4, 1, 4)), # (8, 32, 16)
            nn.LeakyReLU(0.2),

            nn.Conv3d(32, 64, (4, 1, 4), (4, 1, 4)), # (2, 32, 4)
            nn.LeakyReLU(0.2),

            nn.Conv3d(64, 128, (2, 1, 4), (2, 1, 4)), # (1, 32, 1)
        )
        

    def forward(self, x):
        x = x.view(-1, 1, 128, 32, self.periodicity_feature_size)
        x = self.down(x).reshape(-1, 128, 32) # (batch, 128, 32)
        x = x.permute(0, 2, 1) # (batch, 32, 128)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        band = zounds.FrequencyBand(20, samplerate.nyquist)
        scale = zounds.MelScale(band, 128)
        self.period = BandEncoder(128, 257)
        self.fb = zounds.learn.FilterBank(
            samplerate, 512, scale, 0.1, normalize_filters=True)
        self.aim = AuditoryImage(512, 32)

        self.embed_pos = nn.Linear(33, 128)
        self.initial = LinearOutputStack(128, 2)
        self.down = nn.Linear(256, 128)

        self.t = Transformer(128, 5, return_features=True)
        self.final = LinearOutputStack(128, 2, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        # apply filter bank
        x = self.fb(x, normalize=False)
        # look at lags/periods
        x = self.aim(x)
        # reduce peridocity data back to channels
        x = self.period(x)

        pos = pos_encoded(x.shape[0], 32, 16, device=x.device)
        pos = self.embed_pos(pos)

        x = self.initial(x)
        x = torch.cat([x, pos], dim=-1)
        x = self.down(x)
        x, features = self.t(x)
        x = self.final(x)
        return x, features


gen = Generator(n_samples).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


def train_gen(batch, z):
    gen_optim.zero_grad()
    h, n = gen(z)
    fake = h + n
    j, feat = disc(fake)
    _, real_feat = disc(batch)
    loss = least_squares_generator_loss(j)

    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return h, n


def train_disc(batch, z):
    disc_optim.zero_grad()

    h, n = gen(z)
    fake = h + n
    fj, _ = disc(fake)

    rj, _ = disc(batch)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


codec = AudioCodec(MelScale())


def latent_stream(batch_size, overfit=False):
    with torch.no_grad():
        z = get_latent(batch_size, 128)
    while True:
        yield z
        if not overfit:
            with torch.no_grad():
                z = get_latent(batch_size, 128)

@readme
class MetaFormerExperiment(object):
    def __init__(self, overfit, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.overfit = overfit

        self.n_samples = n_samples
        self.harm = None
        self.noise = None
        self.orig = None

    def real(self):
        return zounds.AudioSamples(self.orig[0].squeeze(), samplerate).pad_with_silence()
    
    def real_spec(self):
        with torch.no_grad():
            audio = torch.from_numpy(self.orig)
            spec = codec.to_frequency_domain(audio)
            return spec[0, ..., 0].data.cpu().numpy().squeeze()

    def fake(self, harm=1, noise=1):
        result = (self.harm * harm) + (self.noise * noise)
        return zounds.AudioSamples(
            result[0].data.cpu().numpy().squeeze(), 
            samplerate).pad_with_silence()

    def fake_spec(self, harm=1, noise=1):
        with torch.no_grad():
            result = (self.harm * harm) + (self.noise * noise)
            spec = codec.to_frequency_domain(result.reshape(-1, self.n_samples))
            return spec[0, ..., 0].data.cpu().numpy().squeeze()

    def run(self):
        stream = batch_stream(
            Config.audio_path(),
            '*.wav',
            self.batch_size,
            self.n_samples,
            self.overfit)
        
        lstream = latent_stream(self.batch_size, self.overfit)

        while True:
            self.orig = audio = next(stream)
            batch = torch.from_numpy(audio).to(
                device).reshape(-1, 1, n_samples)
            
            z = next(lstream)
            step = next(gan_cycle)
            if step == 'gen':
                h, n = train_gen(batch, z)
                self.harm = h
                self.noise = n
            else:
                train_disc(batch, z)
