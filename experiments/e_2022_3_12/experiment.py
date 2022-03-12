from imp import init_builtin
from torch import nn
import torch

from config.dotenv import Config
from data.datastore import batch_stream
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.metaformer import MetaFormer, PoolMixer
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded
import zounds
from train.gan import get_latent
from util import device
from torch.optim import Adam
from train import gan_cycle
from util.readmedocs import readme
import numpy as np

from util.weight_init import make_initializer

n_samples = 2 ** 14

samplerate = zounds.SR22050()

init_weights = make_initializer(0.1)


class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.up = ExpandUsingPosEncodings(128, 64, 16, 128)
        self.t = MetaFormer(128, 4, lambda channels: PoolMixer(7))
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
            lowest_freq=0.05,
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
        x = x.permute(0, 2, 1)
        h = self.osc(x)
        n = self.noise(x)
        audio = n + h
        return audio


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        band = zounds.FrequencyBand(20, samplerate.nyquist)
        scale = zounds.MelScale(band, 512)
        self.fb = zounds.learn.FilterBank(
            samplerate, 512, scale, 0.1, normalize_filters=True)

        self.embed_pos = nn.Linear(33, 128)
        self.initial = LinearOutputStack(128, 2, in_channels=512)
        self.t = MetaFormer(128, 4, lambda channels: PoolMixer(7))
        self.final = LinearOutputStack(128, 2, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.fb(x)
        x = self.fb.temporal_pooling(x, 512, 256)[..., :-1]
        pos = pos_encoded(x.shape[0], 64, 16)
        pos = self.embed_pos(pos)
        x = x.permute(0, 2, 1)
        x = self.initial(x)
        x = x + pos
        x = self.t(x)
        x = self.final(x)
        return x


gen = Generator(n_samples).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


def train_gen(batch):
    gen_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    j = disc(fake)
    loss = least_squares_generator_loss(j)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake

def train_disc(batch):
    disc_optim.zero_grad()

    z = get_latent(batch.shape[0], 128)
    fake = gen(z)
    fj = disc(fake)

    rj = disc(batch)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class MetaFormerExperiment(object):
    def __init__(self, overfit, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.overfit = overfit

        self.n_samples = n_samples
        self.decoded = None
    
    def fake(self):
        return zounds.AudioSamples(self.decoded[0].data.cpu().numpy().squeeze(), samplerate).pad_with_silence()
    
    def fake_spec(self):
        return np.log(0.01 + np.abs(zounds.spectral.stft(self.fake())))

    def run(self):
        stream = batch_stream(
            Config.audio_path(),
            '*.wav',
            self.batch_size,
            self.n_samples,
            self.overfit)

        while True:
            audio = next(stream)
            batch = torch.from_numpy(audio).to(
                device).reshape(-1, 1, n_samples)

            step = next(gan_cycle)
            if step == 'gen':
                self.decoded = train_gen(batch)
            else:
                train_disc(batch)
