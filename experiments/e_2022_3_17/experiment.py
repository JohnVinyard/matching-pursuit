
from random import sample
from matplotlib.pyplot import sca
import torch
from modules.latent_loss import latent_loss
from modules.pif import AuditoryImage
from modules.pos_encode import ExpandUsingPosEncodings
from util import device
from data.audiostream import audio_stream
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from train.optim import optimizer
from upsample import ConvUpsample
from util import playable
import zounds
from torch import nn
from torch.nn import functional as F
from train import gan_cycle
from util.readmedocs import readme

from util.weight_init import make_initializer

codec = AudioCodec(MelScale())

init_weights = make_initializer(0.025)

samplerate = zounds.SR22050()
n_samples = 2 ** 14

class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.up = ConvUpsample(128, 128, 4, 64, mode='learned', out_channels=128)

        self.to_harm = nn.Conv1d(128, 128, 1, 1, 0)
        self.to_noise = nn.Conv1d(128, 128, 1, 1, 0)

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


    def forward(self, x):
        x = self.up(x)
        h = self.to_harm(x)
        n = self.to_noise(x)
        h = self.osc(h)
        n = self.noise(n)
        return h, n


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        band = zounds.FrequencyBand(20, samplerate.nyquist)
        scale = zounds.MelScale(band, 128)
        self.fb = zounds.learn.FilterBank(samplerate, 512, scale, 0.1, normalize_filters=True)
        self.aim = AuditoryImage(512, 64, do_windowing=True)


        self.down = nn.Sequential(
            # (257, 128, 64)

            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(257, 128, (3, 3), (2, 2), (1, 1)) # (64, 32)
            ),

            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)) # (32, 16)
            ),

            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)) # (16, 8)
            ),

            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)) # (8, 4)
            ),

            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1)) # (4, 2)
            ),

            nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, (4, 2), (4, 2)) # (1, 1)
            ),
        )

        # self.down = nn.Sequential(
        #     # (128, 64, 257)

        #     nn.Sequential(
        #         nn.Conv3d(1, 8, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        #         nn.LeakyReLU(0.2) # (64, 16, 128)
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(8, 16, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        #         nn.LeakyReLU(0.2) # (32, 8, 64)
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(16, 32, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        #         nn.LeakyReLU(0.2) # (16, 4, 32)
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(32, 64, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        #         nn.LeakyReLU(0.2) # (8, 2, 16)
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(64, 128, (3, 3, 3), (2, 2, 2), (1, 1, 1)),
        #         nn.LeakyReLU(0.2) # (4, 1, 8)
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(128, 256, (3, 3, 5), (2, 2, 4), (1, 1, 1)),
        #         nn.LeakyReLU(0.2) # (2, 1, 2)
        #     ),

        #     nn.Sequential(
        #         nn.Conv3d(256, 128, (2, 1, 2), (2, 1, 2)),
        #     ),
        # )
        
    def forward(self, x):
        x = self.fb(x, normalize=False)
        x = self.aim(x)

        x = x\
            .view(x.shape[0], 128, 64, 257)\
            .permute(0, 3, 1, 2)\
            .reshape(-1, 257, 128, 64)
        
        features = []
        for layer in self.down:
            x = layer(x)
            features.append(x.reshape(x.shape[0], -1))
        x = x.reshape(-1, 128)
        features = torch.cat(features, dim=1)
        return x, features


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.final = LinearOutputStack(128, 3, out_channels=1)
        self.apply(init_weights)
    
    def forward(self, x):
        x, features = self.encoder(x)
        x = self.final(x)
        return x, features


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.generator = Generator(n_samples)
        self.apply(init_weights)
    
    def encode(self, x):
        x, _ = self.encoder(x)
        return x
    
    def decode(self, x):
        x = self.generator(x)
        return x
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

generator = AutoEncoder().to(device)
gen_optim = optimizer(generator, lr=1e-4)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(batch):
    gen_optim.zero_grad()
    encoded, decoded = generator(batch)
    h, n = decoded

    fj, ff = disc(h + n)
    # rj, rf = disc(batch)
    ll = latent_loss(encoded)

    # loss = F.mse_loss(ff, rf) + ll
    loss = least_squares_generator_loss(fj) + ll
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return encoded, h, n

def train_disc(batch):
    disc_optim.zero_grad()
    encoded, decoded = generator(batch)
    h, n = decoded

    fj, ff = disc(h + n)
    rj, rf = disc(batch)

    
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())

@readme
class AdversarialAutoEncoderSingleResolution(object):
    def __init__(self, overfit=False, batch_size=4):
        super().__init__()
        self.overfit = overfit
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.samplerate = samplerate

        self.orig = None
        self.z = None

        self.harm = None
        self.noise = None
    

    def real(self):
        return playable(self.orig, samplerate)
    
    def real_spec(self):
        audio = self.orig.view(self.batch_size, self.n_samples)
        spec = codec.to_frequency_domain(audio)
        return spec[0].data.cpu().numpy().squeeze()[..., 0]
    
    def fake(self):
        return playable(self.recon, samplerate)
    
    def fake_spec(self):
        audio = self.recon.view(self.batch_size, self.n_samples)
        spec = codec.to_frequency_domain(audio)
        return spec[0].data.cpu().numpy().squeeze()[..., 0]
    
    def latent(self):
        return self.z.data.cpu().numpy().squeeze()
    
    @property
    def recon(self):
        return self.harm + self.noise
    
    def run(self):
        stream = audio_stream(
            self.batch_size, 
            self.n_samples, 
            overfit=self.overfit, 
            as_torch=True)
        
        for item in stream:
            self.orig = item

            step = next(gan_cycle)
            if step == 'gen':
                e, h, n = train_gen(item)
                self.harm = h
                self.noise = n
                self.z = e
            else:
                train_disc(item)