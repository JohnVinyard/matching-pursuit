import numpy as np
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.atoms import AudioEvent
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from train.gan import get_latent, gan_cycle
from train.optim import optimizer
from util import device, playable, readme
from torch import nn
import zounds
import torch
from torch.nn import functional as F

from util.weight_init import make_initializer

sequence_length = 64
n_harmonics = 64
n_samples = 2 ** 14
samplerate = zounds.SR22050()
n_events = 8
latent_dim = 128

init_weights = make_initializer(0.1)


band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    samplerate, 
    512, 
    scale, 
    0.01, 
    normalize_filters=True, 
    a_weighting=False).to(device)
aim = AuditoryImage(512, 64, do_windowing=False, check_cola=True).to(device)

def perceptual_features(x):
    x = fb.forward(x, normalize=False)
    x = aim(x)
    return x

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.norm = norm

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm1d(out_channels)
        self.nl = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        x = self.nl(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 4, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 8, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 16, 3, 2, 1), # 2048
            nn.LeakyReLU(0.2),

            nn.Conv1d(16, 32, 3, 2, 1), # 1024
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 64, 3, 2, 1), # 512
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 128, 3, 2, 1), # 256
            nn.LeakyReLU(0.2),

            nn.Conv1d(128, 256, 3, 2, 1), # 128
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 512, 3, 2, 1), # 64
            nn.LeakyReLU(0.2),

            nn.Conv1d(512, 512, 7, 4, 1), # 16
            nn.LeakyReLU(0.2),

            nn.Conv1d(512, 512, 3, 2, 1), # 8
            nn.LeakyReLU(0.2),

            nn.Conv1d(512, 512, 3, 2, 1), # 4
            nn.LeakyReLU(0.2),

            nn.Conv1d(512, 1, 4, 4, 0), # 4
        )

        self.apply(init_weights)
    
    def forward(self, x):
        x = self.net(x)
        return x


class Generator(nn.Module):
    def __init__(self, n_atoms=n_events):
        super().__init__()
        self.atoms = AudioEvent(
            sequence_length=sequence_length,
            n_samples=n_samples,
            n_events=n_events,
            min_f0=20,
            max_f0=1000,
            n_harmonics=n_harmonics,
            sr=samplerate,
            noise_ws=512,
            noise_step=256)
        self.n_atoms = n_atoms
        self.ln = nn.Linear(128, 128 * n_atoms)
        self.net = LinearOutputStack(128, 5, out_channels=70 * 64)
        self.baselines = LinearOutputStack(128, 3, out_channels=1)

        self.apply(init_weights)

    
    def forward(self, x):


        x = x.view(-1, latent_dim)
        x = self.ln(x)
        x = x.view(-1, self.n_atoms, 128)
        baselines = self.baselines(x).view(-1, self.n_atoms, 1)
        x = self.net(x)
        x = x.view(-1, self.n_atoms, 70, 64)

        # scale and shift
        x = (x + 1) / 2

        f0 = x[:, :, 0, :]
        osc_env = x[:, :, 1, :]
        noise_env = x[:, :, 2, :]
        overall_env = x[:, :, 3, :]
        noise_std = x[:, :, 4, :]
        harm_env = x[:, :, 5:-1, :]


        x = self.atoms.forward(
            f0,
            overall_env,
            osc_env,
            noise_env,
            harm_env,
            noise_std,
            baselines)
        x = x.sum(dim=1, keepdim=True)
        return x
        
    

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.atoms = AudioEvent(
#             sequence_length=sequence_length,
#             n_samples=n_samples,
#             n_events=n_events,
#             min_f0=20,
#             max_f0=1000,
#             n_harmonics=n_harmonics,
#             sr=samplerate,
#             noise_ws=512,
#             noise_step=256)

#         self.f0 = nn.Parameter(torch.zeros(
#             1, n_events, sequence_length).uniform_(0, 1))
#         self.osc_env = nn.Parameter(torch.zeros(
#             1, n_events, sequence_length).uniform_(0, 0.1))
#         self.noise_env = nn.Parameter(torch.zeros(
#             1, n_events, sequence_length).uniform_(0, 0.1))
#         self.overall_env = nn.Parameter(torch.zeros(
#             1, n_events, sequence_length).uniform_(0, 1))
#         self.noise_std = nn.Parameter(torch.zeros(
#             1, n_events, sequence_length).uniform_(0, 1))

        
#         self.harm_env = nn.Parameter(torch.zeros(
#             1, n_events, n_harmonics, sequence_length).uniform_(0, 0.1))

#         self.f0_baselines = nn.Parameter(torch.zeros(1, n_events, 1).uniform_(0, 1))

#     def forward(self, x):
#         x = self.atoms.forward(
#             self.f0,
#             self.overall_env,
#             self.osc_env,
#             self.noise_env,
#             self.harm_env,
#             self.noise_std,
#             self.f0_baselines)
#         x = x.sum(dim=1, keepdim=True)
#         return x


model = Generator().to(device)
optim = optimizer(model, lr=1e-4)


disc = Discriminator().to(device)
disc_optim = optimizer(disc)

def train_generator(batch):
    optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    fake = model(z)
    j = disc(fake)
    loss = least_squares_generator_loss(j)
    loss.backward()
    optim.step()
    print('G', loss.item())
    return fake

def train_disc(batch):
    disc_optim.zero_grad()
    z = get_latent(batch.shape[0], 128)
    fake = model(z)
    fj = disc(fake)
    rj = disc(batch)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())
    return batch

# def train_model(batch):
#     optim.zero_grad()
#     recon = model(None)
#     real_feat = perceptual_features(batch)
#     fake_feat = perceptual_features(recon)
#     loss = F.mse_loss(fake_feat, real_feat)
#     loss.backward()
#     optim.step()
#     print(loss.item())
#     return recon


@readme
class AtomsRLExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.recon = None
        self.real = None

    def orig(self):
        return playable(self.real, samplerate)

    def listen(self):
        return playable(self.recon, samplerate)
    
    def look(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for item in self.stream:
            item = item.view(-1, 1, n_samples)

            self.real = item
            step = next(gan_cycle)

            if step == 'gen':
                self.recon = train_generator(item)
            else:
                train_disc(item)

