import zounds
import torch
from torch import nn
from torch.nn import functional as F
from modules.phase import morlet_filter_bank
import numpy as np
from modules.psychoacoustic import PsychoacousticFeature
from util import device
from util.readmedocs import readme
from train.optim import optimizer
from util.weight_init import make_initializer
from util import playable

n_bands = 2048

model_dim = 256
window_size = 512
step_size = 256


n_samples = 2 ** 15
samplerate = zounds.SR22050()
n_frames = n_samples // step_size


band = zounds.FrequencyBand(40, samplerate.nyquist - 1000)
scale = zounds.MelScale(band, n_bands)


fb = zounds.learn.FilterBank(
    samplerate,
    512,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False)


init_weights = make_initializer(0.1)

pif = PsychoacousticFeature([128] * 6).to(device)


def perceptual_feature(x):
    x = x.view(-1, 1, n_samples)
    bands = pif.compute_feature_dict(x)
    return torch.cat(list(bands.values()), dim=-2)


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)

filters = morlet_filter_bank(
    samplerate, n_samples, scale, 0.1, normalize=True).real

print(filters.shape)

noise = np.random.uniform(-1, 1, (1, n_samples))

noise_spec = np.fft.rfft(noise, axis=-1, norm='ortho')
filter_spec = np.fft.rfft(filters, axis=-1, norm='ortho')
filtered_noise = noise_spec * filter_spec
filtered_noise = np.fft.irfft(filtered_noise, norm='ortho')

filters = torch.from_numpy(filters).view(n_bands, n_samples)
noise = torch.from_numpy(filtered_noise).view(n_bands, n_samples)

all_bands = torch.cat([filters, noise], dim=0).view(1, n_bands * 2, n_samples).float()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.downsample = nn.Sequential(
            nn.Conv1d(n_bands, model_dim, 7, 2, 3),  # 64
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 32
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 7, 2, 3),  # 16
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 1, 1, 0)
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, n_bands * 2, 1, 1, 0)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, n_samples)
        x = fb.forward(x, normalize=False)
        x = F.pad(x, (0, step_size))
        x = fb.temporal_pooling(x, window_size, step_size)

        x = self.downsample(x)
        x = self.upsample(x) ** 2

        x = F.interpolate(x, size=n_samples, mode='nearest')
        x = x * all_bands
        x = torch.mean(x, dim=1)
        return x


model = Model()
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return recon, loss


@readme
class SimpleWaveTableExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        self.af = all_bands.data.cpu().numpy().squeeze()
        self.filters = fb.filter_bank.data.cpu().numpy().squeeze()
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def orig(self):
        return playable(self.real, samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            self.fake, loss = train(item)
            print(loss.item())
