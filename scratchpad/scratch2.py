from enum import auto
import torch
import zounds
from data.audioiter import AudioIterator
from modules import stft
from modules.ddsp import NoiseModel, OscillatorBank
from modules.pif import AuditoryImage
from modules.recurrent import SerialGenerator
from torch import nn
from train.optim import optimizer
from upsample import ConvUpsample
from util import device, playable

from util.weight_init import make_initializer
from torch.nn import functional as F

n_samples = 2 ** 15
samplerate = zounds.SR22050()
model_dim = 128
frames = n_samples // 256

init_weights = make_initializer(0.05)


band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    samplerate,
    512,
    scale,
    0.01,
    normalize_filters=True,
    a_weighting=False).to(device)

aim = AuditoryImage(512, 128, do_windowing=False, check_cola=False)



def perceptual_feature(x):
    x = fb.forward(x, normalize=False)
    x = aim.forward(x)
    return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    loss = F.mse_loss(a, b)
    return loss


class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim, 
            128, 
            n_samples, 
            constrain=True, 
            lowest_freq=40 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)
        
        self.noise = NoiseModel(
            model_dim,
            frames,
            frames * 8,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)
        
    
    def forward(self, x):
        x = x.view(-1, model_dim, frames)
        harm = self.osc.forward(x)
        noise = self.noise(x)
        dry = harm + noise
        return dry


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SerialGenerator(model_dim, 4, [1, 2, 4, 8, 16, 32], activation=lambda x: F.leaky_relu(x, 0.2))
        self.audio = AudioModel(n_samples)

        # self.audio = ConvUpsample(128, 128, frames, n_samples, mode='nearest', out_channels=1, from_latent=False)
        self.apply(init_weights)
    
    def forward(self, x):
        latent = x
        seq = None
        while seq is None or seq.shape[1] < frames:
            latent, seq = self.model.forward(latent, seq)
        
        seq = seq[:, :frames, :]

        seq = seq.permute(0, 2, 1)
        audio = self.audio(seq)
        return audio

iterator = AudioIterator(
    1, n_samples, samplerate, normalize=True, overfit=True)


model = Model().to(device)
optim = optimizer(model, lr=1e-4)

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    samples = next(iterator.__iter__()).view(1, 1, n_samples)
    latent = torch.zeros(1, 1, 128).normal_(0, 1).to(device)

    def listen():
        return playable(recon, samplerate)
    
    def orig():
        return playable(samples, samplerate)

    while True:
        optim.zero_grad()
        recon = model.forward(latent).view(1, 1, n_samples)

        loss = perceptual_loss(recon, samples)
        loss.backward()
        optim.step()
        print(loss.item())

    input('Waiting...')