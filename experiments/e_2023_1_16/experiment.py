import zounds
from config.experiment import Experiment
from modules.overfitraw import OverfitRawAudio
from modules.stft import morlet_filter_bank
from train.optim import optimizer
from util import playable
from util.music import MusicalScale
from util.readmedocs import readme
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class PerceptualAudioModel(nn.Module):
    def __init__(self):
        super().__init__()

        scale = zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist), 128)

        orig_filters = filters = morlet_filter_bank(
            exp.samplerate, exp.kernel_size, scale, scaling_factor=0.1, normalize=True)
        filters = np.fft.rfft(filters, axis=-1, norm='ortho')

        self.register_buffer('orig', torch.from_numpy(orig_filters))

        padded = np.pad(orig_filters, [(0, 0), (0, exp.n_samples - exp.kernel_size)])
        full_size_filters = np.fft.rfft(
            padded, axis=-1, norm='ortho')

        self.register_buffer('filters', torch.from_numpy(filters))
        self.register_buffer('full_size_filters',
                             torch.from_numpy(full_size_filters))
        

    def forward(self, x):
        x = x.view(-1, 1, exp.n_samples)

        spec = torch.fft.rfft(x, dim=-1, norm='ortho')


        conv = spec * self.full_size_filters[None, ...]

        spec = torch.fft.irfft(conv, dim=-1, norm='ortho')

        # half-wave rectification
        spec = torch.relu(spec)

        # compression
        spec = torch.sqrt(spec)

        # loss of phase locking (TODO: make this independent of sample rate)
        spec = F.avg_pool1d(spec, kernel_size=3, stride=1, padding=1)

        # compute within-band periodicity
        spec = F.pad(spec, (0, 256)).unfold(-1, 512, 256)
        spec = spec * torch.hamming_window(512, device=spec.device)[None, None, None, :]

        real = spec @ self.orig.real.T
        imag = spec @ self.orig.imag.T

        spec = torch.complex(real, imag)
        spec = torch.abs(spec)

        pooled = spec[..., 0]

        # only frequencies below the current band matter
        spec = torch.tril(spec)

        # we care about the *shape* and not the magnitude here
        norms = torch.norm(spec, dim=-1, keepdim=True)
        spec = spec / (norms + 1e-8)
        return pooled, spec


loss_model = PerceptualAudioModel()

model = OverfitRawAudio((1, 1, exp.n_samples), std=1e-4)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(None)
    r1, r2 = loss_model.forward(batch)
    f1, f2 = loss_model.forward(recon)

    spec_loss = F.mse_loss(f1, r1) 
    periodicity_loss = F.mse_loss(f2, r2)

    loss = spec_loss + periodicity_loss
    loss.backward()
    optim.step()
    return loss, recon, r1


@readme
class ScatteringLossExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None
        self.fake = None
        self.spec = None
        self.model = loss_model
    
    def orig(self):
        return playable(self.real, exp.samplerate)

    def listen(self):
        return playable(self.fake, exp.samplerate)
    
    def real_spec(self):
        return self.spec.data.cpu().numpy().squeeze().T
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r, s = train(item)
            self.spec = s
            print(l.item())
            self.fake = r
