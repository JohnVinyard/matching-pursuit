import torch
import zounds
from data.audiostream import audio_stream
from torch import nn
from modules.ddsp import overlap_add
from modules.phase import morlet_filter_bank, windowed_audio
from modules.psychoacoustic import PsychoacousticFeature
from train.optim import optimizer
from util import device, playable
from torch.nn import functional as F


samplerate = zounds.SR22050()
n_samples = 2**15
window_size = 512
step_size = window_size // 2
n_frames = n_samples // step_size
batch_size = 8

n_freq_bands = 256


band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, n_freq_bands)
fb = zounds.learn.FilterBank(
    samplerate, window_size, scale, 0.1, normalize_filters=True, a_weighting=True)

pif = PsychoacousticFeature([128] * 6)


def perceptual_feature(x):
    # x = fb.forward(x, normalize=True)
    return x
    # return pif.scattering_transform(
    # x, window_size=512, time_steps=128)


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    loss = F.mse_loss(a, b)
    return loss


class MelScale(nn.Module):
    def __init__(self):
        super().__init__()
        # self.basis = nn.Parameter(torch.from_numpy(morlet_filter_bank(
        # samplerate, window_size, scale, 0.01)))
        self.basis = nn.Parameter(torch.complex(
            torch.zeros(window_size, n_freq_bands).normal_(0, 1).T,
            torch.zeros(window_size, n_freq_bands).normal_(0, 1).T,
        ))

    def transformation_basis(self, other_scale):
        return other_scale._basis(self.scale, zounds.OggVorbisWindowingFunc())

    def n_time_steps(self, n_samples):
        return n_samples // (self.fft_size // 2)

    def to_time_domain(self, norms, spec):
        # spec = spec.data.cpu().numpy()

        # windowed = (spec @ self.basis).real[..., ::-1]
        windowed = torch.flip((spec @ self.basis).real, dims=(-1,))

        # enusre the windowed audio has unit norm
        windowed = windowed * torch.hann_window(window_size).to(norms.device)
        norms = torch.norm(windowed, dim=-1, keepdim=True)
        windowed = windowed / (norms + 1e-8)

        windowed = windowed * norms
        td = overlap_add(windowed[:, None, :, :], apply_window=False)
        return td

    def to_frequency_domain(self, audio_batch):
        windowed = windowed_audio(
            audio_batch, window_size, step_size)

        norms = torch.norm(windowed, dim=-1, keepdim=True)
        windowed = windowed / (norms + 1e-8)

        basis_norms = torch.norm(self.basis, dim=-1, keepdim=True)
        basis = self.basis / (basis_norms + 1e-8)

        real = windowed @ basis.real.T
        imag = windowed @ basis.imag.T

        freq_domain = torch.complex(real, imag)
        return norms, freq_domain

    def encode(self, x):
        n, f = self.to_frequency_domain(x)
        return n, f


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = MelScale()
        self.scale2 = MelScale()
        self.scale3 = MelScale()

    def _level(self, scale, x):
        norms, x = scale.encode(x)
        mags = torch.abs(x)

        _, indices = torch.max(mags, dim=-1, keepdim=True)
        values = torch.gather(x, -1, indices)
        sparse = torch.zeros_like(x)
        sparse = torch.scatter(sparse, -1, indices, values)
        recon = self.scale.to_time_domain(
            norms, sparse.view(-1, n_frames, n_freq_bands))
        return recon[..., :n_samples]

    def forward(self, x):
        r1 = self._level(self.scale, x)
        residual = x - r1

        r2 = self._level(self.scale2, residual)

        residual = residual - r2
        r3 = self._level(self.scale3, residual)
        
        return r1 + r2 + r3


model = Model()
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread()

    def listen():
        return playable(recon, samplerate)

    stream = audio_stream(batch_size, n_samples, normalize=True)
    for item in stream:
        item = item.view(-1, 1, n_samples)

        optim.zero_grad()
        recon = model.forward(item)
        loss = perceptual_loss(recon, item)
        loss.backward()
        optim.step()
        print(loss.item())
