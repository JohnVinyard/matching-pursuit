from scipy.misc import face
import zounds
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F
import numpy as np
from modules import stft
from modules.upsample import FFTUpsampleBlock
from util import device
from scipy.signal.windows import tukey


sr = zounds.SR22050()
n_samples = 2 ** 14
synth = zounds.SineSynthesizer(sr)
a_note = 440
c_note = 261.625565


def make_signal():
    duration = sr.frequency * (n_samples / 2)
    first_part = synth.synthesize(
        duration, [a_note / 2, a_note, a_note * 2])
    second_part = synth.synthesize(
        duration, [c_note / 2, c_note, c_note * 2])
    window = tukey(n_samples // 2)
    full = np.concatenate([first_part * window, second_part * window])
    samples = zounds.AudioSamples(full, sr)
    spec = np.abs(zounds.spectral.stft(samples))
    t = torch.from_numpy(samples).float().to(device)
    return samples, spec, t

class Predictor(nn.Module):
    def __init__(self, n_osc, n_samples):
        super().__init__()
        self.n_osc = n_osc
        self.n_samples = n_samples

        self.net = nn.Sequential(
            nn.ConvTranspose2d(2, 8, (4, 4), (2, 2), (1, 1)), # (6, 32)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(8, 16, (4, 4), (2, 2), (1, 1)), # (12, 64)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 32, (4, 4), (2, 2), (1, 1)), # (24, 128)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 64, (4, 4), (2, 2), (1, 1)), # (48, 256)
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        # target is 
        x = x.view(1, 2, 3, 16)
        x = self.net(x)
        print(x.shape)
        return x


class OscBank(nn.Module):
    def __init__(self, n_osc, n_samples):
        super().__init__()
        self.n_osc = n_osc
        self.osc_samples = nn.Parameter(
            torch.FloatTensor(n_osc, n_samples).uniform_(0, 1))
        self.amplitudes = nn.Parameter(
            torch.FloatTensor(n_osc, n_samples).uniform_(0, 1))

        self.upsample = FFTUpsampleBlock(
            self.n_osc, self.n_osc, size=n_samples, factor=(2**14) // n_samples)

    def _upsample(self, x):
        # return F.upsample(
        # x[None, ...], size=n_samples, mode='linear').view(self.n_osc, -1)
        x = self.upsample.upsample(x[None, ...])
        return x.view(self.n_osc, -1)

    def synth_params(self):
        return torch.cat([self.amplitudes, self.osc_samples], dim=1)

    def forward(self, x):
        """
        Indexing is in [-1, 1], so the nyquist frequency is 1
        """

        clamped = torch.clamp(self.osc_samples, 0, 1)
        amp = torch.clamp(self.amplitudes, 0, 1)

        clamped = self._upsample(clamped)
        amp = self._upsample(amp)

        clamped = torch.cumsum(clamped, dim=-1)
        x = torch.sin(np.pi * clamped * 2)

        x = x * amp
        return torch.sum(x, dim=0)


def spectral_mag_loss(inp, t):
    inp = stft(inp)
    t = stft(t)
    print(inp.shape)
    return F.mse_loss(inp, t)


if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    samples, spec, t = make_signal()
    spec = np.log(1e-12 + spec)

    n_model_samples = 16
    n_osc = 3

    model = OscBank(n_osc, n_model_samples).to(device)
    optim = Adam(model.parameters(), lr=1e-4, betas=(0, 0.9))

    pred = Predictor(n_osc, n_samples)
    pred(model.synth_params())

    while True:
        optim.zero_grad()
        recon = model(None)
        # loss = F.mse_loss(recon, t)
        loss = spectral_mag_loss(recon, t)
        loss.backward()
        optim.step()
        print(loss.item())

        r = zounds.AudioSamples(
            recon.data.cpu().numpy(), sr).pad_with_silence()
        rspec = np.log(1e-12 + np.abs(zounds.spectral.stft(r))[:32])
        arr = np.array(r)

        params = model.osc_samples.data.cpu().numpy().squeeze()

    input('input...')
