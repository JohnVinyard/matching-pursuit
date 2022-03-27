import torch
from data.audiostream import audio_stream
import zounds
from torch import nn
from modules.audio_features import MFCC, Chroma
from modules.ddsp import NoiseModel, OscillatorBank
from modules.phase import MelScale
from train.optim import optimizer
from torch.nn import functional as F

from upsample import ConvUpsample
from util import device
from util.playable import playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2**14
samplerate = zounds.SR22050()

init_weights = make_initializer(0.18)

class Generator(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.ln = nn.Linear(128, 128 * 4)

        self.up = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)), # (8, 8)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1)), # (16, 16)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)), # (32, 32)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(16, 8, (4, 4), (2, 2), (1, 1)), # (64, 64)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(8, 4, (4, 4), (2, 2), (1, 1)), # (64, 64)
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(4, 1, (4, 1), (2, 1), (1, 0)), # (128, 64)
            nn.LeakyReLU(0.2),
        )

        # self.up = ConvUpsample(
        #     128, 128, 4, 64, mode='learned', out_channels=128)

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
        x = self.ln(x)
        x = x.reshape(x.shape[0], 128, 2, 2)
        x = self.up(x)
        x = x.view(x.shape[0], 128, 64)

        # x = self.up(x)
        h = self.to_harm(x)
        n = self.to_noise(x)
        h = self.osc(h)
        n = self.noise(n)
        return h, n


class TimeFreqRepr(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.transform = MelScale()
        self.time_steps = self.transform.n_time_steps(n_samples)
        self.n_samples = n_samples

    @property
    def freq_band(self):
        return self.transform.freq_band

    @property
    def n_bands(self):
        return self.transform.scale.n_bands

    @property
    def chroma_basis(self):
        return zounds.ChromaScale(self.freq_band)._basis(
            self.transform.scale, zounds.HanningWindowingFunc())

    def forward(self, x):
        x = x.view(-1, self.n_samples)
        x = self.transform.to_frequency_domain(x)
        x = x.permute(0, 2, 1).reshape(
            x.shape[0], self.transform.scale.n_bands, -1)
        return x


class Encoder(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples
        self.transform = TimeFreqRepr(n_samples)

        self.net = nn.Sequential(
            nn.Conv2d(1, 8, (3, 3), (2, 2), (1, 1)),  # (128, 32)
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 16, (3, 3), (2, 2), (1, 1)),  # (64, 16)
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, (3, 3), (2, 2), (1, 1)),  # (32, 8)
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2), (1, 1)),  # (16, 4)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3), (2, 2), (1, 1)),  # (8, 2)
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1)),  # (4, 1)
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, (4, 1), (4, 1), (0, 0)),  # (1, 1)
        )

    def forward(self, x):
        x = self.transform(x)
        x = x.real
        x = x.view(-1, 1, self.transform.n_bands, self.transform.time_steps)
        x = self.net(x)
        x = x.view(-1, 128)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.encoder = Encoder(n_samples)
        self.decoder = Generator(n_samples)
        self.apply(init_weights)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AudioFeatures(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.transform = TimeFreqRepr(n_samples)
        self.n_samples = n_samples
        self.chroma = Chroma(self.transform.chroma_basis)
        self.mfcc = MFCC()

    def forward(self, x):
        x = x.view(-1, 1, self.n_samples)
        orig_spec = x = self.transform(x)

        phase = torch.angle(x)
        phase = torch.diff(phase, dim=-1)
        phase = torch.sin(phase)
        phase = torch.mean(phase, dim=-1)

        x = torch.abs(x)
        

        pooled = F.avg_pool2d(x[:, None, :, :], (32, 1), (32, 1))
        pooled = pooled.view(x.shape[0], 8, -1)

        loudness = torch.norm(x, dim=1, keepdim=True)
        chroma = self.chroma(x)
        mfcc = self.mfcc(x)
        return loudness, chroma, mfcc, pooled, phase, orig_spec


ae = AutoEncoder(n_samples).to(device)
optim = optimizer(ae, lr=1e-4)

features = AudioFeatures(n_samples).to(device)


def train_ae(batch):
    optim.zero_grad()
    encoded, decoded = ae(batch)
    harm, noise = decoded
    decoded = harm + noise

    rl, rc, rm, rp, r_phase, rspec = features(batch)
    fl, fc, fm, fp, f_phase, fspec = features(decoded)

    chroma_loss = F.mse_loss(fc, rc) * 1
    mfcc_loss = F.mse_loss(fm, rm) * 1
    pooled_loss = F.mse_loss(fp, rp) * 100
    phase_loss = F.mse_loss(f_phase, r_phase) * 1

    # print(chroma_loss.item(), mfcc_loss.item(), pooled_loss.item(), phase_loss.item())

    loss = chroma_loss + mfcc_loss + pooled_loss + phase_loss

    loss.backward()
    optim.step()
    print(loss.item())
    return encoded, decoded, r_phase


@readme
class ClassicalFeatureBasedLoss(object):
    def __init__(self, overfit, batch_size):
        super().__init__()
        self.overfit = overfit
        self.batch_size = batch_size

        self.orig = None
        self.recon = None
        self.spec = None

    def real(self):
        return playable(self.orig, samplerate)

    def fake(self):
        return playable(self.recon, samplerate)
    
    def s(self):
        return self.spec[0].data.cpu().numpy().T

    def run(self):
        stream = audio_stream(
            self.batch_size,
            n_samples,
            self.overfit,
            normalize=False,
            as_torch=True)

        for item in stream:
            self.orig = item
            e, d, spec = train_ae(item)
            self.recon = d
            self.spec = spec
