import torch
from torch import nn
import zounds
from modules.ddsp import overlap_add
from modules.phase import MelScale
from modules.psychoacoustic import PsychoacousticFeature
from modules.sparse import sparsify_vectors
from modules.stft import stft
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from modules.linear import LinearOutputStack

from util import device, playable
from util import make_initializer
from torch.nn import functional as F
from modules import pos_encoded

from util.readmedocs import readme
import numpy as np

samplerate = zounds.SR22050()
n_samples = 2 ** 15

window_size = 512
step_size = window_size // 2
n_frames = n_samples // step_size

n_bands = 128
kernel_size = 512
model_dim = 128

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


init_weights = make_initializer(0.1)

pif = PsychoacousticFeature([128] * 6).to(device)

mel_scale = MelScale()


def perceptual_feature(x):
    bands = pif.compute_feature_dict(x)
    return torch.cat(list(bands.values()), dim=-2)


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)


class UnitNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norms + 1e-8)
        return x


class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self):
        super().__init__()

        encoder = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(
            encoder, 4, norm=UnitNorm())
        self.reduce = nn.Linear(33 + model_dim, model_dim)
        self.to_latent = LinearOutputStack(model_dim, 3)
        self.decoder = PosEncodedUpsample(
            latent_dim=model_dim,
            channels=model_dim,
            size=n_samples,
            out_channels=1,
            layers=4,
            concat=False,
            learnable_encodings=False,
            multiply=False,
            transformer=False,
            filter_bank=True)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, n_samples)
        x = torch.abs(fb.convolve(x))
        x = fb.temporal_pooling(x, window_size, step_size)[..., :n_frames]
        pos = pos_encoded(
            batch, n_frames, 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.reduce(x)
        x = self.context(x)
        # x, _ = torch.max(x, dim=-1)
        x = x[:, -1, :]
        x = self.to_latent(x)
        x = self.decoder(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(init_weights)

    def forward(self, x):
        x = self.summary(x)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class NerfExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None

    def orig(self):
        return playable(self.real, samplerate, normalize=True)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, samplerate, normalize=True)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)

            self.real = item
            loss, self.fake = train_model(item)
            print('GEN', loss.item())
