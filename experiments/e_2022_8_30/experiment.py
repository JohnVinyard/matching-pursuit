import torch
from torch import nn
import zounds
from modules.phase import MelScale
from modules.psychoacoustic import PsychoacousticFeature
from modules.sparse import sparsify_vectors
from modules.stft import stft
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample

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

n_events = 16

model_dim = 128
event_dim = 128

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

pif = PsychoacousticFeature().to(device)

mel_scale = MelScale()

def perceptual_feature(x):
    # return stft(x, 512, 256, log_amplitude=True)
    bands = pif.compute_feature_dict(x)
    return torch.cat(list(bands.values()), dim=-1)

    # x = torch.abs(mel_scale.to_frequency_domain(x))
    # x = fb.forward(x, normalize=False)
    # x = fb.temporal_pooling(x, 512, 256)
    # return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)


class AudioSegmentRenderer(object):
    def __init__(self):
        super().__init__()

    def render(self, x, params, indices):
        x = x.view(-1, n_events, n_samples)
        batch = x.shape[0]

        times = indices * step_size

        output = torch.zeros(batch, 1, n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(n_events):
                time = times[b, i]
                output[b, :, time: time + n_samples] += x[b, i][None, :]

        output = output[..., :n_samples]
        return output


class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self):
        super().__init__()

        encoder = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True, activation=F.gelu)
        self.context = nn.TransformerEncoder(
            encoder, 6, norm=None)

        self.reduce = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)

        self.attend = nn.Linear(model_dim, 1)
        self.to_events = nn.Linear(model_dim, event_dim)

        self.env_factor = 32

    def forward(self, x, add_noise=False):
        batch = x.shape[0]
        x = x.view(-1, 1, n_samples)
        x = torch.abs(fb.convolve(x))
        x = fb.temporal_pooling(x, window_size, step_size)[..., :n_frames]
        pos = pos_encoded(batch, n_frames, 16,
                          device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        x = x.permute(0, 2, 1)
        x = self.context(x)
        if self.disc:
            x = self.judge(x)
            x = torch.sigmoid(x)
            return x

        attn = torch.softmax(self.attend(x).view(batch, n_frames), dim=-1)
        x = x.permute(0, 2, 1)
        x, indices = sparsify_vectors(x, attn, n_events, normalize=True)
        x = self.to_events(x)  # (batch, n_events, event_dim)

        # => (batch, n_events, n_frames, n_samples)

        return x


class Model(nn.Module):
    def __init__(self, disc=False):
        super().__init__()
        self.summary = Summarizer()
        self.audio_renderer = AudioSegmentRenderer()
        self.disc = disc

        self.encode = nn.Sequential(
            nn.Conv1d(1, 8, 11, 4, 5),  # 8192
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 16, 11, 4, 5),  # 2048
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 11, 4, 5),  # 512
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 11, 4, 5),  # 128
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 11, 4, 5),  # 32
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 11, 4, 5),  # 8
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 3, 2, 1),  # 4
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 512, 4, 4, 0)
        )

        self.up = PosEncodedUpsample(
            latent_dim=512,
            channels=model_dim,
            size=n_samples,
            out_channels=1,
            layers=5,
            concat=False,
            learnable_encodings=False,
            multiply=False,
            transformer=False)

        self.apply(init_weights)


    def forward(self, x):
        x = self.encode(x)
        x = self.up(x)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    recon_loss = perceptual_loss(recon, batch)
    loss = recon_loss
    loss.backward()
    optim.step()
    return loss, recon


@readme
class TransferFunctionExperiment(object):
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
