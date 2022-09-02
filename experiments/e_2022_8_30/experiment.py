import torch
from torch import nn
import zounds
from modules.atoms import unit_norm
from modules.ddsp import overlap_add
from modules.dilated import DilatedStack
from modules.phase import MelScale
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature
from modules.sparse import sparsify_vectors
from modules.stft import morlet_filter_bank, stft
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
event_dim = model_dim

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

# pif = PsychoacousticFeature([128] * 6).to(device)

aim = AuditoryImage(128, 64, do_windowing=False, check_cola=False)

# mel_scale = MelScale()


def perceptual_feature(x):
    # bands = pif.compute_feature_dict(x)
    # return torch.cat(list(bands.values()), dim=-2)
    x = fb.forward(x, normalize=False)
    x = aim.forward(x)
    return x
    

def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)


class AudioSegmentRenderer(object):
    def __init__(self):
        super().__init__()

    def render(self, x, indices):
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

        # encoder = nn.TransformerEncoderLayer(
        #     model_dim, 4, model_dim, batch_first=True)
        # self.context = nn.TransformerEncoder(
        #     encoder, 4, norm=UnitNorm())
        self.context = DilatedStack(model_dim, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)
        self.attend = nn.Linear(model_dim, 1)
        self.to_events = nn.Linear(model_dim, event_dim)
        self.env_factor = 32

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, n_samples)
        x = torch.abs(fb.convolve(x))
        x = fb.temporal_pooling(x, window_size, step_size)[..., :n_frames]
        
        pos = pos_encoded(
            batch, n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        # Transformer version
        # x = x.permute(0, 2, 1)
        # x = self.context(x)

        # Conv version
        x = self.context(x)
        x = x.permute(0, 2, 1)

        attn = torch.softmax(self.attend(x).view(batch, n_frames), dim=-1)
        x = x.permute(0, 2, 1)
        x, indices = sparsify_vectors(x, attn, n_events, normalize=False)
        x = self.to_events(x) 

        x = x.view(batch * n_events, model_dim)

        x = unit_norm(x, axis=-1)

        return x, indices


class SequenceGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # self.env = PosEncodedUpsample(
        #     latent_dim=model_dim,
        #     channels=model_dim,
        #     size=n_frames,
        #     out_channels=1,
        #     layers=5,
        #     concat=False,
        #     learnable_encodings=False,
        #     multiply=False,
        #     transformer=False)

        self.env = ConvUpsample(
            model_dim, model_dim, 4, n_frames, mode='nearest', out_channels=1, batch_norm=True)

        n_coeffs = window_size // 2 + 1
        self.n_coeffs = n_coeffs

        # self.transfer = PosEncodedUpsample(
        #     latent_dim=model_dim,
        #     channels=model_dim,
        #     size=n_samples,
        #     out_channels=1,
        #     layers=1,
        #     concat=False,
        #     learnable_encodings=True,
        #     multiply=False,
        #     transformer=False,
        #     filter_bank=True)

        
        self.transfer = ConvUpsample(
            model_dim, model_dim, 4, n_frames, mode='nearest', out_channels=n_coeffs * 2, batch_norm=True
        )

        

    def forward(self, x):
        x = x.view(-1, model_dim)

        # TODO: envelope generator
        env = self.env(x) ** 2
        env = F.interpolate(env, size=n_samples, mode='linear')
        noise = torch.zeros(1, 1, n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise


        tf = self.transfer(x)
        real = tf[:, :self.n_coeffs, :]
        imag = tf[:, self.n_coeffs:, :]

        real = real * torch.cos(imag)
        imag = real * torch.sin(imag)

        tf = torch.complex(real, imag)
        tf = torch.fft.irfft(tf, dim=1, norm='ortho')
        tf = tf.permute(0, 2, 1).view(-1, 1, n_frames, window_size) * torch.hamming_window(window_size, device=tf.device)[None, None, None, :]

        # TODO: Option to cut off
        tf = overlap_add(tf)[..., :n_samples]
        
        

        env_spec = torch.fft.rfft(env, dim=-1, norm='ortho')
        tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
        spec = env_spec * tf_spec
        final = torch.fft.irfft(spec, dim=-1, norm='ortho')

        return final


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.audio_renderer = AudioSegmentRenderer()
        self.gen = SequenceGenerator()
        self.apply(init_weights)

    def forward(self, x):
        x, indices = self.summary(x)
        x = self.gen(x)
        x = self.audio_renderer.render(x, indices)

        # mx, _ = torch.max(x, dim=-1, keepdim=True)
        # x = x / (mx + 1e-8)

        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
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
