import torch
from torch import nn
import zounds
from config.dotenv import Config
from modules.ddsp import overlap_add
from modules.phase import MelScale
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from modules.sparse import sparsify_vectors
from modules.stft import stft
from modules.waveguide import waveguide_synth
from train.optim import optimizer
from upsample import ConvUpsample, Linear, PosEncodedUpsample
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

n_events = 4

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


event_resolution = 8
max_delay = 128
max_filter_size = 10


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


class SynthParamGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.event_dim = event_resolution * 4
        self.net = LinearOutputStack(model_dim, 5, out_channels=event_resolution * 4)
    
    def forward(self, x):
        x = self.net(x)

        x = torch.sigmoid(x)

        delays = x[:, :, :event_resolution]
        dampings = x[:, :, event_resolution: event_resolution * 2]
        filter_sizes = x[:, :, event_resolution * 2: event_resolution * 3]
        impulse = x[:, :, event_resolution * 3:]

        return delays, dampings, filter_sizes, impulse

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
        self.attend = nn.Linear(model_dim, 1)
        self.param_gen = SynthParamGenerator()


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

        attn = torch.softmax(self.attend(x).view(batch, n_frames), dim=-1)
        x = x.permute(0, 2, 1)
        x, indices = sparsify_vectors(x, attn, n_events, normalize=True)

        delays, dampings, filter_sizes, impulse = self.param_gen(x)

        return x, delays, dampings, filter_sizes, impulse, indices


class LossFunction(nn.Module):
    def __init__(self):
        super().__init__()
        encoder = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(
            encoder, 4, norm=UnitNorm())
        self.reduce = nn.Linear(33 + (model_dim * 2), model_dim)

        self.to_loss = LinearOutputStack(model_dim, 3, out_channels=257)
        self.apply(init_weights)
    
    def forward(self, params, indices, audio):
        batch = params.shape[0]

        audio = audio.view(-1, 1, n_samples)
        spec = torch.abs(fb.convolve(audio))
        spec = fb.temporal_pooling(spec, window_size, step_size)[..., :n_frames]
        spec = spec.permute(0, 2, 1)

        x = torch.zeros(batch, n_frames, model_dim)
        for b in range(batch):
            for i in range(n_events):
                index = indices[b, i]
                x[b, index] = params[b, i]
        
        pos = pos_encoded(
            batch, n_frames, 16, device=x.device)
        x = torch.cat([x, pos, spec], dim=-1)
        x = self.reduce(x)
        x = self.context(x)
        l = self.to_loss(x)
        return l


class NonDifferentiableSynth(object):
    def __init__(self):
        super().__init__()
    
    def render(self, delays, dampings, filter_sizes, impulse, indices):
        

        delays = delays.reshape((-1, 1, event_resolution))
        dampings = dampings.reshape((-1, 1, event_resolution))
        filter_sizes = filter_sizes.reshape((-1, 1, event_resolution))
        impulse = impulse.reshape((-1, 1, event_resolution))

        delays = F.interpolate(delays, size=n_samples, mode='linear').view(-1, n_events, n_samples)
        dampings = F.interpolate(dampings, size=n_samples, mode='linear').view(-1, n_events, n_samples)
        filter_sizes = F.interpolate(filter_sizes, size=n_samples, mode='linear').view(-1, n_events, n_samples)
        impulse = F.interpolate(impulse, size=n_samples, mode='linear').view(-1, n_events, n_samples)

        noise = torch.zeros_like(impulse).uniform_(-1, 1)
        impulse = impulse * noise

        delays = 1 + (delays * max_delay).long()
        filter_sizes = 1 + (filter_sizes * max_filter_size).long()

        delays = delays.data.cpu().numpy()
        dampings = dampings.data.cpu().numpy()
        filter_sizes = filter_sizes.data.cpu().numpy()
        impulse = impulse.data.cpu().numpy()

        batch = delays.shape[0]

        output = np.zeros((batch, n_samples * 2))

        for b in range(batch):
            for i in range(n_events):
                signal = waveguide_synth(impulse[b, i], delays[b, i], dampings[b, i], filter_sizes[b, i])
                start = indices[b, i].item() * 256
                end = start + n_samples
                output[b, start: end] += signal
        
        return output[:, :n_samples]
        




class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(init_weights)

    def forward(self, x):
        x = self.summary(x)
        return x

renderer = NonDifferentiableSynth()

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

loss_func = LossFunction().to(device)
loss_optim = optimizer(loss_func, lr=1e-3)


def train_model(batch):
    optim.zero_grad()
    params, delays, dampings, filter_sizes, impulse, indices = model.forward(batch)
    loss = loss_func.forward(params, indices, batch)
    loss = (loss ** 2).mean()

    loss.backward()
    optim.step()
    recon = renderer.render(delays, dampings, filter_sizes, impulse, indices)
    return loss, recon


def train_loss_func(batch):
    loss_optim.zero_grad()
    params, delays, dampings, filter_sizes, impulse, indices = model.forward(batch)
    recon = renderer.render(delays, dampings, filter_sizes, impulse, indices)

    recon = torch.from_numpy(recon).float().to(device).view(-1, 1, n_samples)

    pred_loss = loss_func.forward(params, indices, batch)

    real_spec = stft(batch, 512, 256, pad=True)
    fake_spec = stft(recon, 512, 256, pad=True)
    real_loss = fake_spec - real_spec

    loss = torch.abs(pred_loss - real_loss).sum()
    loss.backward()
    loss_optim.step()
    return loss


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

            if i % 2 == 0:
                self.real = item
                loss, self.fake = train_model(item)
                print('GEN', loss.item())
            else:
                loss = train_loss_func(item)
                print('LOSS', loss.item())
