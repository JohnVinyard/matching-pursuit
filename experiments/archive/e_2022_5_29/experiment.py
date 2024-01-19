from matplotlib.pyplot import step
import torch
from torch import nn
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from old_stuff.ddsp import overlap_add
from train.optim import optimizer
from util import playable
from util.readmedocs import readme
import zounds
import numpy as np
from torch.nn import functional as F

from util.weight_init import make_initializer

init_weights = make_initializer(0.1)

mel_scale = MelScale()
codec = AudioCodec(mel_scale)


def feature(x):
    x = x.view(-1, n_samples)
    x = codec.to_frequency_domain(x)[..., 0]
    return x


samplerate = zounds.SR22050()
n_samples = 2 ** 15
window_size = 512
step_size = window_size // 2
channels = 128
n_frames = n_samples // step_size
n_frames_to_drop = 16

class ComplexLayer(nn.Module):
    def __init__(self, input_size, channels):
        super().__init__()
        self.input_size = input_size
        self.channels = channels

        # filters are in the frequency domain
        self.analysis_filters = nn.Parameter(
            torch.complex(
                torch.zeros(self.input_size, self.channels).normal_(0, 1),
                torch.zeros(self.input_size, self.channels).normal_(0, 1)
            )
        )

    def forward(self, x):
        batch, time, channels = x.shape
        real = x @ self.analysis_filters.real
        imag = x @ self.analysis_filters.imag
        x = torch.complex(real, imag)
        return x.real


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis = ComplexLayer(window_size, channels)
        self.register_buffer('window', torch.hamming_window(window_size))
        self.reduce = LinearOutputStack(
            channels, 2, out_channels=channels, in_channels=channels + 33)
        layer = nn.TransformerEncoderLayer(
            channels, 4, dim_feedforward=channels, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 8)
        self.synthesis = ComplexLayer(channels, window_size)
        self.apply(init_weights)

    def forward(self, x, drop_slice=None):

        
        x = x.view(-1, 1, n_samples)
        x = torch.cat([x, torch.zeros(x.shape[0], 1, step_size)], dim=-1)
        x = x.unfold(-1, window_size, step_size) * \
            self.window[None, None, None, :]
        x = x.view(x.shape[0], -1, window_size)
        orig = x.clone()


        if drop_slice is not None:
            z = torch.zeros_like(x)
            z[:] = x
            z[:, drop_slice, :] = 0
            x = z

        x = self.analysis(x)

        pos = pos_encoded(x.shape[0], x.shape[1], 16, device=x.device)

        x = torch.cat([x, pos], dim=-1)
        x = self.reduce(x)

        x = self.transformer(x)
        x = self.synthesis(x)

        orig[:, drop_slice, :] = x[:, drop_slice, :]
        x = orig

        x = overlap_add(x[:, None, :, :], apply_window=False)
        return z, x[..., :n_samples]


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis = ComplexLayer(window_size, channels)
        self.register_buffer('window', torch.hamming_window(window_size))
        self.reduce = LinearOutputStack(
            channels, 2, out_channels=channels, in_channels=channels + 33)
        layer = nn.TransformerEncoderLayer(
            channels, 4, dim_feedforward=channels, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, 8)
        self.judge = LinearOutputStack(channels, 2, out_channels=1)
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, n_samples)
        x = torch.cat([x, torch.zeros(x.shape[0], 1, step_size)], dim=-1)
        x = x.unfold(-1, window_size, step_size) * \
            self.window[None, None, None, :]
        x = x.view(x.shape[0], -1, window_size)

        x = self.analysis(x)

        pos = pos_encoded(x.shape[0], x.shape[1], 16, device=x.device)

        x = torch.cat([x, pos], dim=-1)
        x = self.reduce(x)

        x = self.transformer(x)
        x = self.judge(x)
        return x


model = Model()
optim = optimizer(model, lr=1e-4)

disc = Discriminator()
disc_optim = optimizer(disc, lr=1e-4)

def train_disc(batch):
    disc_optim.zero_grad()
    optim.zero_grad()
    b, _, time = batch.shape
    drop_start = np.random.randint(0, n_frames - n_frames_to_drop)
    drop_end = drop_start + n_frames_to_drop
    degraded, output = model.forward(batch, drop_slice=slice(drop_start, drop_end))

    rj = disc(batch)
    fj = disc(output)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())
    

def train_model(batch):
    optim.zero_grad()
    b, _, time = batch.shape
    drop_start = np.random.randint(0, n_frames - n_frames_to_drop)
    drop_end = drop_start + n_frames_to_drop
    degraded, output = model.forward(batch, drop_slice=slice(drop_start, drop_end))

    j = disc(output)
    loss = least_squares_generator_loss(j)

    fake_spec = feature(output)
    real_spec = feature(batch)
    # loss = F.mse_loss(fake_spec, real_spec) + loss
    loss.backward()
    optim.step()
    print('G', loss.item())
    return degraded, batch, real_spec, output, fake_spec


@readme
class ComplexTransformerExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        self.real_spec = None
        self.fake_spec = None
        self.degraded = None

    def rs(self):
        return self.real_spec.data.cpu().numpy().squeeze()[0]

    def fs(self):
        return self.fake_spec.data.cpu().numpy().squeeze()[0]

    def r(self):
        return playable(self.real, samplerate)

    def f(self):
        return playable(self.fake, samplerate)

    def d(self):
        return self.degraded.data.cpu().numpy().squeeze()[0]

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            if i % 2 == 0:
                self.degraded, self.real, self.real_spec, self.fake, self.fake_spec = train_model(
                    item)
            else:
                train_disc(item)
