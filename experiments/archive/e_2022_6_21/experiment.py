from torch import nn
import zounds
import torch
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import overlap_add

from modules.linear import LinearOutputStack
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
import numpy as np


samplerate = zounds.SR22050()
n_samples = 2 ** 15
model_dim = 128
n_heads = 4
n_layers = 6
window_size = 512
step_size = 256
sequence_length = n_samples // step_size
n_frames_to_predict = 16
n_samples_to_predict = n_frames_to_predict * step_size

init_weights = make_initializer(0.1)


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
        real = x @ self.analysis_filters.real
        imag = x @ self.analysis_filters.imag
        x = torch.complex(real, imag)
        return x.real


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos = LinearOutputStack(model_dim, 3, in_channels=33)
        self.frame = LinearOutputStack(model_dim, 3, in_channels=window_size)
        self.reduce = LinearOutputStack(
            model_dim, 3, in_channels=model_dim * 2)

        self.register_buffer('window', torch.hamming_window(
            window_size)[None, None, None, :])

        layer = nn.TransformerEncoderLayer(
            model_dim, n_heads, model_dim, batch_first=True)
        layer.norm1 = NoOp()
        layer.norm2 = NoOp()
        self.net = nn.TransformerEncoder(layer, n_layers)
        self.judge = LinearOutputStack(model_dim, 3, out_channels=1)
        self.apply(init_weights)

    def forward(self, x):
        x = torch.cat([x, torch.zeros(x.shape[0], 1, step_size)], dim=-1)
        x = x.unfold(-1, window_size, step_size) * self.window
        x = self.frame(x)

        pos = pos_encoded(x.shape[0], sequence_length, 16, device=x.device)
        pos = self.pos(pos)

        x = torch.cat([x.view(-1, sequence_length, model_dim), pos], dim=-1)
        x = self.reduce(x)

        x = self.net.forward(x)
        x = self.judge(x)
        return x


class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos = LinearOutputStack(model_dim, 3, in_channels=33)
        self.frame = LinearOutputStack(model_dim, 3, in_channels=window_size)
        self.reduce = LinearOutputStack(
            model_dim, 3, in_channels=model_dim * 2)

        self.register_buffer('window', torch.hamming_window(
            window_size)[None, None, None, :])

        layer = nn.TransformerEncoderLayer(
            model_dim, n_heads, model_dim, batch_first=True)
        layer.norm1 = NoOp()
        layer.norm2 = NoOp()
        self.net = nn.TransformerEncoder(layer, n_layers)
        self.to_samples = LinearOutputStack(
            model_dim, 3, out_channels=window_size)

        self.to_samples = LinearOutputStack(
            model_dim, 3, out_channels=window_size)

        self.apply(init_weights)

    def forward(self, x):

        x = torch.cat([x, torch.zeros(x.shape[0], 1, step_size)], dim=-1)
        x = x.unfold(-1, window_size, step_size) * self.window
        x = self.frame(x)

        pos = pos_encoded(x.shape[0], sequence_length, 16, device=x.device)
        pos = self.pos(pos)

        x = torch.cat([x.view(-1, sequence_length, model_dim), pos], dim=-1)
        x = self.reduce(x)

        x = self.net.forward(x)
        x = self.to_samples(x)
        x = overlap_add(x[:, None, :, :], apply_window=True)[..., :n_samples]
        return x


gen = Generator().to(device)
gen_optim = optimizer(gen)

disc = Discriminator().to(device)
disc_optim = optimizer(disc)


def train_gen(batch):
    gen_optim.zero_grad()

    batch = batch.clone()
    batch[:, :, -n_samples_to_predict:] = 0

    pred = gen.forward(batch)
    # only consider the final judgement

    combined = torch.cat([
        batch[:, :, :-n_samples_to_predict],
        pred[:, :, -n_samples_to_predict:]
    ], dim=-1)

    j = disc.forward(combined)[:, -n_frames_to_predict:, :]
    loss = -torch.mean(j)
    loss.backward()
    gen_optim.step()
    return loss, combined


def train_disc(batch):
    disc_optim.zero_grad()

    degraded = batch.clone()
    degraded[:, :, -n_samples_to_predict:] = 0

    pred = gen.forward(degraded)

    combined = torch.cat([
        batch[:, :, :-n_samples_to_predict],
        pred[:, :, -n_samples_to_predict:]
    ], dim=-1)

    # only consider the final judgements
    fj = disc.forward(combined)[:, -n_frames_to_predict:, :]

    rj = disc.forward(batch)[:, -n_frames_to_predict:, :]
    loss = -(torch.mean(rj) - torch.mean(fj))
    loss.backward()
    disc_optim.step()
    for p in disc.parameters():
        try:
            p.data.clamp_(-0.01, 0.01)
        except:
            continue
    return loss


@readme
class AdversarialTransformerExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.pred = None

    def listen(self):
        return playable(self.pred, samplerate)

    def spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        gen_loss = torch.zeros(1)
        disc_loss = torch.zeros(1)

        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)

            if i % 2 == 0:
                gen_loss, self.pred = train_gen(item)
            else:
                disc_loss = train_disc(item)

            if i % 10 == 0:
                print('GEN', gen_loss.item())
                print('DISC', disc_loss.item())
