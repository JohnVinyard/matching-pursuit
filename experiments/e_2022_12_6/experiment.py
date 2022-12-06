import zounds
from config.experiment import Experiment
from modules import stft
from train.optim import optimizer
from util import playable
from util.readmedocs import readme
import torch
from torch import nn
from torch.nn import functional as F

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class Model(nn.Module):
    def __init__(
            self,
            n_samples,
            samples_per_block,
            room_dim,
            mic_location,
            impulse_envelope_size):

        super().__init__()
        self.samples_per_block = samples_per_block
        self.room_dim = room_dim
        self.n_blocks = n_samples // samples_per_block
        self.mic_location = mic_location
        self.impulse_envelope_size = impulse_envelope_size

        self.n_transfer_coeffs = samples_per_block // 2 + 1

        self.transfer_functions = nn.Parameter(
            torch.zeros(self.n_blocks, *self.room_dim, self.n_transfer_coeffs).uniform_(0, 1))

        self.impulses = nn.Parameter(
            torch.zeros(self.n_blocks, *self.room_dim, self.impulse_envelope_size).uniform_(0, 1))

        self.register_buffer('kernel', torch.ones(3, 3, 3))

    def forward(self, x):
        output_buffer = []

        sound_field = torch.zeros(*self.room_dim, self.samples_per_block)

        for i in range(self.n_blocks):

            # calculate impulses
            envelopes = self.impulses[i].view(-1, 1, self.impulse_envelope_size)
            envelopes = F.interpolate(
                envelopes, size=self.samples_per_block, mode='linear')
            envelopes = envelopes.view(*self.room_dim, self.samples_per_block)
            envelopes = torch.clamp(envelopes, 0, 1)
            noise = torch.zeros_like(envelopes).uniform_(-1, 1)
            envelopes = envelopes * noise

            # add impulses to sound field
            sound_field = sound_field + envelopes

            # apply the resonance functions
            resonance = self.transfer_functions[i]
            resonance = torch.clamp(resonance, 0, 1)
            spec = torch.fft.rfft(sound_field, dim=-1, norm='ortho')
            filtered = resonance * spec
            filtered = torch.fft.irfft(filtered, dim=-1, norm='ortho')

            sound_field = filtered

            output_buffer.append(sound_field[self.mic_location])

            # propagate
            windowed = F.unfold(
                sound_field[None, ...],
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                stride=(1, 1, 1))
            print(windowed.shape)

            sound_field = (windowed * self.kernel).sum(dim=(-1, -2, -3))

        return torch.cat(output_buffer).view(-1, 1, self.n_samples)


model = Model(exp.n_samples, 64, (32, 32, 32), (16, 16, 16), 8)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    real = stft(batch, 512, 256, pad=True)
    fake = stft(recon, 512, 256, pad=True)
    loss = F.mse_loss(fake, real)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class RoomSimulationExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None
        self.fake = None

    def orig(self):
        return playable(self.real, exp.samplerate)

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, self.fake = train(item)
            print(l.item())
