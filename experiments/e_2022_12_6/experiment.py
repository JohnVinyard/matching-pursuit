import zounds
from config.experiment import Experiment
from modules import stft
from modules.ddsp import overlap_add
from modules.normalization import max_norm
from train.optim import optimizer
from util import playable
from util.readmedocs import readme
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.jit._script import ScriptModule, script_method

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def activation(x, scale=1):
    x = torch.clamp(x, 0, 1)
    return x * scale

class Model(nn.Module):
    def __init__(
            self,
            n_samples,
            samples_per_block,
            room_dim,
            mic_location,
            impulse_envelope_size):

        super().__init__()

        self.n_samples = n_samples
        self.samples_per_block = samples_per_block
        self.room_dim = room_dim
        self.n_blocks = n_samples // (samples_per_block)
        self.mic_location = mic_location
        self.impulse_envelope_size = impulse_envelope_size

        self.n_transfer_coeffs = samples_per_block // 2 + 1

        self.transfer_functions = nn.Parameter(
            torch.zeros(self.n_blocks, *self.room_dim, self.n_transfer_coeffs).uniform_(0, 1))

        self.impulses = nn.Parameter(
            torch.zeros(self.n_blocks, *self.room_dim, self.impulse_envelope_size).uniform_(0, 1e-3))

        self.register_buffer('kernel', torch.ones(3, 3))

    def forward(self, x):
        output_buffer = []

        sound_field = torch.zeros(*self.room_dim, self.samples_per_block)


        for i in range(self.n_blocks):

            # calculate impulses
            envelopes = self.impulses[i].view(-1, 1, self.impulse_envelope_size)
            envelopes = F.interpolate(
                envelopes, size=self.samples_per_block, mode='linear')
            envelopes = envelopes.view(*self.room_dim, self.samples_per_block)
            envelopes = activation(envelopes)
            noise = torch.zeros_like(envelopes).uniform_(-1.0, 1.0)
            envelopes = envelopes * noise
            # envelopes = envelopes * torch.hamming_window(self.samples_per_block)[None, None, :]
            # print(envelopes.max().item())

            # add impulses to sound field
            # sound_field = sound_field + envelopes

            # apply the resonance functions
            resonance = self.transfer_functions[i]
            resonance = activation(resonance)
            spec = torch.fft.rfft(envelopes, dim=-1, norm='ortho')
            filtered = resonance * spec
            filtered = torch.fft.irfft(filtered, dim=-1, norm='ortho')
            # print(filtered.max().item())

            sound_field = filtered + sound_field
            # print(sound_field.max().item())

            output_buffer.append(sound_field[self.mic_location[0], self.mic_location[1]][None, ...])

            # propagate
            sf = sound_field.permute(2, 0, 1).view(-1, self.samples_per_block, *self.room_dim)
            sf = F.pad(sf, (1, 1, 1, 1), mode='reflect')
            windowed = F.unfold(
                sf,
                kernel_size=(3, 3),
                # padding=(1, 1),
                stride=(1, 1)).view(self.samples_per_block, 3, 3, *self.room_dim)

            sound_field = (windowed * self.kernel[None, :, :, None, None]).mean(dim=(1, 2))
            sound_field = sound_field.permute(1, 2, 0)
            # print(sound_field.max().item())

            # input('next...')
            # print('=================================')

        final = torch.cat(output_buffer, dim=0)
        # final = final * torch.hamming_window(self.samples_per_block)[None, ...]
        # final = final.view(1, 1, -1, self.samples_per_block)
        # final = overlap_add(final, apply_window=False).view(1, 1, -1)[..., :self.n_samples]
        final = final.view(1, 1, -1)[..., :self.n_samples]
        # final = max_norm(final, dim=-1, epsilon=1e-12)
        return final


model = Model(exp.n_samples, 512, (16, 16), (4, 4), 8)
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
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, exp.samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, self.fake = train(item)
            print(i, l.item())
