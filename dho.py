import numpy as np
import torch
from torch import nn

from modules import interpolate_last_axis, stft, sparsify
from modules.transfer import damped_harmonic_oscillator
from modules.upsample import upsample_with_holes
from util import make_initializer
from util.overfit import overfit_model

init_weights = make_initializer(0.02)



class DampedHarmonicOscillatorBlock(nn.Module):

    def __init__(self, n_samples: int, control_rate: int, n_oscillators: int):
        super().__init__()
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_oscillators = n_oscillators
        self.n_frames = int(n_samples / control_rate)
        print('FRAMES', self.n_frames)

        self.mass = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-6, 6))

        self.base_damping = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-6, 6))
        self.damping = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-0.01, 0.01))

        self.base_tension = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(4, 9))
        self.tension = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-0.01, 0.01))

        self.initial_displacement = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-0.01, 0.01))




    def forward(self, t: torch.Tensor, tension_modifier: torch.Tensor = None, influence: torch.Tensor = None):



        # print(self.base_damping.shape, self.damping.shape)
        damping = interpolate_last_axis(self.base_damping + self.damping, desired_size=self.n_samples)
        tension = interpolate_last_axis(self.base_tension + self.tension, desired_size=self.n_samples)

        if tension_modifier is not None:
            tension = tension + (tension_modifier * influence)

        # print(t.shape, self.mass.shape, damping.shape, tension.shape, self.initial_displacement.shape)
        x = damped_harmonic_oscillator(
            time=t,
            mass=torch.sigmoid(self.mass) * 2,
            damping=torch.sigmoid(damping) * 30,
            tension=10 ** torch.abs(tension),
            initial_displacement=self.initial_displacement,
            initial_velocity=0,
            do_clamp=False
        )
        # x = torch.sum(x, dim=0, keepdim=True)
        return x

class DampedHarmonicOscillatorController(nn.Module):
    def __init__(self, n_samples: int, control_rate: int, n_oscillators: int):
        super().__init__()
        self.n_samples = n_samples
        self.control_rate = control_rate
        self.n_oscillators = n_oscillators
        self.n_frames = int(n_samples / control_rate)

        self.max_time = 1
        self.time_step = self.max_time // n_samples
        self.register_buffer('base_time', torch.zeros(n_oscillators, 1, n_samples).fill_(self.time_step))

        self.times = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-0.001, 0.001))

        # self.t = nn.Parameter(torch.zeros(n_oscillators, 1, self.n_frames).uniform_(-0.001, 0.001))

        self.dho1 = DampedHarmonicOscillatorBlock(n_samples, control_rate, n_oscillators)
        self.dho2 = DampedHarmonicOscillatorBlock(n_samples, control_rate, n_oscillators)
        self.dho3 = DampedHarmonicOscillatorBlock(n_samples, control_rate, n_oscillators)
        self.influence = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-0.001, 0.001))
        self.influence2 = nn.Parameter(torch.zeros(n_oscillators, 1, 1).uniform_(-0.0001, 0.0001))

    def forward(self):

        time_modifier = upsample_with_holes(self.times, desired_size=self.n_samples)
        t = self.base_time + time_modifier
        t = torch.cumsum(t, dim=-1)
        t = torch.clamp(t, 0, self.max_time)
        # t = torch.sigmoid(interpolate_last_axis(self.t, desired_size=self.n_samples))


        x = self.dho1.forward(t)
        x = self.dho2.forward(t, x, self.influence)
        x = self.dho3.forward(t, x, self.influence2)

        x = torch.sum(x, dim=0, keepdim=True)
        return x


def compute_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    t = stft(target, 2048, 256, pad=True)
    r = stft(recon, 2048, 256, pad=True)
    return torch.abs(t - r).sum()

if __name__ == '__main__':
    n_samples = 2 ** 17

    overfit_model(
        n_samples=n_samples,
        model=DampedHarmonicOscillatorController(
            n_samples=n_samples,
            control_rate=256,
            n_oscillators=32),
        loss_func=compute_loss,
        collection_name='dho',
        learning_rate=1e-3
    )
