import torch
from torch import nn
from modules.atoms import unit_norm
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from util.weight_init import make_initializer


def activation(x):
    return torch.sin(x * 30)


class RecurrentSynth(nn.Module):
    def __init__(self, layers, channels, is_silent=False, silent_factor=1e-6):
        super().__init__()
        self.net = LinearOutputStack(
            channels, layers, activation=activation)
        self.gate = LinearOutputStack(
            channels, layers, out_channels=1, activation=activation)
        self.is_silent = is_silent
        self.silent_factor = silent_factor

    def forward(self, x, max_iter=10):
        results = []
        x = unit_norm(x)
        for i in range(max_iter):
            x = self.net(x)
            x = unit_norm(x)
            results.append(x[..., None])

        x = torch.cat(results, dim=-1)

        z = x.permute(0, 2, 1)
        z = torch.relu(self.gate(z))
        z = z.permute(0, 2, 1)

        if self.is_silent:
            z = z * self.silent_factor

        return x, z

# (batch, voices, time, channels)


class Conductor(nn.Module):
    def __init__(self, layers, channels, voices):
        super().__init__()
        self.layers = layers
        self.channels = channels
        self.voices = voices

        instruments = \
            [RecurrentSynth(layers, channels) for _ in range(self.voices)] \
            + [(RecurrentSynth(layers, channels, is_silent=True))]
        
        self.instruments = nn.ParameterList(instruments)

        self.net = LinearOutputStack(channels, layers)
        self.router = LinearOutputStack(
            channels, layers, out_channels=self.voices + 1)
        
        # self.osc = OscillatorBank(
        #     input_channels=128,
        #     n_osc=128,
        #     n_audio_samples=n_samples,
        #     activation=torch.sigmoid,
        #     amp_activation=torch.abs,
        #     return_params=True,
        #     constrain=True,
        #     log_frequency=False,
        #     lowest_freq=40 / samplerate.nyquist,
        #     sharpen=False,
        #     compete=False)

        # self.noise = NoiseModel(
        #     input_channels=128,
        #     input_size=128,
        #     n_noise_frames=512,
        #     n_audio_samples=n_samples,
        #     channels=128,
        #     activation=lambda x: x,
        #     squared=False,
        #     mask_after=1,
        #     return_params=True)

    def forward(self, x):
        batch, time, channels = x.shape
        if batch != 1:
            raise NotImplementedError('Batch mode not implemented')
        
        x = x.view(time, channels)

        x = self.net(x)

        routes = self.router(x)
        routes = torch.softmax(routes, dim=-1)
        indices = torch.argmax(routes, dim=1)

        for i in range(time):
            r = routes[indices[i]]

            if r == self.voices:
                continue

            inst = self.instruments[r]
            params, env = inst.forward(x[i], time - i)
            # TODO: Pass params to synthesizer and apply envelope =)-hi
