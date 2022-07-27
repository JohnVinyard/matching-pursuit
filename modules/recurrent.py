import torch
from torch import nn
from modules.atoms import unit_norm
from modules.ddsp import noise_bank2
from modules.linear import LinearOutputStack
from modules.reverb import NeuralReverb
from util.weight_init import make_initializer
import numpy as np
from torch.nn import functional as F

init_weights = make_initializer(0.05)


def activation(x):
    # return torch.sin(x * 30)
    return F.leaky_relu(x, 0.2)


class Synth(nn.Module):
    def __init__(self, layers, channels, samples_per_frame, n_osc=64):
        super().__init__()
        self.layers = layers
        self.channels = channels
        self.samples_per_frame = samples_per_frame
        self.n_coeffs = ((samples_per_frame * 2) // 2) + 1

        self.to_noise_params = LinearOutputStack(
            channels, layers, out_channels=self.n_coeffs)
        self.n_osc = n_osc

        # (batch, time, channels, 2)
        self.to_osc_params = LinearOutputStack(
            channels, layers, out_channels=2 * self.n_osc)

    def forward(self, x):
        batch, time, channels = x.shape
        if batch != 1:
            raise NotImplementedError('Only batch size of 1 is supported')

        noise = self.to_noise_params(x).permute(0, 2, 1)
        noise = noise_bank2(noise)

        osc = self.to_osc_params(x).view(batch, time, self.n_osc, 2)
        amp = torch.norm(osc, dim=-1).permute(0, 2, 1)
        freq = (torch.angle(torch.complex(
            osc[..., 0], osc[..., 1])) / np.pi).permute(0, 2, 1)

        freq = (freq * .98) + 0.0036

        amp = F.interpolate(
            amp, size=self.samples_per_frame * time, mode='linear')
        freq = F.interpolate(
            freq, size=self.samples_per_frame * time, mode='linear')

        osc = torch.sin(torch.cumsum(freq * np.pi, dim=-1)) * amp
        osc = torch.sum(osc, dim=1, keepdim=True)
        signal = osc + noise
        return signal


class RecurrentSynth(nn.Module):
    def __init__(self, layers, channels, samples_per_frame):
        super().__init__()
        self.net = LinearOutputStack(
            channels, layers, activation=activation)
        self.gate = LinearOutputStack(channels, layers, out_channels=2)

        mask = torch.zeros(2, channels)
        mask[0, :] = 1

        self.samples_per_frame = samples_per_frame

        self.synth = Synth(layers, channels, samples_per_frame, n_osc=64)

        self.register_buffer('mask', mask)

    def forward(self, x, max_iter=10):
        results = []
        x = unit_norm(x)

        for i in range(max_iter):
            x = self.net(x)
            x = unit_norm(x)

            g = self.gate(x)
            g = torch.softmax(g, dim=-1)
            mask = g @ self.mask
            x = x * mask

            results.append(x[None, ...])

            index = torch.argmax(g, dim=-1)
            if index.item() == 1:
                break

        x = torch.cat(results, dim=0)[None, ...]

        signal = self.synth.forward(x).view(-1)

        return signal


class Silence(nn.Module):
    def __init__(self, samples_per_frame):
        super().__init__()
        self.samples_per_frame = samples_per_frame
        self.register_buffer('silence', torch.zeros(
            samples_per_frame).fill_(1e-8))

    def forward(self, x, max_iter=10):
        samples = torch.cat([self.silence for _ in range(max_iter)], dim=0)
        return samples


class Conductor(nn.Module):
    def __init__(self, layers, channels, voices, samples_per_frame, total_frames):
        super().__init__()
        self.layers = layers
        self.channels = channels
        self.voices = voices
        self.samples_per_frame = samples_per_frame
        self.total_frames = total_frames

        instruments = [RecurrentSynth(
            layers, channels, samples_per_frame) for _ in range(self.voices)]

        self.instruments = nn.ModuleList(instruments)

        self.net = LinearOutputStack(channels, layers)
        self.router = LinearOutputStack(
            channels, layers, out_channels=self.voices)

        n_rooms = 8
        self.to_rooms = LinearOutputStack(
            channels, layers, out_channels=n_rooms)
        self.to_mix = LinearOutputStack(channels, layers, out_channels=1)

        self.verb = NeuralReverb(
            self.total_frames * self.samples_per_frame, n_rooms)

        self.apply(init_weights)

    def forward(self, x):
        batch, time, channels = x.shape
        if batch != 1:
            raise NotImplementedError('Batch mode not implemented')

        total_samples = self.samples_per_frame * self.total_frames

        out_samples = torch.zeros(total_samples)

        x = x.view(time, channels)

        x = self.net(x)

        z = torch.mean(x, dim=0, keepdim=True)
        rooms = torch.softmax(self.to_rooms(z), dim=-1)
        mix = torch.sigmoid(self.to_mix(z))

        routes = self.router(x)
        # get probabilities for each instrument at each time step
        routes = torch.softmax(routes, dim=-1)
        # get the winning instrument at each time step
        indices = torch.argmax(routes, dim=-1)

        for i in range(time):
            # get the winning instrument index a this time step
            index = indices[i].item()

            # get the factor used to scale the output from the instrument
            r = routes[i, index]
            inst = self.instruments[index]

            # run the instrument and scale it by the probability
            max_iter = time - i
            inst_output = inst.forward(x[i], max_iter) * r

            if len(inst_output.shape) == 1:
                start_sample = i * self.samples_per_frame
                end_sample = start_sample + inst_output.shape[0]

                if end_sample >= total_samples:
                    slce = out_samples[start_sample:end_sample]
                    size = slce.shape[0]
                    out_samples[start_sample:end_sample] = out_samples[start_sample:end_sample] + \
                        inst_output[:size]
                else:
                    out_samples[start_sample:end_sample] = out_samples[start_sample:end_sample] + inst_output

        wet = self.verb(out_samples, rooms)
        dry = out_samples

        out_samples = (dry * mix) + (wet * (1 - mix))
        return out_samples


class SerialGenerator(nn.Module):
    def __init__(self, channels, frames, strides, activation=lambda x: F.leaky_relu(x, 0.2), max_length=512):
        super().__init__()

        self.strides = strides
        self.channels = channels
        self.frames = frames
        self.latent_embedding = LinearOutputStack(
            channels, 3, activation=activation)
        self.context_embedding = LinearOutputStack(
            channels, 3, activation=activation)

        self.down = LinearOutputStack(channels, 3, in_channels=channels * len(strides))
        self.strided = nn.ModuleDict(
            {str(k): LinearOutputStack(channels, 3, activation=activation) for k in strides})
        self.generator = LinearOutputStack(
            channels, 3, out_channels=channels * frames, activation=activation, in_channels=channels*2)
        
        self.register_buffer('env', torch.linspace(0, 1, max_length) ** 2)

    def forward(self, z, seq=None):
        # n = torch.norm(z, dim=-1, keepdim=True)
        # z = z / (n + 1e-8)

        latent = self.latent_embedding(z)
        latent = latent + z

        if seq is None:
            seq = torch.zeros(z.shape[0], 1, self.channels).to(latent.device)
            orig_seq = seq
        else:
            orig_seq = seq
            summaries = {}

            for stride in self.strides:
                env_seg = self.env[-seq.shape[0]:]
                s = seq * env_seg[None, :, None]
                s, _ = s[:, ::stride, :].max(dim=1, keepdim=True)
                s = self.strided[str(stride)].forward(s)
                summaries[stride] = s

            seq = torch.cat(list(summaries.values()), dim=-1)
            seq = self.down(seq)

        seq = self.context_embedding(seq)

        x = torch.cat([latent.view(z.shape[0], 1, self.channels), seq], dim=-1)

        x = self.generator(x)

        x = x.reshape(z.shape[0], self.frames, self.channels)

        x = torch.cat([orig_seq, x], dim=1)

        return latent, x
