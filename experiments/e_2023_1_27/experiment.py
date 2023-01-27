import torch
from torch import nn
from torch.nn import functional as F
import zounds
import numpy as np
from config.dotenv import Config
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_shift

from modules.linear import LinearOutputStack
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
from modules.sparse import VectorwiseSparsity
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.music import MusicalScale
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 8
n_harmonics = 16

class Atoms(nn.Module):
    def __init__(
            self,
            n_frames: int,
            n_samples: int,
            n_harmonics: int,
            layers: int,
            channels: int,
            samplerate: zounds.SampleRate,
            freq_change_factor: float = 0.1,
            activation=lambda x: F.leaky_relu(x, 0.2),
            discrete_activation=lambda x: F.gumbel_softmax(x, tau=1, hard=True, dim=-1)):

        super().__init__()
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.n_harmonics = n_harmonics
        self.layers = layers
        self.channels = channels
        self.freq_change_factor = freq_change_factor

        self.noise_filter_size = 256

        self.activation = activation
        self.discrete_activation = discrete_activation

        self.register_buffer(
            'harmonics', torch.linspace(1, n_harmonics + 1, steps=n_harmonics))

        # the total number of bands for spectral shape will be
        # n_harmonics + f0
        self.total_bands = self.n_harmonics + 1

        scale = MusicalScale()
        center_freqs \
            = np.array(list(scale.center_frequencies)) / samplerate.nyquist

        self.register_buffer('center_freqs', torch.from_numpy(center_freqs).float())

        self.f0 = LinearOutputStack(
            channels,
            layers,
            out_channels=len(scale),
            activation=activation)

        self.f0_change = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode='learned')

        self.amp = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode='learned')

        self.decay = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode='learned')

        self.mix = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode='learned')

        self.spec_shape = LinearOutputStack(
            channels,
            layers,
            out_channels=self.n_harmonics,
            activation=activation)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, self.channels)

        f0 = self.discrete_activation(self.f0.forward(x))
        f0 = (f0 @ self.center_freqs).view(-1, 1, 1)

        f0_change = torch.tanh(self.f0_change.forward(x))
        f0_change = (f0 * self.freq_change_factor) * f0_change

        freq = f0.view(-1, 1, 1) + f0_change
        harmonics = freq * self.harmonics[None, :, None]

        all_tones = torch.cat([freq, harmonics], dim=1)

        indices = torch.where(all_tones >= 1)
        all_tones[indices] = 0

        all_tones = F.interpolate(
            all_tones, size=self.n_samples, mode='linear')
        osc = torch.sin(torch.cumsum(all_tones * np.pi, dim=-1))

        noise = torch.zeros(batch, 1, self.n_samples).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')

        filt = osc[:, :, :256]
        filt = F.pad(filt, (0, self.n_samples - 256))
        filt_spec = torch.fft.rfft(filt, dim=-1, norm='ortho')

        filtered = noise_spec * filt_spec
        bl_noise = torch.fft.irfft(filtered, dim=-1, norm='ortho')

        mix = torch.sigmoid(self.mix.forward(x))
        mix = F.interpolate(mix, size=self.n_samples, mode='linear')

        osc_mix = mix
        noise_mix = 1 - mix

        amp = torch.sigmoid(self.amp.forward(x))
        factors = torch.sigmoid(self.spec_shape.forward(x)).view(-1, self.n_harmonics, 1) * amp
        
        amp = torch.cat([amp, factors], dim=1)

        decay = torch.sigmoid(self.decay.forward(x))

        amp_with_decay = torch.zeros_like(amp)

        for i in range(self.n_frames):
            if i == 0:
                amp_with_decay[:, :, i] = amp[:, :, i]
            else:
                amp_with_decay[:, :, i] = amp[:, :, i] + \
                    (amp[:, :, i - 1] * decay[:, :, i])

        amp_with_decay = F.interpolate(
            amp_with_decay, size=self.n_samples, mode='linear')

        full_osc = osc * amp_with_decay

        full_signal = (full_osc * osc_mix) + (bl_noise * noise_mix)

        full_signal = torch.sum(full_signal, dim=1, keepdim=True)
        return full_signal


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)
        
        self.atoms = Atoms(
            exp.n_frames, 
            exp.n_samples, 
            n_harmonics, 
            3, 
            exp.model_dim, 
            exp.samplerate)
        
        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)

        self.n_rooms = self.verb.n_rooms

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(
            exp.model_dim, 2, out_channels=self.n_rooms)

        self.apply(lambda x: exp.init_weights(x))


    def forward(self, x):
        orig = x

        batch = x.shape[0]
        spec = exp.pooled_filter_bank(x)
        x = self.net(spec)

        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)

        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        x, indices = self.sparse.forward(x)

        g, _ = torch.max(x, dim=1, keepdim=True)

        rm = F.gumbel_softmax(self.to_room(g), tau=1, hard=True, dim=-1)
        mx = torch.sigmoid(self.to_mix(g)).view(-1, 1, 1)

        events = self.atoms.forward(x)

        # find the best fit
        orig_spec = torch.fft.rfft(orig, dim=-1, norm='ortho')
        event_spec = torch.fft.rfft(events, dim=-1, norm='ortho')
        fits = orig_spec * event_spec
        fits = torch.fft.irfft(fits, dim=-1, norm='ortho')

        indices = torch.argmax(fits, dim=-1, keepdim=True) #/ exp.n_samples

        shifted = torch.zeros(batch * n_events, 1, exp.n_samples)
        for i in range(batch * n_events):
            idx = indices[i, 0, 0]
            evt = events[i, 0, :exp.n_samples - idx]
            shifted[i, 0, idx:] = evt

        # events = fft_shift(events, indices)
        events = shifted

        events = events.view(batch, n_events, exp.n_samples)

        events = torch.sum(events, dim=1, keepdim=True)

        wet = self.verb.forward(events, rm)

        events = (events * mx) + (wet * (1 - mx))
        return events


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class BestMatchSchedulingExperiment(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)