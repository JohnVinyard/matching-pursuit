import torch
from torch import nn
from torch.nn import functional as F
import zounds
import numpy as np
from config.dotenv import Config
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_convolve, fft_shift

from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm, unit_norm
from modules.perceptual import PerceptualAudioModel
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
from modules.sparse import VectorwiseSparsity
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.music import MusicalScale
from util.readmedocs import readme
from torch.distributions import Uniform

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 8
n_harmonics = 16

def activation(x):
    # return torch.sigmoid(x)

    x_backward = x
    x_forward = torch.clamp(x, 0, 1)
    y = x_backward + (x_forward - x_backward).detach()
    return y

upsample_mode = 'nearest'

def softmax(x):
    # return torch.softmax(x, dim=-1)
    return F.gumbel_softmax(x, tau=1, hard=True, dim=-1)

class Atoms(nn.Module):
    def __init__(
            self,
            n_frames: int,
            n_samples: int,
            n_harmonics: int,
            layers: int,
            channels: int,
            samplerate: zounds.SampleRate,
            freq_change_factor: float = 0.05,
            unit_activation = activation,
            activation=lambda x: F.leaky_relu(x, 0.2),
            discrete_activation=softmax):

        super().__init__()
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.n_harmonics = n_harmonics
        self.layers = layers
        self.channels = channels
        self.freq_change_factor = freq_change_factor
        self.unit_activation = unit_activation

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
            mode=upsample_mode)

        self.amp = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode=upsample_mode)

        self.decay = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode=upsample_mode)

        self.mix = ConvUpsample(
            channels,
            channels,
            start_size=8,
            end_size=n_frames,
            out_channels=1,
            mode=upsample_mode)

        self.spec_shape = LinearOutputStack(
            channels,
            layers,
            out_channels=self.n_harmonics,
            activation=activation)
        

        self.noise_level = nn.Parameter(torch.zeros(1).fill_(1))

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, self.channels)

        f0 = self.discrete_activation(self.f0.forward(x))
        f0 = (f0 @ self.center_freqs)
        # f0 = self.unit_activation(self.f0.forward(x))
        f0 = f0.view(-1, 1, 1)

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

        dist = Uniform(-self.noise_level, self.noise_level)
        noise = dist.sample((batch, 1, self.n_samples)).view(batch, 1, self.n_samples).to(osc.device)
        # noise = torch.zeros(batch, 1, self.n_samples).uniform_(-100, 100)
        # noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')

        noise_filter_size = 64
        filt = osc[:, :, :noise_filter_size]

        filt = F.pad(filt, (0, self.n_samples - noise_filter_size))

        bl_noise = fft_convolve(noise, filt)

        # filt_spec = torch.fft.rfft(filt, dim=-1, norm='ortho')
        # filtered = noise_spec * filt_spec
        # bl_noise = torch.fft.irfft(filtered, dim=-1, norm='ortho')

        mix = self.unit_activation(self.mix.forward(x))
        mix = F.interpolate(mix, size=self.n_samples, mode='linear')

        osc_mix = mix
        noise_mix = 1 - mix

        amp = torch.relu(self.amp.forward(x))
        factors = self.unit_activation(self.spec_shape.forward(x)).view(-1, self.n_harmonics, 1) * amp
        
        amp = torch.cat([amp, factors], dim=1)

        decay = 0.8 + (0.2 * self.unit_activation(self.decay.forward(x)))

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
        bl_noise = bl_noise * amp_with_decay

        full_signal = (full_osc * osc_mix) + (bl_noise * noise_mix)

        full_signal = torch.mean(full_signal, dim=1, keepdim=True)
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
        

        self.norm = ExampleNorm()

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
        x = self.norm(x)

        g, _ = torch.max(x, dim=1, keepdim=True)

        rm = softmax(self.to_room(g))
        mx = torch.sigmoid(self.to_mix(g)).view(-1, 1, 1)

        events = self.atoms.forward(x.view(-1, exp.model_dim))
        # events = unit_norm(events, dim=-1)

        # find the best fit
        # orig_spec = torch.fft.rfft(orig.view(-1, 1, exp.n_samples), dim=-1, norm='ortho')
        # event_spec = torch.fft.rfft(events.view(-1, n_events, exp.n_samples), dim=-1, norm='ortho')
        # fits = orig_spec * event_spec
        # fits = torch.fft.irfft(fits, dim=-1, norm='ortho')
        # fits = fits.view(batch * n_events, 1, exp.n_samples)

        fits = fft_convolve(
            orig.view(-1, 1, exp.n_samples), 
            events.view(-1, n_events, exp.n_samples))
        
        fits = fits.view(batch * n_events, 1, exp.n_samples)

        # account for boundary effects 
        # TODO: Give events unit norm and then multiplby activation!!
        # TODO: convolve in feature domain, not time
        # weighting = torch.linspace(1, 2, exp.n_samples)
        # fits = fits * weighting[None, None, :]

        indices = torch.argmax(fits, dim=-1, keepdim=True) #/ exp.n_samples

        shifted = torch.zeros(batch * n_events, 1, exp.n_samples, device=fits.device)
        for i in range(batch * n_events):
            idx = indices[i, 0, 0]
            evt = events[i, 0, :exp.n_samples - idx] #* fits[i, 0, idx]
            shifted[i, 0, idx:] = evt

        # events = fft_shift(events, indices)
        events = shifted

        events = events.view(batch, n_events, exp.n_samples)

        events = torch.mean(events, dim=1, keepdim=True)


        wet = self.verb.forward(events, rm.view(batch, -1))

        events = (events * mx) + (wet * (1 - mx))

        return events


model = Model().to(device)
optim = optimizer(model, lr=1e-4)

loss_model = PerceptualAudioModel(exp, norm_second_order=True).to(device)

def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)

    # loss = exp.perceptual_loss(recon, batch)
    loss = loss_model.loss(recon, batch)

    loss.backward()
    optim.step()
    return loss, recon


@readme
class BestMatchSchedulingExperiment(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)