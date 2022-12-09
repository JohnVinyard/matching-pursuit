from torch import Tensor, nn
from config.dotenv import Config
from config.experiment import Experiment
import zounds
import torch
import numpy as np
from torch.jit import ScriptModule, script_method
from torch.nn import functional as F
from modules.normalization import max_norm
from modules.reverb import NeuralReverb
from modules.stft import stft

from train.optim import optimizer
from util import playable
from util.music import MusicalScale
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


total_noise_coeffs = exp.n_samples // 2 + 1

min_f0_hz = 40
max_f0_hz = 3000

min_f0 = (min_f0_hz / exp.samplerate.nyquist)
max_f0 = (max_f0_hz / exp.samplerate.nyquist)
f0_span = max_f0 - min_f0


min_resonance = 0.1
res_span = 1 - min_resonance

n_frames = 128
n_harmonics = 16
harmonic_factors = torch.arange(1, n_harmonics + 1, step=1)
freq_domain_filter_size = 64
n_events = 8

scale = MusicalScale()
frequencies = torch.from_numpy(np.array(list(scale.center_frequencies)) / exp.samplerate.nyquist).float()

param_sizes = {
    'f0': len(scale),
    'f0_fine': n_frames,
    'amp': n_frames,
    'harmonic_amps': n_harmonics,
    'harmonic_decay': n_harmonics,
    'freq_domain_envelope': freq_domain_filter_size,
}

total_synth_params = sum(param_sizes.values())


def build_param_slices():
    current = 0
    slices = {}
    for k, v in param_sizes.items():
        slices[k] = slice(current, current + v)
        current = current + v
    return slices


param_slices = build_param_slices()


def activation(x):
    # return (torch.sin(x * 30) + 1) * 0.5
    return torch.sigmoid(x)


'''
Resonator
-------------------
f0
f0_variance * n_frames
harm_amp * n_harmonics
decay * n_harmonics

Impulse
--------------------
amp * n_frames
freq domain envelope
window (if impulses must be localized in time)


This still needs to be viewed as individual events, because
of polyphonic instruments like the piano

It can be factored into

instrument
------------
harm_amp * n_harmonics
decay * n_harmonics
freq_domain_envelope

event
-----------
f0
f0_variance * n_frames
amp * n_frames


For the initial over-fitting experiment(s), 
we should just optimize the synth parameters directly
and exclude any latent generation/hierarchical network
stuff

https://freesound.org/people/LiftPizzas/sounds/586617/
https://freesound.org/people/Seidhepriest/sounds/232014/
https://freesound.org/people/Walter_Odington/sounds/25602/
https://freesound.org/people/hellska/sounds/328727/
https://freesound.org/people/ananth-pattabi/sounds/44335/
https://freesound.org/people/ldk1609/sounds/55944/
https://freesound.org/people/MTG/sounds/358332/

'''


class SynthParams(object):
    def __init__(self, packed):
        super().__init__()
        self.packed = packed

    @property
    def f0(self) -> Tensor:
        x = self.packed[:, param_slices['f0']].view(-1, len(scale))
        x = F.gumbel_softmax(x, dim=-1, hard=True)
        x = x @ frequencies
        return x.view(-1, 1).repeat(1, n_frames)

    @property
    def f0_fine(self) -> Tensor:
        return self.packed[:, param_slices['f0_fine']]

    @property
    def harmonic_amps(self) -> Tensor:
        return self.packed[:, param_slices['harmonic_amps']]

    @property
    def harmonic_decay(self) -> Tensor:
        return self.packed[:, param_slices['harmonic_decay']]

    @property
    def freq_domain_envelope(self) -> Tensor:
        return self.packed[:, param_slices['freq_domain_envelope']]

    @property
    def amp(self) -> Tensor:
        return self.packed[:, param_slices['amp']]


class Synthesizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, synth_params: SynthParams):

        f0 = synth_params.f0.view(-1, 1, n_frames) * exp.samplerate.nyquist

        f0 = min_f0 + (f0 * f0_span)

        proportion = (f0 / exp.samplerate.nyquist) * 0.1
        f0_fine = synth_params.f0_fine.view(-1, 1, n_frames) * \
            exp.samplerate.nyquist * proportion
        f0 + f0 + f0_fine

        batch = f0.shape[0]

        osc = f0 * harmonic_factors[None, :, None]
        radians = (osc / exp.samplerate.nyquist) * np.pi
        radians = F.interpolate(radians, size=exp.n_samples, mode='linear')
        osc_bank = torch.sin(torch.cumsum(radians, dim=-1))

        harm_decay = synth_params.harmonic_decay.view(-1, n_harmonics)
        harm_decay = min_resonance + (harm_decay * res_span)
        harm_amp = synth_params.harmonic_amps.view(-1, n_harmonics)

        amp = (synth_params.amp.view(-1, 1, n_frames) * 2) - 1
        amp_full = F.interpolate(amp, size=exp.n_samples, mode='linear')

        noise_filter = synth_params.freq_domain_envelope.view(
            -1, 1, freq_domain_filter_size)
        noise_filter = F.interpolate(
            noise_filter, exp.n_samples // 2, mode='linear')
        noise_filter = F.pad(noise_filter, (0, 1))

        noise = torch.zeros(batch, 1, exp.n_samples).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')

        filtered_noise = noise_spec * noise_filter
        noise = torch.fft.irfft(filtered_noise, dim=-1, norm='ortho')
        noise = noise * torch.clamp(amp_full, 0, 1)

        current = torch.zeros(batch, n_harmonics, 1)
        output = []

        for i in range(n_frames):
            current = current + (amp[..., i:i + 1] * harm_amp[..., None])
            current = torch.clamp(current, 0, 1)
            output.append(current)
            current = current * harm_decay[..., None]

        x = torch.cat(output, dim=-1).view(-1, n_harmonics, n_frames)
        x = F.interpolate(x, size=exp.n_samples, mode='linear')

        x = x * osc_bank
        x = noise + x

        x = torch.mean(x, dim=(0, 1), keepdim=True)
        # x = max_norm(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.Parameter(
            torch.zeros(n_events, total_synth_params).uniform_(0.01, 0.999))

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)

        self.rooms = nn.Parameter(torch.zeros(
            self.verb.n_rooms).uniform_(-1, 1))
        self.mix = nn.Parameter(torch.zeros(1).uniform_(0, 1))
        self.synth = Synthesizer()

    def forward(self, x):
        p = activation(self.params)
        params = SynthParams(p)
        samples = self.synth.forward(params)

        rm = F.gumbel_softmax(self.rooms, dim=-1, hard=True)[None, ...]
        wet = self.verb.forward(samples, rm)

        mx = torch.sigmoid(self.mix).view(-1, 1, 1)

        samples = (mx * wet) + ((1 - mx) * samples)

        return samples


model = Model()
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    # loss = exp.perceptual_loss(recon, batch)

    real_spec = stft(batch, 512, 256, pad=True, log_amplitude=True)
    fake_spec = stft(recon, 512, 256, pad=True, log_amplitude=True)

    loss = F.mse_loss(fake_spec, real_spec)
    loss.backward()
    optim.step()

    return loss, recon


@readme
class ResonatorModelExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def orig(self):
        return playable(self.real, exp.samplerate)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, recon = train(item)
            self.fake = recon
            print(i, l.item())