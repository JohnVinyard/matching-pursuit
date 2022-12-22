from torch import nn
import zounds
import torch
from torch.distributions import Normal
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.ddsp import overlap_add

from modules.decompose import fft_resample
from modules.normalization import max_norm
from modules.stft import stft
from train.optim import optimizer
from upsample import FFTUpsampleBlock
from util import device, playable
import numpy as np

from util.readmedocs import readme
from torch.nn import functional as F

'''
event
---------------
- mean (time) 1
- std (width) 1
- resonance magnitudes 257 (recurrently multiply)
- noise spectral shape 16 (upscale and multiply in frequency domain)


/home/john/workspace/matching-pursuit/config/experiment.py:78: UserWarning: Using a target size 
(torch.Size([1, 128, 128, 257])) that is different to the input size 
(torch.Size([32, 128, 128, 257])). 
This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.

'''

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

total_params = 1 + 1 + 1 + 257 + 16
n_events = 16
total_coeffs = exp.n_samples // 2 + 1
window_size = 512
step_size = window_size // 2
n_coeffs = window_size // 2 + 1
noise_spectral_shape = 16
n_frames = exp.n_samples // step_size
min_resonance = 0.01
res_span = 1 - min_resonance


def unit_activation(x):
    # return torch.sigmoid(x)
    return torch.clamp(x, 0, 1)


def to_mag_and_phase(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    return mag, phase


def to_complex(real, imag):
    spec = real * torch.exp(1j * imag)
    return spec


def localized_noise(means, stds, spec_shape, n_samples, device):
    """
    Create a band-limited noise impulse, localized in time via a
    gaussian window/probability-density function
    """

    # create the gaussian windows
    rng = torch.arange(0, n_samples, device=device)
    dist = Normal(
        # torch.clamp(means * n_samples, 0, 1), 
        0,
        torch.clamp((1e-8 + stds) * n_samples, 0, n_samples - 1))
    probs = torch.exp(dist.log_prob(rng[None, ...])).view(-1, n_events, n_samples)
    probs = max_norm(probs)

    # create white noise
    noise = torch.zeros(
        means.shape[0], n_events, n_samples, device=device).uniform_(-1, 1)

    spec_shape = F.interpolate(spec_shape, size=total_coeffs, mode='linear')

    noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')

    # bandpass filter noise in the frequency domain
    noise_spec = noise_spec * spec_shape
    noise = torch.fft.irfft(noise_spec, dim=-1, norm='ortho')

    # localize the band-limited noise impulses in time
    # using the gaussian windows
    noise = probs * noise
    noise = noise.view(-1, n_events, exp.n_samples)
    noise = fft_shift(noise, means)[..., :exp.n_samples]
    return noise


def resonance(noise_atoms, resonance):

    noise_atoms = F.pad(noise_atoms, (0, step_size))
    windowed = \
        noise_atoms.unfold(-1, 512, 256) \
        * torch.hamming_window(window_size)[None, None, None, :]
    spec = torch.fft.rfft(windowed, dim=-1, norm='ortho')

    res_specs = []

    m = spec

    for i in range(n_frames):
        # each successive frame should be impulse + (impulse * resonance) + (impulse-1 * resonance)
        res = resonance 
        imp = m[:, :, i, :]
        if i == 0:
            curr = imp + (imp * res)
        else:
            prev = res_specs[-1]
            curr =  imp + ((prev.view(-1, n_events, n_coeffs)) * res)
        
        curr = curr[:, :, None, :]
        res_specs.append(curr)
    
    final = torch.cat(res_specs, dim=2)
    res = torch.fft.irfft(final, dim=-1, norm='ortho')
    res = overlap_add(res, apply_window=False)[..., :exp.n_samples]
    return res


class Atoms(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = unit_activation(x)

        # means can be before or after the audio frame
        means = x[..., 0:1]
        stds = x[..., 1:2]
        res_mags = min_resonance + (res_span * x[..., 2:259])
        spec_shape = x[..., 259:-1]
        amps = x[..., -1:].view(-1, n_events, 1)
        atoms = localized_noise(means, stds, spec_shape, exp.n_samples, device=device)
        atoms = atoms * amps
        res = resonance(atoms, res_mags)
        res = res.view(-1, n_events, exp.n_samples)

        res = torch.sum(res, dim=1, keepdim=True)
        res = max_norm(res)
        return res


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n_events, total_params).uniform_(0, 1))
        self.atoms = Atoms()
    
    def forward(self, x):
        return self.atoms(self.p)

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)

    # real = stft(batch, 512, 256, pad=True)
    # fake = stft(recon, 512, 256, pad=True)

    # loss = F.mse_loss(fake, real)

    loss.backward()
    optim.step()
    return loss, recon

@readme
class ResonantMatchingPursuitExperiment(object):
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
            print('R', i, l.item())