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


mean_slice = slice(0, 1)
std_slice = slice(1, 2)
amp_slice = slice(2, 3)
mag_slice = slice(3, 260)
phase_slice = slice(260, 260 + 257)
noise_coeff_slice = slice(260 + 257, 260 + 257 + 16)


total_params = 1 + 1 + 1 + 257 + 257 + 16


n_events = 16
total_coeffs = exp.n_samples // 2 + 1
window_size = 512
step_size = window_size // 2
n_coeffs = window_size // 2 + 1
noise_spectral_shape = 16
n_frames = exp.n_samples // step_size
min_resonance = 0.01
res_span = 1 - min_resonance

freqs = torch.fft.rfftfreq(window_size) * np.pi


def unpack(x):
    means = (x[..., mean_slice] * 2) - 1
    stds = x[..., std_slice] * 0.1
    amps = x[..., amp_slice]

    mag = min_resonance + (res_span * x[..., mag_slice])

    phase = (x[..., phase_slice] * np.pi * 2) - np.pi
    phase = phase + freqs

    noise_coeff = x[..., noise_coeff_slice]

    return means, stds, amps, mag, phase, noise_coeff



def unit_activation(x):
    # return torch.sigmoid(x)
    return torch.clamp(x, 0, 1)
    # return (torch.sin(x) + 1) * 0.5



def to_mag_and_phase(x):

    r, i = x.real, x.imag

    paired = torch.cat([r[..., None], i[..., None]], dim=-1)

    reference = torch.zeros_like(paired)
    reference[..., 1] = 1

    norm = torch.norm(paired, dim=-1) + 1e-8
    ref_norm = torch.norm(reference, dim=-1)

    # normalized angle between vectors, in radians
    phase = ((paired * reference).sum(dim=-1) / (norm * ref_norm)) * np.pi

    mag = norm
    return mag, phase


def to_complex(real, imag):

    tf = torch.complex(
        real * torch.cos(imag),
        real * torch.sin(imag)
    )
    return tf

    # spec = real * torch.exp(1j * imag)
    # return spec


def localized_noise(means, stds, spec_shape, n_samples, device):
    """
    Create a band-limited noise impulse, localized in time via a
    gaussian window/probability-density function
    """

    # create the gaussian windows
    rng = torch.arange(0, n_samples, device=device)
    dist = Normal(
        torch.clamp(means * n_samples, -(n_samples // 2), n_samples * 1.5),
        # 0,
        torch.clamp((1e-8 + stds) * n_samples, 0, n_samples - 1))
    probs = torch.exp(dist.log_prob(
        rng[None, ...])).view(-1, n_events, n_samples)
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
    # noise = fft_shift(noise, means)[..., :exp.n_samples]
    return noise


def resonance(noise_atoms, resonance, phase):

    noise_atoms = F.pad(noise_atoms, (0, step_size))
    windowed = \
        noise_atoms.unfold(-1, 512, 256) \
        * torch.hamming_window(window_size)[None, None, None, :]
    spec = torch.fft.rfft(windowed, dim=-1, norm='ortho')

    res_mags = []
    res_phases = []

    mags, phases = to_mag_and_phase(spec)

    for i in range(n_frames):

        curr_mag =     mags[:, :, i:i + 1, :]
        curr_phase = phases[:, :, i:i + 1, :]

        if i == 0:
            res_mags.append(curr_mag)
            res_phases.append(curr_phase)
        else:
            prev_mag = res_mags[i - 1]
            prev_phase = res_phases[i - 1]

            m = curr_mag + (resonance[:, :, None, :] * prev_mag)
            # p = curr_phase + (phase[:, :, None, :] + prev_phase)
            p = curr_phase + phase[:, :, None, :]


            res_mags.append(m)
            res_phases.append(p)
        

    mags = torch.cat(res_mags, dim=2)
    phases = torch.cat(res_phases, dim=2)

    final = to_complex(mags, phases)

    res = torch.fft.irfft(final, dim=-1, norm='ortho')
    res = overlap_add(res, apply_window=False)[..., :exp.n_samples]
    return res


class Atoms(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = unit_activation(x)

        means, stds, amps, res_mags, res_phases, spec_shape = unpack(x)

        atoms = localized_noise(
            means, stds, spec_shape, exp.n_samples, device=device)
        atoms = atoms * amps
        res = resonance(atoms, res_mags, res_phases)
        res = res.view(-1, n_events, exp.n_samples)

        res = torch.sum(res, dim=1, keepdim=True)
        res = max_norm(res)
        return res


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(
            1, n_events, total_params).uniform_(0, 1))
        self.atoms = Atoms()

    def forward(self, x):
        return self.atoms(self.p)


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
