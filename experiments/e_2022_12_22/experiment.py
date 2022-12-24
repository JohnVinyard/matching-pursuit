from torch import nn
import zounds
import torch
from torch.distributions import Normal
from config.dotenv import Config
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.ddsp import overlap_add

from modules.decompose import fft_resample
from modules.fft import fft_convolve
from modules.normalization import max_norm
from modules.reverb import NeuralReverb
from modules.stft import stft
from train.optim import optimizer
from upsample import FFTUpsampleBlock
from util import device, playable
import numpy as np

from util.readmedocs import readme
from torch.nn import functional as F


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_harmonics = 8

n_events = 16
total_coeffs = exp.n_samples // 2 + 1
window_size = 512
step_size = window_size // 2
n_coeffs = window_size // 2 + 1
noise_spectral_shape = 16
n_frames = exp.n_samples // step_size
min_resonance = 0.01
res_span = 1 - min_resonance

mean_slice = slice(0, 1)
std_slice = slice(1, 2)
amp_slice = slice(2, 3)
f0_slice = slice(3, 4)
factors_slice = slice(4, 12)
mag_slice = slice(12, 20)
noise_coeff_slice = slice(20, 20 + noise_spectral_shape)
fine_env_slice = slice(20 + noise_spectral_shape, 20 + noise_spectral_shape + exp.n_frames)
amp_factors_slice = slice(20 + noise_spectral_shape + exp.n_frames, 20 + noise_spectral_shape + exp.n_frames + n_harmonics)


total_params = 1 + 1 + 1 + 1 + 12 + 12 + 16 + 128

min_freq = 20 / exp.samplerate.nyquist
max_freq = 3000 / exp.samplerate.nyquist
freq_span = max_freq - min_freq


def unpack(x):
    means = x[..., mean_slice]
    stds = x[..., std_slice] * 0.1
    amps = x[..., amp_slice] ** 2
    f0 = x[..., f0_slice] ** 2
    factors = x[..., factors_slice] * 8
    mags = x[..., mag_slice]
    noise_coeff = x[..., noise_coeff_slice]
    fine_env = (x[..., fine_env_slice] * 2) - 1
    amp_factors = x[..., amp_factors_slice] ** 2

    return means, stds, amps, f0, factors, mags, noise_coeff, fine_env, amp_factors



def unit_activation(x):
    # return torch.sigmoid(x)
    return torch.clamp(x, 0, 1)
    # return (torch.sin(x) + 1) * 0.5



def generate_resonance(f0, factors, mag, amp_factors):

    f0 = f0.view(-1, 1)
    factors = factors.view(-1, n_harmonics)
    mag = mag.view(-1, n_harmonics)

    f0 = min_freq + (f0 * freq_span)

    indices = torch.where(f0 > 1)
    f0[indices] = 0

    all_freqs = f0 * factors
    all_freqs = all_freqs * np.pi
    all_freqs = all_freqs.view(-1, n_harmonics, 1).repeat(1, 1, exp.n_samples)
    all_freqs = torch.sin(torch.cumsum(all_freqs, dim=-1)) * amp_factors.view(-1, n_harmonics, 1)


    mag = mag.view(-1, n_harmonics, 1).repeat(1, 1, n_frames)
    mag = torch.cumprod(mag, dim=-1)
    # mag = torch.exp(torch.cumsum(torch.log(mag + 1e-8), dim=-1))
    mag = F.interpolate(mag, size=exp.n_samples, mode='linear')

    resonance = all_freqs * mag


    resonance = torch.sum(resonance, dim=1, keepdim=True)

    resonance = resonance.view(-1, n_events, exp.n_samples)
    return resonance



def localized_noise(means, stds, spec_shape, n_samples, device):
    """
    Create a band-limited noise impulse, localized in time via a
    gaussian window/probability-density function
    """

    # create the gaussian windows
    rng = torch.arange(0, n_samples, device=device)
    dist = Normal(
        torch.clamp(means * n_samples, -(n_samples // 2), n_samples * 1.5),
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




class Atoms(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = unit_activation(x)
        means, stds, amps, f0, factors, mags, spec_shape, fine_env, amp_factors  = unpack(x)

        fine_env = fine_env.view(-1, 1, exp.n_frames)
        fine_env = torch.cumsum(fine_env, dim=-1)
        fine_env = torch.clamp(fine_env, 0, 1)

        fine_env = F.interpolate(fine_env, size=exp.n_samples, mode='linear')
        fine_env = fine_env.view(-1, exp.n_samples)

        atoms = localized_noise(
            means, stds, spec_shape, exp.n_samples, device=device)
        atoms = atoms * amps * fine_env

        res = generate_resonance(f0, factors, mags, amp_factors)

        res = fft_convolve(atoms, res)
        res = res.view(-1, n_events, exp.n_samples)
        res = torch.sum(res, dim=1, keepdim=True)
        # res = max_norm(res)
        return res


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(
            1, n_events, total_params).uniform_(0, 1))
        self.atoms = Atoms()
        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)

        self.mix = nn.Parameter(torch.zeros(1, 1, 1).uniform_(0, 1))
        self.rooms = nn.Parameter(torch.zeros(1, self.verb.n_rooms).uniform_(-1, 1))

    def forward(self, x):
        atoms = self.atoms(self.p)

        wet = self.verb.forward(atoms, torch.softmax(self.rooms, dim=-1))

        mx = torch.clamp(self.mix, 0, 1)

        final = (mx * wet) + ((1 - mx) * atoms)
        return final


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)

    # fake = stft(recon, 512, 256, pad=True)
    # real = stft(batch, 512, 256, pad=True)
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
