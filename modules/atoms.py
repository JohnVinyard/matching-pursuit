from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
import zounds
from torch.optim import Adam

from modules.ddsp import noise_bank2

def unit_norm(x, axis=-1):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=axis, keepdims=True)
    else:
        n = torch.norm(x, dim=-1, keepdim=True)
    return x / (n + 1e-12)

def nl(x):
    return torch.sigmoid(x) ** 2
    # return torch.clamp(x, 0, 1)
    # return (torch.sin(x) + 1) / 2

class Sequence(nn.Module):
    def __init__(self, atom_latent, n_frames, channels, out_channels):
        super().__init__()
        self.atom_latent = atom_latent
        self.n_frames = n_frames
        self.channels = channels
        self.out_channels = out_channels

        layers = int(np.log2(n_frames) - np.log2(4))
        self.initial = nn.Conv1d(atom_latent, self.channels * 4, 1, 1, 0)
        self.net = nn.Sequential(*[
            nn.Sequential(
                # nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Conv1d(channels, channels, 7, 1, 3),

                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2)
            )
            for _ in range(layers)])
        self.final = nn.Conv1d(channels, out_channels, 3, 1, 1)

    def forward(self, x):
        batch, atoms, latent = x.shape

        x = x.view(batch * atoms, latent, 1)
        x = self.initial(x)
        x = x.reshape(batch * atoms, self.channels, 4)
        x = self.net(x)
        x = self.final(x)
        x = x.reshape(batch, atoms, self.out_channels, self.n_frames)
        x = x.permute(0, 1, 3, 2)
        return x  # (batch, atoms, frames, channels)


class Noise(nn.Module):
    def __init__(self, atom_latent, n_audio_samples, n_noise_samples, channels):
        super().__init__()
        self.atom_latent = atom_latent
        self.n_audio_samples = n_audio_samples
        self.n_noise_samples = n_noise_samples
        noise_frames = (n_audio_samples // n_noise_samples) * 2
        self.noise_coeffs = (noise_frames // 2) + 1
        print('NOISE COEFFS', self.noise_coeffs)
        self.channels = channels

        self.env = Sequence(atom_latent, n_noise_samples, channels, 1)
        self.noise = Sequence(atom_latent, n_noise_samples,
                              channels, self.noise_coeffs)
        self.amp_factor = nn.Parameter(torch.FloatTensor(1).fill_(0.01))

    def forward(self, x):
        batch, atoms, latent = x.shape

        env = nl(self.env(x)) * self.amp_factor
        noise = unit_norm(self.noise(x), axis=1)

        x = env * noise
        x = x.reshape(batch * atoms, self.n_noise_samples, self.noise_coeffs)
        x = x.permute(0, 2, 1)
        noise_params = x

        x = noise_bank2(x)
        x = x.reshape(batch, atoms, self.n_audio_samples)
        return x, noise_params


def upsample(x, size):
    batch, atoms, frames, channels = x.shape
    x = x.view(batch * atoms, frames, channels)
    x = x.permute(0, 2, 1)
    x = F.upsample(x, size, mode='linear')
    x = x.permute(0, 2, 1)
    x = x.view(batch, atoms, size, channels)
    return x

def smooth(x, kernel=7):

    padding = kernel // 2

    batch, atoms, frames, channels = x.shape
    x = x.view(batch * atoms, frames, channels)
    x = x.permute(0, 2, 1)

    x = F.pad(x, (padding, padding), mode='reflect')
    x = F.avg_pool1d(x, kernel, 1)

    x = x.permute(0, 2, 1)
    x = x.view(batch, atoms, frames, channels)
    return x

class Harmonic(nn.Module):
    def __init__(
            self,
            atom_latent,
            n_audio_samples,
            n_frames,
            channels,
            min_f0=20,
            max_f0=800,
            n_harmonics=32,
            sr=zounds.SR22050()):

        super().__init__()
        self.atom_latent = atom_latent
        self.n_audio_samples = n_audio_samples
        self.n_frames = n_frames
        self.channels = channels
        self.n_harmonics = n_harmonics

        self.min_f0 = min_f0 / sr.nyquist
        self.max_f0 = max_f0 / sr.nyquist
        self.f0_diff = self.max_f0 - self.min_f0

        self.env = Sequence(atom_latent, n_frames, channels, 1)
        self.f0 = Sequence(atom_latent, n_frames, channels, 1)
        self.harmonics = Sequence(atom_latent, n_harmonics, channels, 1)
        self.harmonic_amp = Sequence(atom_latent, n_harmonics, channels, 1)

        self.amp_factor = nn.Parameter(torch.FloatTensor(1).fill_(0.01))

        self.register_buffer('harmonic_factor', torch.arange(2, 2 + n_harmonics, 1))
    
    def forward(self, x):
        batch, atoms, latent = x.shape


        # (batch, atoms, frames, channels)
        env = nl(self.env(x)).view(batch, atoms, -1, 1) * self.amp_factor
        env = smooth(env, kernel=3)

        f0 = self.min_f0 + (nl(self.f0(x)).view(batch, atoms, -1, 1) * self.f0_diff)
        f0 = smooth(f0, kernel=13)

        # harm = 1 + (nl(self.harmonics(x)).view(batch, atoms, 1, self.n_harmonics) * 10)
        harm_amp = nl(self.harmonic_amp(x)).view(batch, atoms, 1, self.n_harmonics)

        # harmonic amps are a factor of envelope
        harm_amp = env * harm_amp

        f_params = f0
        env_params = env

        # harmonics are factors of f0
        f = f0 * self.harmonic_factor[None, None, None, :]

        env = upsample(env, self.n_audio_samples)
        f0 = upsample(f0, self.n_audio_samples)
        f = upsample(f, self.n_audio_samples)
        harm_amp = upsample(harm_amp, self.n_audio_samples)

        f0 = torch.sin(torch.cumsum(f0 * np.pi, dim=2)) * env
        f = torch.sin(torch.cumsum(f * np.pi, dim=2)) * harm_amp

        x = f0 + torch.sum(f, dim=-1, keepdim=True)
        return x, f_params, env_params


class Atoms(nn.Module):
    def __init__(self, atom_latent, n_audio_samples, channels):
        super().__init__()
        self.n_audio_samples = n_audio_samples
        self.harmonic = Harmonic(atom_latent, n_audio_samples, 32, channels)
        self.noise = Noise(atom_latent, n_audio_samples, 64, channels)
    
    def forward(self, x):
        batch, atoms, latent = x.shape

        h, fp, ap = self.harmonic(x)
        h = h.view(batch, atoms, self.n_audio_samples)
        n, noise_params = self.noise(x)
        n = n.view(batch, atoms, self.n_audio_samples)
        # combine all atoms
        x = (h + n).sum(dim=1)
        return x, fp, ap, noise_params


