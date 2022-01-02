from torch import nn
import numpy as np
from torch.nn.modules.container import Sequential
import torch
from torch.nn import functional as F
from ddsp import noise_bank2
import zounds


def unit_norm(x, axis=-1):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=axis, keepdims=True)
    else:
        n = torch.norm(x, axis=-1, keepdim=True)
    return x / (n + 1e-12)

def nl(x):
    return torch.clamp(x, 0, 1)

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
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(channels, channels, 3, 1, 1),
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
        noise_frames = n_audio_samples // n_noise_samples
        self.noise_coeffs = (noise_frames // 2) + 1
        self.channels = channels

        self.env = Sequence(atom_latent, n_noise_samples, channels, 1)
        self.noise = Sequence(atom_latent, n_noise_samples,
                              channels, self.noise_coeffs)

    def forward(self, x):
        batch, atoms, latent = x.shape

        env = torch.clamp(self.env(x), 0, 1)
        noise = unit_norm(F.relu(self.noise(x)), axis=1)

        x = env * noise
        x = x.reshape(batch * atoms, self.n_noise_samples, self.noise_coeffs)
        x = x.permute(0, 2, 1)

        x = noise_bank2(x)
        x = x.reshape(batch, atoms, self.n_audio_samples)
        return x


def upsample(x, size):
    batch, atoms, frames, channels = x.shape
    x = x.view(batch * atoms, frames, channels)
    x = x.permute(0, 2, 1)
    x = F.upsample(x, size, mode='linear')
    x = x.permute(0, 2, 1)
    x = x.view(batch, atoms, size, channels)
    return x

class Harmonic(nn.Module):
    def __init__(
            self,
            atom_latent,
            n_audio_samples,
            n_frames,
            channels,
            min_f0=20,
            max_f0=8000,
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
    
    def forward(self, x):
        batch, atoms, latent = x.shape


        # (batch, atoms, frames, channels)
        env = nl(self.env(x)).view(batch, atoms, -1, 1)
        f0 = self.min_f0 + (nl(self.f0(x)).view(batch, atoms, -1, 1) * self.f0_diff)
        harm = 1 + (nl(self.harmonics(x)).view(batch, atoms, 1, self.n_harmonics) * 10)
        harm_amp = nl(self.harmonic_amp(x)).view(batch, atoms, 1, self.n_harmonics)

        # harmonic amps are a factor of envelope
        harm_amp = env * harm_amp

        # harmonics are factors of f0
        f = f0 * harm

        env = upsample(env, self.n_audio_samples)
        f0 = upsample(f0, self.n_audio_samples)
        f = upsample(f, self.n_audio_samples)
        harm_amp = upsample(harm_amp, self.n_audio_samples)

        f0 = torch.sin(torch.cumsum(f0 * np.pi, dim=2)) * env
        f = torch.sin(torch.cumsum(f * np.pi, dim=2)) * harm_amp

        x = f0 + torch.sum(f, dim=-1, keepdim=True)
        return x


class Atoms(nn.Module):
    def __init__(self, atom_latent, n_audio_samples, channels):
        super().__init__()
        self.harmonic = Harmonic(atom_latent, n_audio_samples, 64, channels)
        self.noise = Noise(atom_latent, n_audio_samples, 128, channels)
    
    def forward(self, x):
        h = self.harmonic(x)
        n = self.noise(x)
        # combine all atoms
        x = (h + n).sum(dim=1)
        return x

if __name__ == '__main__':
    batch = 4
    atoms = 16
    latent = 8

    x = torch.FloatTensor(batch, atoms, latent).normal_(0, 1)

    n = Harmonic(latent, 16384, 64, 128)

    result = n.forward(x)
    print(result.shape)
