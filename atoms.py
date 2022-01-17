from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
from datastore import batch_stream
from ddsp import noise_bank2
import zounds
from torch.optim import Adam
from fnet import Transformer
from modules import pos_encode_feature
from modules3 import LinearOutputStack

from test_optisynth import PsychoacousticFeature

sr = zounds.SR22050()
n_samples = 2**14
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/home/john/workspace/audio-data/musicnet/train_data'
torch.backends.cudnn.benchmark = True
learning_rate = 1e-3
n_harmonics = 16
init_value = 0.01


feature = PsychoacousticFeature().to(device)


def nl(x):
    # return torch.clamp(x, 0, 1)
    # return torch.sigmoid(x)
    x = (torch.sin(x) + 1) / 2
    # return (torch.tanh(x) + 1) / 2

    return x ** 2

class SequenceNL(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # return torch.sin(x)
        return F.leaky_relu(x, 0.2)


def init_weights(p):

    with torch.no_grad():
        try:
            p.weight.uniform_(-init_value, init_value)
        except AttributeError:
            pass


def stft(x):
    x = x.unfold(-1, 512, 256)
    win = torch.hann_window(512).to(device)
    x = x * win[None, None, :]
    x = torch.fft.rfft(x)
    x = torch.abs(x)
    return x


def loss(a, b):

    # return F.mse_loss(stft(a), stft(b))

    # a, _ = feature(a)
    # b, _ = feature(b)
    # return F.mse_loss(a, b)

    a = feature.compute_feature_dict(a)
    b = feature.compute_feature_dict(b)

    loss = 0
    for k, v in a.items():
        loss = loss + F.mse_loss(v, b[k])

    return loss


def unit_norm(x, axis=-1):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=axis, keepdims=True)
    else:
        n = torch.norm(x, dim=-1, keepdim=True)
    return x / (n + 1e-12)


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
                # nn.Conv1d(channels, channels, 3, 1, 1),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                # nn.Conv1d(channels, channels, 3, 1, 1),

                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                SequenceNL()
            )
            for _ in range(layers)])
        self.final = nn.Conv1d(channels, out_channels, 3, 1, 1)
        self.apply(init_weights)

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

        self.env = Sequence(channels, n_noise_samples, channels, 1)
        self.noise = Sequence(
            channels, n_noise_samples, channels, self.noise_coeffs)

        self.amp_factor = nn.Parameter(torch.FloatTensor(1).fill_(0.01))

        self.apply(init_weights)

    def forward(self, x):
        batch, atoms, latent = x.shape

        env = nl(self.env(x)) * self.amp_factor
        env = smooth(env, kernel=3)
        n = self.noise(x)
        noise = unit_norm(n, axis=-1)

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
            max_f0=8000,
            n_harmonics=n_harmonics,
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

        self.env = Sequence(channels, n_frames, channels, 1)
        self.f0 = Sequence(channels, n_frames, channels, 1)
        self.harmonics = Sequence(channels, n_harmonics, channels, 1)
        self.harmonic_amp = Sequence(channels, n_harmonics, channels, 1)

        self.amp_factor = nn.Parameter(torch.FloatTensor(1).fill_(0.01))

        self.register_buffer(
            'harmonic_factor',
            torch.arange(2, 2 + n_harmonics, 1)
        )
        self.apply(init_weights)

    def forward(self, x):
        batch, atoms, latent = x.shape

        # (batch, atoms, frames, channels)
        env = nl(self.env(x)).view(batch, atoms, -1, 1) * self.amp_factor
        # encourage amplitude to change smoothly over short periods
        env = smooth(env, kernel=3)

        f0 = self.min_f0 + \
            (nl(self.f0(x).view(batch, atoms, -1, 1) * self.f0_diff))
        # try to ensure that frequencies change smoothly, encouraging different atoms to
        # handle different notes/events
        f0 = smooth(f0, kernel=13)

        # harm = 1 + (nl(self.harmonics(x)).view(batch, atoms, 1, self.n_harmonics) * 10)
        harm_amp = nl(self.harmonic_amp(x)).view(
            batch, atoms, 1, self.n_harmonics)

        # harmonic amps are a factor of envelope
        harm_amp = env * harm_amp

        h_params = harm_amp

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

        return x, f_params, env_params, h_params


class Atoms(nn.Module):
    def __init__(self, atom_latent, n_audio_samples, channels):
        super().__init__()
        self.n_audio_samples = n_audio_samples

        self.transform_latent = LinearOutputStack(
            channels, 3, in_channels=atom_latent, shortcut=False)

        self.mix = LinearOutputStack(channels, 3, out_channels=1)

        self.harmonic = Harmonic(atom_latent, n_audio_samples, 32, channels)
        self.noise = Noise(atom_latent, n_audio_samples, 1024, channels)
        self.apply(init_weights)

    def forward(self, x):
        batch, atoms, latent = x.shape

        x = self.transform_latent(x)

        # harm_amp = torch.sigmoid(self.mix(x))
        # noise_amp = 1 - harm_amp

        h, fp, ap, hp = self.harmonic(x)
        h = h.view(batch, atoms, self.n_audio_samples)

        n, noise_params = self.noise(x)
        n = n.view(batch, atoms, self.n_audio_samples)

        # h = h * harm_amp
        # n = n * noise_amp

        # combine all atoms
        x = (h + n).sum(dim=1)
        return x, fp, ap, noise_params, hp


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        latent = 16
        n_atoms = 16
        channels = 64

        self.atoms = Atoms(latent, n_samples, channels)
        self.apply(init_weights)
        self.params = nn.Parameter(torch.FloatTensor(
            1, n_atoms, latent).normal_(0, 1))

    def forward(self, x):
        x, fp, ap, noise_params, hp = self.atoms(self.params)
        return x, fp, ap, noise_params, hp


model = Model().to(device)
optim = Adam(model.parameters(), lr=learning_rate, betas=(0, 0.9))


def fake_spec():
    return np.log(0.001 + np.abs(zounds.spectral.stft(r)))


def real_spec():
    return np.log(0.001 + np.abs(zounds.spectral.stft(o)))


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    stream = batch_stream(path, '*.wav', 1, n_samples)

    orig = next(stream)
    o = zounds.AudioSamples(orig.squeeze(), sr).pad_with_silence()

    orig = torch.from_numpy(orig).to(device)

    while True:
        optim.zero_grad()
        recon, freq_params, amp_params, noise_params, harm_params = model.forward(
            None)

        freq = freq_params.data.cpu().numpy().squeeze().T
        amp = amp_params.data.cpu().numpy().squeeze().T
        noise = noise_params.data.cpu().numpy().squeeze().sum(axis=0).T
        harm = harm_params.data.cpu().numpy().squeeze()

        l = loss(recon, orig)
        l.backward()
        optim.step()
        r = zounds.AudioSamples(
            recon.squeeze().data.cpu().numpy(), sr).pad_with_silence()
        print(l.item())
