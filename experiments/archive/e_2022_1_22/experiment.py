import torch
from torch import nn
from torch.nn import functional as F
from config.dotenv import Config
from data.datastore import batch_stream
from modules import stft
from modules.ddsp import overlap_add
from modules.linear import LinearOutputStack
from util import device
import zounds
import numpy as np
from torch.optim import Adam

from modules.normal_pdf import pdf
from util.readmedocs import readme
from util.weight_init import make_initializer

init_weights = make_initializer(0.1)


class EnvelopeFromLatent(nn.Module):
    """
    Given a "latent" vector of size (latent_dim,), generate a sequence
    of length (seq_length,)
    """

    def __init__(
            self,
            latent_dim,
            seq_length,
            seq_channels,
            channels,
            activation=nn.LeakyReLU(0.2)):

        super().__init__()
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.seq_channels = seq_channels
        self.channels = channels
        self.initial_size = 4
        self.activation = activation

        self.initial = nn.Linear(latent_dim, self.initial_size * self.channels)
        self.n_layers = int(np.log2(self.seq_length) -
                            np.log2(self.initial_size))
        self.upscale = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                self.activation
            )
            for _ in range(self.n_layers)
        ])

        self.final = nn.Conv1d(channels, self.seq_channels, 3, 1, 1)

    def forward(self, x):
        """
        Accept a number of latent representations in with shape 
        (batch, n_atoms, latent_dim) and transform them into sequences
        of shape (batch, n_atoms, seq_channels, seq_length)
        """
        batch, n_atoms, latent = x.shape

        x = self.initial(x)\
            .view(batch * n_atoms, self.channels, self.initial_size)
        x = self.upscale(x)
        x = self.final(x)
        x = torch.sigmoid(x)
        x = x.view(batch, n_atoms, self.seq_channels, self.seq_length)
        return x


class NoiseModel(nn.Module):

    def __init__(
            self,
            n_audio_samples,
            n_noise_frames,
            n_atoms,
            atom_latent,
            channels):

        super().__init__()
        self.n_audio_samples = n_audio_samples
        self.n_noise_frames = n_noise_frames
        self.n_atoms = n_atoms
        self.atom_latent = atom_latent

        self.fft_step = n_audio_samples // n_noise_frames
        self.fft_window = self.fft_step * 2
        self.n_coeffs = (self.fft_window // 2) + 1
        self.channels = channels

        self.register_buffer(
            'domain', torch.arange(0, self.n_coeffs, step=1))

        # transform each latent representation into a mean and std deviation
        # for the filter
        self.to_filters = LinearOutputStack(
            self.channels, 4, out_channels=2, in_channels=atom_latent)

        # transform each latent representation into a sequence representing
        # the mix of pure sine wave and noise
        self.to_mix = EnvelopeFromLatent(
            atom_latent, self.n_noise_frames, 1, self.channels)

        # transform each latent representation into a sequence representing
        # each atoms overall envelope
        self.to_envelope = EnvelopeFromLatent(
            atom_latent, self.n_noise_frames, 1, self.channels)

        self.apply(init_weights)

    def _noise_spec(self, device):
        """
        Create a spectrogram of white noise with shape
        (n_noise_frames, n_coeffs)
        """

        # create time-domain noise
        x = torch.FloatTensor(
            self.n_audio_samples).uniform_(-1, 1).to(device)
        x = F.pad(x, (0, self.fft_step))
        x = x.unfold(-1, self.fft_window, self.fft_step)

        # take the STFT of the noise
        window = torch.hamming_window(self.fft_window).to(device)
        x = x * window[None, :]
        x = torch.fft.rfft(x, norm='ortho')
        return x

    def forward(self, x):
        batch_size = x.shape[0]

        latent = x.view(batch_size, self.n_atoms, self.atom_latent)

        # generate parameters from the latent representation of each atom
        filter_params = self.to_filters(latent)
        mix_params = torch.sigmoid(self.to_mix(latent).view(
            batch_size, self.n_atoms, 1, self.n_noise_frames))
        env = (torch.sigmoid(self.to_envelope(latent))).view(
            batch_size * self.n_atoms, 1, self.n_noise_frames)
        audio_rate_env = F.upsample(env, size=self.n_audio_samples, mode='linear')
        env = env.view(batch_size, self.n_atoms, 1, self.n_noise_frames).permute(0, 1, 3, 2)

        noise_amp = mix_params.view(
            batch_size, self.n_atoms, 1, self.n_noise_frames)
        sine_amp = (1 - mix_params).view(batch_size * self.n_atoms, 1, self.n_noise_frames)
        sine_amp = F.upsample(sine_amp, size=self.n_audio_samples, mode='linear')\
            .view(batch_size, self.n_atoms, 1, self.n_audio_samples)

        # mean and spec are of shape (batch, n_atoms, 1)
        means = (torch.sigmoid(filter_params[:, :, :1]) ** 2) * self.n_coeffs
        stds = (torch.sigmoid(filter_params[:, :, 1:]) ** 2) * self.n_coeffs
        stds = torch.clamp(stds, 1, self.n_coeffs)

        print('-------------------------------------------')
        print(means.data.cpu().numpy().squeeze())
        print(stds.data.cpu().numpy().squeeze())
        print(env.data.cpu().numpy().squeeze().std(axis=-1))

        # sine_freq will be of shape (batch, n_atoms, n_audio_samples)
        sine_freq = torch.zeros(batch_size, self.n_atoms, self.n_audio_samples).to(x.device)
        sine_freq[:] = means / self.n_coeffs
        wave = torch.sin(torch.cumsum(sine_freq * torch.pi, dim=-1))\
            .view(batch_size, self.n_atoms, self.n_audio_samples)
        wave = wave[:, :, None, :] * sine_amp * audio_rate_env
        wave = wave.view(batch_size, self.n_atoms, 1, self.n_audio_samples)
        wave = wave.sum(dim=1) / self.n_atoms

        x = self._noise_spec(x.device)
        x = x.view(self.n_noise_frames, self.n_coeffs)

        # create a frequency domain filter for the noise and
        # apply it
        domain = self.domain[None, None, :]

        freq_domain_filter = pdf(domain, means, stds)
        freq_domain_filter = freq_domain_filter / (freq_domain_filter.max() + 1e-12)

        # multiply freq domain filters by spectral noise
        x = freq_domain_filter[:, :, None, :] * x[None, None, :, :]
        x = x.view(batch_size, self.n_atoms,
                   self.n_noise_frames, self.n_coeffs)
        # apply the noise amplitude
        noise_amp = noise_amp.permute(0, 1, 3, 2).view(
            batch_size, self.n_atoms, self.n_noise_frames, 1)
        x = x * noise_amp * env
        # sum over all atoms in frequency dimension
        x = x.sum(dim=1).view(batch_size, self.n_noise_frames, self.n_coeffs) / self.n_atoms

        # take the inverse STFT to translate back into the
        # time domain
        x = torch.fft.irfft(x, norm='ortho')

        # overlap add wants (batch, channels, frames, samples)
        x = overlap_add(x[:, None, :, :])

        # overlap/add and truncate to recover original audio length
        noise = x[..., :self.n_audio_samples]

        return (noise + (wave * 0.1))


class WrapperModel(nn.Module):
    def __init__(self, n_atoms, atom_latent):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_latent = atom_latent

        self.latents = nn.Parameter(torch.FloatTensor(
            1, self.n_atoms, self.atom_latent).normal_(0, 1))
        self.model = NoiseModel(2**14, 64, n_atoms, atom_latent, 64)

    def forward(self, x):
        result = self.model(self.latents)
        return result


@readme
class NoiseGuided(object):
    def __init__(self):
        super().__init__()

        self.n_samples = 2 ** 14
        self.n_noise_frames = 128

        self.batch_size = 1
        self.n_atoms = 16
        self.atom_latent = 16
        self.channels = 64

        self.oned = None
        self.twod = None
        self.noise = None
        self.noise_spec = None
        self.recon = None
        self.orig = None

        self.sr = zounds.SR22050()

    def real(self):
        return zounds.AudioSamples(self.orig.squeeze(), self.sr).pad_with_silence()

    def real_spec(self):
        return np.log(0.001 + np.abs(zounds.spectral.stft(self.real())))

    def fake(self):
        return zounds.AudioSamples(self.recon.data.cpu().numpy().squeeze(), self.sr).pad_with_silence()

    def fake_spec(self):
        return np.log(0.001 + np.abs(zounds.spectral.stft(self.fake())))

    def run(self):

        stream = batch_stream(Config.audio_path(), '*.wav', self.batch_size, self.n_samples)
        samples = next(stream)
        samples /= (samples.max() + 1e-12)

        self.orig = samples
        samples = torch.from_numpy(samples).to(device).float()
        spec = stft(samples)

        model = WrapperModel(self.n_atoms, self.atom_latent).to(device)
        optim = Adam(model.parameters(), lr=1e-4, betas=(0, 0.9))

        while True:
            optim.zero_grad()
            recon = model(None)
            self.recon = recon
            recon_spec = stft(recon)

            loss = F.mse_loss(spec, recon_spec)
            loss.backward()
            optim.step()
            print('ATOMS', loss.item())
