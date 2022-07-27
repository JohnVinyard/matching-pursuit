import numpy as np
import zounds
import torch
from torch.nn import functional as F
from torch import nn
from modules.ddsp import NoiseModel, OscillatorBank
from modules.decompose import fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer

from upsample import ConvUpsample, FFTUpsampleBlock
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
import torch.jit as jit
from scipy.signal import tukey

model_dim = 128
n_samples = 2 ** 15
samplerate = zounds.SR22050()

# Core
n_atoms = 64
atoms_to_keep = 32
atom_latent = 16
atom_size = 4096

# Loss function
n_bands = 128
kernel_size = 256




window = torch.from_numpy(tukey(atom_size, 0.1)).float().to(device)

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)

aim = AuditoryImage(512, 128, do_windowing=False, check_cola=False)

init_weights = make_initializer(0.1)


def perceptual_feature(x):
    x = fb.forward(x, normalize=False)
    x = aim.forward(x)
    return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    loss = F.mse_loss(a, b)
    return loss


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        # self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        # self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        # self.scale = nn.Conv1d(channels, channels, 3, 1,
        #                        dilation=dilation, padding=dilation)
        # self.gate = nn.Conv1d(channels, channels, 3, 1,
        #                       dilation=dilation, padding=dilation)
        self.net = nn.Conv1d(
            channels, channels, 3, 1, dilation=dilation, padding=dilation)

    def forward(self, x):
        batch = x.shape[0]
        skip = x

        # scale = self.scale(x)
        # gate = self.gate(x)
        # x = torch.tanh(scale) * F.sigmoid(gate)
        x = F.leaky_relu(self.net(x + skip), 0.2)

        # out = self.out(x)
        # next = self.next(x) + skip
        # return next, out
        return x


class ContextModel(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        c = channels

        self.initial = nn.Conv1d(n_bands, channels, 1, 1, 0)

        self.stack = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 27),
            DilatedBlock(c, 81),
            DilatedBlock(c, 243),
            DilatedBlock(c, 1),
            nn.Conv1d(c, c, 1, 1, 0),
        )

        self.apply(init_weights)

    def forward(self, x):
        x = fb.forward(x, normalize=False)
        x = F.leaky_relu(self.initial(x), 0.2)
        x = self.stack(x)
        return x


class AtomPlacement(jit.ScriptModule):
    def __init__(self, n_samples, n_atoms, atom_size, to_keep):
        super().__init__()
        self.n_samples = n_samples
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.to_keep = to_keep

    @jit.script_method
    def forward(self, features, atoms):
        batch = features.shape[0]
        features = features.view(-1, self.n_atoms, self.n_samples)
        atoms = atoms.view(batch, self.n_atoms, self.atom_size)

        features = features.view(batch, -1)
        values, indices = torch.topk(features, self.to_keep, dim=-1)

        output = torch.zeros(batch, 1, self.n_samples +
                             self.atom_size).to(features.device)
        for b in range(batch):
            for j in range(self.to_keep):
                index = indices[b, j]

                atom_index = index // self.n_samples
                sample_index = index % self.n_samples

                atom = atoms[b, atom_index]
                factor = values[b, j]

                start = sample_index
                stop = start + self.atom_size

                output[b, 0, start: stop] += atom * factor

        return output[..., :self.n_samples]


class AtomGenerator(nn.Module):
    def __init__(self, n_atoms, atom_size):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.to_atoms = nn.Linear(model_dim, n_atoms * atom_latent)
        self.to_seq = nn.Linear(atom_latent, 8 * 128)

        self.to_samples = nn.ModuleDict({
            '128': nn.Conv1d(16, 1, 7, 1, 3),
            '256': nn.Conv1d(16, 1, 7, 1, 3),
            '512': nn.Conv1d(16, 1, 7, 1, 3),
            '1024': nn.Conv1d(16, 1, 7, 1, 3),
            '2048': nn.Conv1d(16, 1, 7, 1, 3),
            '4096': nn.Conv1d(16, 1, 7, 1, 3),
        })

        self.up = nn.Sequential(
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, 8, 4, 2),  # 16
                nn.LeakyReLU(0.2),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 16, 8, 4, 2),  # 64
                nn.LeakyReLU(0.2),
            ),

            nn.Sequential(
                nn.ConvTranspose1d(16, 16, 4, 2, 1),  # 128
                nn.LeakyReLU(0.2)
            ),

            nn.Sequential(
                nn.ConvTranspose1d(16, 16, 4, 2, 1),  # 256
                nn.LeakyReLU(0.2)
            ),

            nn.Sequential(
                nn.ConvTranspose1d(16, 16, 4, 2, 1),  # 512
                nn.LeakyReLU(0.2)
            ),

            nn.Sequential(
                nn.ConvTranspose1d(16, 16, 4, 2, 1),  # 1024
                nn.LeakyReLU(0.2)
            ),

            nn.Sequential(
                nn.ConvTranspose1d(16, 16, 4, 2, 1),  # 2048
                nn.LeakyReLU(0.2)
            ),

            nn.Sequential(
                nn.ConvTranspose1d(16, 16, 4, 2, 1),  # 4096
                nn.LeakyReLU(0.2)
            ),

        )

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, 1, model_dim)

        x = self.to_atoms(x)
        x = x.view(batch, n_atoms, atom_latent)

        x = self.to_seq(x).view(-1, 128, 8)

        bands = {}
        for layer in self.up:
            x = layer(x)
            try:
                to_samples = self.to_samples[str(x.shape[-1])]
                bands[int(x.shape[-1])] = to_samples.forward(x)
            except KeyError:
                pass
        x = fft_frequency_recompose(bands, atom_size)

        x = x.view(batch, n_atoms, atom_size)
        x = x * window[None, None, :]
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = ContextModel(model_dim)

        self.to_atom_latent = LinearOutputStack(model_dim, 3)
        self.atom_gen = AtomGenerator(n_atoms, atom_size)

        self.to_placement_latent = nn.Conv1d(model_dim, n_atoms, 1, 1, 0)

        self.placement = AtomPlacement(
            n_samples, n_atoms, atom_size, atoms_to_keep)

        self.verb = NeuralReverb.from_directory(
            '/home/john/Downloads/reverbs', samplerate, n_samples)

        self.to_room = LinearOutputStack(model_dim, 3, out_channels=self.verb.n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)
        
        
        self.apply(init_weights)

    def forward(self, x):
        x = self.context(x)

        x = F.dropout(x, 0.05)

        g, _ = x.max(dim=-1)

        room = torch.softmax(self.to_room(g), dim=-1)
        mix = torch.clamp(self.to_mix(g).view(-1, 1, 1), 0, 1)

        g = self.to_atom_latent(g)
        atoms = self.atom_gen(g)

        x = self.to_placement_latent(x)
        final = self.placement.forward(x, atoms)

        wet = self.verb.forward(final, room)

        final = (mix * wet) + (final * (1 - mix))
        
        return final


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class EfficientSparseModelExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None
        self.fake = None

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss, self.fake = train_model(item)
            print(loss.item())
