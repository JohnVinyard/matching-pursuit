from ast import Mult
from operator import mod
import numpy as np
import zounds
import torch
from torch.nn import functional as F
from torch import nn
from config.dotenv import Config
from fft_basis import morlet_filter_bank
from modules.ddsp import NoiseModel, OscillatorBank
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from train.optim import optimizer

from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
import torch.jit as jit
from scipy.signal import tukey

model_dim = 128
n_samples = 2 ** 15
samplerate = zounds.SR22050()
n_frames = n_samples // 256

# Core
n_atoms = 512
atoms_to_keep = 512
atom_size = 512
atom_latent = 32


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

pif = PsychoacousticFeature([128] * 6).to(device)

init_weights = make_initializer(0.1)


def perceptual_feature(x):
    # x = fb.forward(x, normalize=False)
    # x = aim.forward(x)

    bands = fft_frequency_decompose(x, 1024)
    x = torch.cat(list([b.reshape(x.shape[0], -1) for b in bands.values()]), dim=-1)

    # spec = stft(x, 512, 256, log_amplitude=False)

    # x = torch.cat([x.view(x.shape[0], -1), spec.view(spec.shape[0], -1)], dim=-1)
    return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    # loss = F.mse_loss(a, b)
    loss = torch.abs(a - b).sum() / a.shape[0]
    return loss


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
    def forward(self, x):
        batch = x.shape[0]
        skip = x
        scale = self.scale(x)
        gate = self.gate(x)
        x = torch.tanh(scale) * F.sigmoid(gate)
        out = self.out(x)
        next = self.next(x) + skip
        return next, out


class ContextModel(nn.Module):
    """
    Interface: (batch, 1, n_samples) => (batch, model_dim, n_samples)
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.stack = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 1),
        )

        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]

        n = x = fb.forward(x, normalize=False)
        outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)

        for layer in self.stack:
            n, o = layer.forward(n)
            outputs = outputs + o
        
        return outputs


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


class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim,
            model_dim,
            n_samples,
            constrain=True,
            lowest_freq=30 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)

        self.noise = NoiseModel(
            model_dim,
            n_samples // 256,
            (n_samples // 256) * 2,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)

    def forward(self, x):
        x = x.view(x.shape[0], model_dim, -1)
        harm = self.osc.forward(x)
        noise = self.noise(x)
        signal = harm + noise
        return signal


class AtomGenerator(nn.Module):
    def __init__(self, n_atoms, atom_size):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_size = atom_size


        self.latents = nn.Parameter(torch.zeros(n_atoms, atom_latent).normal_(0, 1))

        self.to_seq = nn.Linear(atom_latent, 8 * 128)

        self.up = nn.Sequential(
            nn.ConvTranspose1d(128, 64, 8, 4, 2),  # 32
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 8, 4, 2),  # 128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 1, 8, 4, 2), # 512
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(16, 1, 4, 2, 1), # 1024
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(8, 1, 4, 2, 1), # 4096
        
            nn.ConvTranspose1d(32, 16, 8, 4, 2),  # 512
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 8, 8, 4, 2),  # 2048
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(8, 1, 4, 2, 1),  # 4096
        )
        # self.up = nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1)
        # self.audio = AudioModel(atom_size)

        # self.to_samples = nn.ModuleDict({
        #     '128': nn.Conv1d(model_dim, 1, 7, 1, 3),
        #     '256': nn.Conv1d(model_dim, 1, 7, 1, 3),
        #     '512': nn.Conv1d(model_dim, 1, 7, 1, 3),
        #     '1024': nn.Conv1d(model_dim, 1, 7, 1, 3),
        #     '2048': nn.Conv1d(model_dim, 1, 7, 1, 3),
        #     '4096': nn.Conv1d(model_dim, 1, 7, 1, 3),
        # })

        # self.up = nn.Sequential(
        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 8, 4, 2),  # 16
        #         nn.LeakyReLU(0.2),
        #     ),
        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 8, 4, 2),  # 64
        #         nn.LeakyReLU(0.2),
        #     ),

        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),  # 128
        #         nn.LeakyReLU(0.2)
        #     ),

        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),  # 256
        #         nn.LeakyReLU(0.2)
        #     ),

        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),  # 512
        #         nn.LeakyReLU(0.2)
        #     ),

        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),  # 1024
        #         nn.LeakyReLU(0.2)
        #     ),

        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),  # 2048
        #         nn.LeakyReLU(0.2)
        #     ),

        #     nn.Sequential(
        #         nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1),  # 4096
        #         nn.LeakyReLU(0.2)
        #     ),

        # )

    def forward(self, x):
        batch = x.shape[0]

        x = x.view(batch, atom_latent)
        l = self.latents.view(n_atoms, atom_latent)

        l = l[None, ...] + x[:, None, :]
        x = l.view(batch, n_atoms, atom_latent)
        x = self.to_seq(x).view(-1, 128, 8)
        x = self.up(x)

        # bands = {}
        # for layer in self.up:
        #     x = layer(x)
        #     try:
        #         to_samples = self.to_samples[str(x.shape[-1])]
        #         bands[int(x.shape[-1])] = to_samples.forward(x)
        #     except KeyError:
        #         pass
        # x = fft_frequency_recompose(bands, atom_size)

        x = x.view(batch, n_atoms, atom_size)
        x = x * window[None, None, :]
        return x


class StaticAtoms(nn.Module):
    def __init__(self, n_atoms, atom_size):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        bank = morlet_filter_bank(
            samplerate,
            atom_size,
            zounds.MelScale(zounds.FrequencyBand(20, 3520), n_atoms),
            0.1).real * 0.1
        self.atoms = nn.Parameter(torch.from_numpy(bank[None, ...]))

    def forward(self, x):
        batch = x.shape[0]

        # norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        # atoms = self.atoms / (norms + 1e-8)

        x = self.atoms.repeat(x.shape[0], 1, 1)
        x = x.view(batch, n_atoms, atom_size)
        x = x * window[None, None, :]
        return x


class MultibandAtoms(nn.Module):
    def __init__(self, n_atoms, atom_size, n_bands):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.n_bands = n_bands

        start = int(np.log2(atom_size))

        sizes = list(range(start, start - n_bands, -1))
        self.atoms = nn.ParameterDict({
            str(size): nn.Parameter(torch.zeros(n_atoms, 1, 2**size).uniform_(-0.01, 0.01))
            for size in sizes})
        
        print('ATOM BANDS', [2**int(k) for k in self.atoms.keys()])
    
    def forward(self, x):        
        z = fft_frequency_recompose(
            {int(k): v for k, v in self.atoms.items()}, self.atom_size)
        
        z = z.view(1, self.n_atoms, self.atom_size)
        z = z.repeat(x.shape[0], 1, 1)
        z = z * window[None, None, :]
        return z



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = ContextModel(model_dim)

        self.to_atom_latent = LinearOutputStack(model_dim, 3, out_channels=atom_latent)
        # self.atom_gen = AtomGenerator(n_atoms, atom_size)

        self.atom_gen = StaticAtoms(n_atoms, atom_size)
        # self.atom_gen = MultibandAtoms(n_atoms, atom_size, 6)

        self.to_placement_latent = nn.Conv1d(model_dim, n_atoms, 1, 1, 0)

        self.placement = AtomPlacement(
            n_samples, n_atoms, atom_size, atoms_to_keep)

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), samplerate, n_samples)

        self.to_room = LinearOutputStack(
            model_dim, 3, out_channels=self.verb.n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.context(x)

        # x = F.dropout(x, 0.05)

        g, _ = x.max(dim=-1)

        room = torch.softmax(self.to_room(g), dim=-1)
        mix = torch.sigmoid(self.to_mix(g).view(-1, 1, 1)) * 0.2

        g = self.to_atom_latent(g)
        atoms = self.atom_gen(g)

        x = self.to_placement_latent(x)

        final = self.placement.forward(x, atoms)

        # wet = self.verb.forward(final, room)
        # final = (mix * wet) + (final * (1 - mix))
        
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
        self.model = model

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))
    
    def atoms(self):
        return np.abs(np.fft.rfft(self.model.atom_gen.atoms.data.cpu().numpy().squeeze(), axis=-1))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss, self.fake = train_model(item)
            print(loss.item())
