from config.experiment import Experiment
from loss.least_squares import least_squares_disc_loss
from modules.atoms import unit_norm
from modules.ddsp import NoiseModel, noise_bank2
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import sparsify, sparsify_vectors
from perceptual.feature import NormalizedSpectrogram
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util.readmedocs import readme
import zounds
from torch import nn
from util import device
from torch.nn import functional as F
import torch
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def choice_softmax(x):
    # return sparse_softmax(x, normalize=False)
    return hard_softmax(x, invert=True)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0, bias=False)

    def forward(self, x):
        orig = x
        x = F.dropout(x, p=0.1)
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        return x


class ImpulseDictionary(nn.Module):
    def __init__(self, n_atoms, n_samples):
        super().__init__()

        self.n_atoms = n_atoms

        self.n_samples = n_samples
        self.window_size = 64
        self.step_size = 32
        self.n_coeffs = self.window_size // 2 + 1
        self.n_frames = self.n_samples // self.step_size

        self.spectral_shapes = nn.Parameter(
            torch.zeros(self.n_atoms, self.n_coeffs, 1).uniform_(-1, 1).repeat(1, 1, self.n_frames))

    def forward(self):
        as_samples = noise_bank2(self.spectral_shapes)
        return as_samples.view(1, self.n_atoms, self.n_samples)


class TransferFunctionDictionary(nn.Module):
    def __init__(self, n_atoms, n_samples, n_frequencies, n_frames):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_samples = n_samples
        self.n_frequencies = n_frequencies
        self.n_frames = n_frames

        self.t = nn.Parameter(torch.zeros(
            1, self.n_atoms, self.n_samples).uniform_(-1, 1) * (torch.linspace(1, 0, self.n_samples) ** 15)[None, None, :])

        # self.osc = nn.Parameter(torch.zeros(self.n_atoms, self.n_frequencies).uniform_(0, 1))
        # self.amp = nn.Parameter(torch.zeros(self.n_atoms, self.n_frequencies).uniform_(0, 1))
        # self.decay = nn.Parameter(torch.zeros(self.n_atoms, self.n_frequencies).uniform_(0, 1))
    
    def forward(self):
        return self.t
        # osc = torch.sigmoid(self.osc.view(self.n_atoms, self.n_frequencies, 1)) ** 2
        # freq = (osc * np.pi).repeat(1, 1, self.n_samples)
        # osc = torch.sin(torch.cumsum(freq, dim=-1))
        # osc = osc * (self.amp.view(self.n_atoms, self.n_frequencies, 1) ** 2)

        # dec = 0.7 + (torch.sigmoid(self.decay.view(self.n_atoms, self.n_frequencies, 1).repeat(1, 1, self.n_frames)) * 0.299999)
        # dec = torch.exp(torch.cumsum(torch.log(dec), dim=-1))
        # # dec = torch.cumprod(dec, dim=-1)
        # dec = F.interpolate(dec, size=self.n_samples, mode='linear')

        # osc = osc * dec
        # osc = torch.sum(osc, dim=1, keepdim=True)
        # return osc.view(1, self.n_atoms, self.n_samples)


class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = DilatedStack(channels, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        self.j = nn.Conv1d(channels, 1, 1, 1, 0)
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x):
        batch, _, n_samples = x.shape

        spec = exp.fb.forward(x, normalize=False)
        pos = pos_encoded(
            batch, spec.shape[-1], 16, device=x.device).permute(0, 2, 1)
        spec = torch.cat([pos, spec], dim=1)
        spec = self.reduce(spec)

        x, features = self.net.forward(spec, return_features=True)

        j = self.j.forward(x)

        return j, features



class Model(nn.Module):
    def __init__(self, channels, n_heads, n_layers, atom_size, n_atoms, k_sparse):
        super().__init__()

        self.k_sparse = k_sparse
        self.n_atoms = n_atoms
        self.atom_size = atom_size

        self.encoder = nn.Sequential(
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 3),
            DilatedBlock(channels, 9),
            DilatedBlock(channels, 27),
            DilatedBlock(channels, 1),
        )

        self.reduce = nn.Conv1d(channels + 33, channels, 1, 1, 0)

        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)

        self.to_impulse_choice = LinearOutputStack(
            channels, 3, out_channels=self.n_atoms)
        self.to_transfer_choice = LinearOutputStack(
            channels, 3, out_channels=self.n_atoms)
        self.to_mixture = LinearOutputStack(channels, 3, out_channels=1)

        self.atom_dictionary = nn.Parameter(torch.zeros(self.n_atoms, self.atom_size).uniform_(-1, 1))

        self.impulse_dict = ImpulseDictionary(n_atoms, self.atom_size)
        self.transfer_dict = TransferFunctionDictionary(
            n_atoms, exp.n_samples, 16, exp.n_samples // exp.step_size)

        self.seq_generator = nn.Conv1d(channels, self.n_atoms, 1, 1, 0)


        self.to_impulse = nn.Conv1d(channels, self.n_atoms, 1, 1, 0)
        self.to_transfer = nn.Conv1d(channels, self.n_atoms, 1, 1, 0)

        self.verb = ReverbGenerator(channels, 3, exp.samplerate, exp.n_samples)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):

        batch, _, n_samples = x.shape

        spec = exp.fb.forward(x, normalize=False)
        pos = pos_encoded(
            batch, spec.shape[-1], 16, device=x.device).permute(0, 2, 1)
        spec = torch.cat([pos, spec], dim=1)
        spec = self.reduce(spec)

        x = self.encoder(spec)

        context = torch.mean(x, dim=-1)

        # upsample to the correct sampling rate
        seq = self.seq_generator.forward(x)

        attn = torch.sigmoid(self.attn.forward(x))
        events, indices = sparsify_vectors(
            x, attn, self.k_sparse, normalize=False, dense=False)

        impulse_choice = self.to_impulse_choice.forward(events)
        impulse_choice = choice_softmax(impulse_choice)

        # d = self.impulse_dict.forward()
        # t = self.transfer_dict.forward()

        # impulses = impulse_choice @ d
        # impulses = torch.cat([
        #     impulses,
        #     torch.zeros(batch, self.k_sparse, exp.n_samples - self.atom_size, device=impulses.device)], dim=-1)

        # transfer_choice = self.to_transfer_choice.forward(events)
        # transfer_choice = choice_softmax(transfer_choice)
        # transfers = transfer_choice @ t

        # d = fft_convolve(impulses, transfers) + impulses

        d = self.atom_dictionary
        d = torch.cat([d, torch.zeros(self.n_atoms, exp.n_samples - self.atom_size, device=d.device)], dim=-1)[None, ...]

        seq = F.dropout(seq, 0.05)
        seq, indices, values = sparsify(
            seq, self.k_sparse, return_indices=True)

        print(seq.shape, d.shape)
        
        output = fft_convolve(seq, d)[..., :exp.n_samples]
        output = torch.sum(output, dim=1, keepdim=True)

        verb = self.verb.forward(context, output)
        return verb


model = Model(
    exp.model_dim,
    n_heads=4,
    n_layers=6,
    atom_size=512,
    n_atoms=512,
    k_sparse=128).to(device)

optim = optimizer(model, lr=1e-3)


disc = Discriminator(exp.model_dim).to(device)
disc_optim = optimizer(disc, lr=1e-3)


def train_gen(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    fake_feat, j = disc.forward(recon)
    real_feat, j = disc.forward(batch)
    loss = F.l1_loss(fake_feat, real_feat)
    loss.backward()
    optim.step()
    return loss, recon

def train_disc(batch):
    disc_optim.zero_grad()
    with torch.no_grad():
        recon = model.forward(batch)
    fj, _ = disc.forward(recon)
    rj, _ = disc.forward(batch)
    # loss = torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    return loss


feat = PsychoacousticFeature([128] * 6).to(device)

def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)

    # loss = exp.perceptual_loss(recon, batch)
    a, _ = feat.forward(recon)
    b, _ = feat.forward(batch)

    loss = F.mse_loss(a, b)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class DynamicDictionary(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None
    
    # def run(self):
    #     for i, item in enumerate(self.iter_items()):
    #         self.real = item
    #         if i % 2 == 0:
    #             g_loss, recon = train_gen(item)
    #             print('G', g_loss.item())
    #             self.fake = recon
    #         else:
    #             d_loss = train_disc(item)
    #             print('D', d_loss.item())
