
import numpy as np
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.ddsp import NoiseModel
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm, unit_norm
from modules.perceptual import PerceptualAudioModel
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import sparsify_vectors
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from torch import nn
import torch
from torch.nn import functional as F

from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def softmax(x):
    # return torch.softmax(x, dim=-1)
    # return sparse_softmax(x, normalize=True)
    return hard_softmax(x, invert=False)



class ImpulseBank(nn.Module):
    def __init__(self, n_atoms):
        super().__init__()
        self.n_atoms = n_atoms
        self.latent_size = 16
        self.latent_frames = 16

        self.latents = nn.Parameter(torch.zeros(
            self.n_atoms, self.latent_size, self.latent_frames).uniform_(-1, 1))

        self.model = NoiseModel(
            self.latent_size, self.latent_frames, 64, 512, exp.model_dim, squared=True, mask_after=1)
    
    def forward(self, x):
        # first, generate the dictionary
        d = self.model.forward(self.latents).view(-1, 512)
        d = max_norm(d)
        return x @ d


class TransferFunctionModel(nn.Module):
    """
    TODO:
        - do non-causal conv work better
        - does a transformer do better at analysis
        - do things still work if impulses are constrained to be band-limited noise?
        - do graphs of transfer functions yield interesting results?
        - can I use hard attention to further reduce the amount of info per atom?
    """

    def __init__(
            self,
            n_samples,
            channels,
            n_atoms,
            atom_size,
            n_transfers,
            transfer_size,
            n_events):

        super().__init__()
        self.n_samples = n_samples
        self.channels = channels
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.n_transfers = n_transfers
        self.transfer_size = transfer_size
        self.n_events = n_events

        scale = zounds.MelScale(zounds.FrequencyBand(
            20, exp.samplerate.nyquist), 128)
        bank = morlet_filter_bank(
            exp.samplerate, 512, scale, 0.1, normalize=True)
        self.register_buffer('bank', torch.from_numpy(
            bank.real).float().view(128, 1, 512))

        scale = zounds.MelScale(zounds.FrequencyBand(
            20, 2000), self.n_transfers)
        bank = morlet_filter_bank(
            exp.samplerate, self.transfer_size, scale, 0.01, normalize=True)
        bank *= np.linspace(1, 0, self.n_samples) ** 10

        # self.atoms = nn.Parameter(torch.zeros(
            # self.n_atoms, self.atom_size).uniform_(-1, 1))
        
        self.atoms = ImpulseBank(self.n_atoms)
        self.transfer = nn.Parameter(
            torch.from_numpy(bank.real).float())

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        # TODO: In order to make the locations where atoms are placed valid,
        # this should probably be the opposite of causal convolutions
        self.net = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1]),
        )

        self.norm = ExampleNorm()
        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)

        self.to_atoms = LinearOutputStack(
            channels, 3, out_channels=self.n_atoms)
        self.to_transfer = LinearOutputStack(
            channels, 3, out_channels=self.n_transfers)
        self.to_atom_amp = LinearOutputStack(channels, 3, out_channels=1)
        # self.to_transfer_amp = LinearOutputStack(channels, 3, out_channels=1)
        self.to_shifts = LinearOutputStack(channels, 3, out_channels=1)

        self.step_size = 1
        self.shift_width = self.step_size * 2
        self.shift_amt = self.shift_width / self.n_samples

        self.apply(lambda x: exp.init_weights(x))

    def give_atoms_unit_norm(self):
        pass

    @property
    def normed_atoms(self):
        return max_norm(self.atoms)

    @property
    def normed_transfer(self):
        return max_norm(self.transfer)

    def forward(self, x):
        batch = x.shape[0]

        # TODO: average and subtract, also all convolutions should be
        # the opposite of causal (only future values are represented)
        x = F.conv1d(x, self.bank, padding=256)
        x = x[..., :self.n_samples]
        x = self.net(x)
        x = self.norm(x)

        context, indices = torch.max(x, dim=-1)

        attn = torch.sigmoid(self.attn(x))

        # the ability for events to start at any sample is essential,
        # otherwise, the frequency and harmonics of the frame rate dominate
        latents, indices = sparsify_vectors(
            x, attn, self.n_events, normalize=False, dense=False)

        atoms = self.to_atoms.forward(latents)
        transfers = self.to_transfer.forward(latents)
        atom_amps = torch.sigmoid(self.to_atom_amp.forward(latents))
        # transfer_amps = self.to_transfer_amp.forward(latents) ** 2
        shifts = torch.tanh(self.to_shifts(latents)) * self.shift_amt

        atoms = softmax(atoms)
        transfers = softmax(transfers)

        # atoms = (atoms @ self.normed_atoms) * atom_amps
        atoms = self.atoms.forward(atoms) * atom_amps
        transfers = (transfers @ self.normed_transfer)  # * transfer_amps

        atoms = F.pad(atoms, (0, self.transfer_size - self.atom_size))

        x = fft_convolve(atoms, transfers) + atoms

        if self.step_size > 1:
            x = fft_shift(x, shifts)[..., :self.n_samples]

        output = torch.zeros(batch, 1, self.n_samples * 2, device=x.device)

        for i in range(batch):
            for e in range(self.n_events):
                index = indices[i, e] * self.step_size
                signal = x[i, e]
                output[:, :, index: index + self.transfer_size] += signal

        dry = output[..., :self.n_samples]
        return dry

        verb = self.verb.forward(context, dry)
        return verb


loss_model = PerceptualAudioModel(exp, norm_second_order=False).to(device)


def experiment_loss(a, b):
    return exp.perceptual_loss(a, b)


model = TransferFunctionModel(
    n_samples=exp.n_samples,
    channels=exp.model_dim,
    n_atoms=512,
    atom_size=512,
    n_transfers=512,
    transfer_size=exp.n_samples,
    n_events=64).to(device)


optim = optimizer(model, lr=1e-4)


def train(batch):
    model.give_atoms_unit_norm()
    optim.zero_grad()
    recon = model.forward(batch)
    loss = experiment_loss(recon, batch)
    loss.backward()
    optim.step()

    return loss, recon


@readme
class KSparse(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
