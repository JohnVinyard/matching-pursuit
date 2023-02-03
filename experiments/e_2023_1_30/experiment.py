
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.perceptual import PerceptualAudioModel
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import sparsify, sparsify_vectors
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
    # return sparse_softmax(x)
    return hard_softmax(x)


class TransferFunctionModel(nn.Module):
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

        self.atoms = nn.Parameter(torch.zeros(
            self.n_atoms, self.atom_size).uniform_(-1, 1))
        self.transfer = nn.Parameter(torch.zeros(
            self.n_transfers, self.transfer_size).uniform_(-1, 1))

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        # TODO: In order to make the locations where atoms are placed valid,
        # this should probably be the opposite of causal convolutions
        self.net = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1]),
        )

        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)

        self.to_atoms = LinearOutputStack(
            channels, 3, out_channels=self.n_atoms)
        self.to_transfer = LinearOutputStack(
            channels, 3, out_channels=self.n_transfers)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        batch = x.shape[0]

        x = F.conv1d(x, self.bank, padding=256)
        x = x[..., :self.n_samples]

        x = self.net(x)
        context, indices = torch.max(x, dim=-1)

        attn = torch.relu(self.attn(x))

        x = F.dropout(x, p=0.05)
        latents, indices = sparsify_vectors(
            x, attn, self.n_events, normalize=False, dense=False)

        atoms = self.to_atoms.forward(latents)
        transfers = self.to_transfer.forward(latents)

        atoms = softmax(atoms)
        transfers = softmax(transfers)

        atoms = atoms @ self.atoms
        transfers = transfers @ self.transfer

        atoms = F.pad(atoms, (0, self.transfer_size - self.atom_size))

        x = fft_convolve(atoms, transfers) + atoms

        output = torch.zeros(batch, 1, self.n_samples * 2, device=x.device)

        for i in range(batch):
            for e in range(self.n_events):
                index = indices[i, e]
                output[:, :, index: index + self.transfer_size] += x[i, e]

        dry = output[..., :self.n_samples]
        verb = self.verb.forward(context, dry)
        return verb


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.n_samples = exp.n_samples

        scale = zounds.MelScale(zounds.FrequencyBand(
            20, exp.samplerate.nyquist), 128)
        bank = morlet_filter_bank(
            exp.samplerate, 512, scale, 0.1, normalize=True)
        self.register_buffer('bank', torch.from_numpy(
            bank.real).float().view(128, 1, 512))

        self.k_sparse = 32
        self.n_atoms = 2048
        self.atom_size = 2048

        self.atoms = nn.Parameter(torch.zeros(
            self.n_atoms, self.atom_size).uniform_(-1, 1))
        self.transfer = nn.Parameter(torch.zeros(
            self.n_atoms, self.atom_size).uniform_(-1, 1))

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        # TODO: In order to make the locations where atoms are placed valid,
        # this should probably be the opposite of causal convolutions
        self.net = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1]),
        )

        self.to_atoms = nn.Conv1d(exp.model_dim, self.n_atoms, 1, 1, 0)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, orig, pooled, spec):
        batch = orig.shape[0]

        x = F.conv1d(orig, self.bank, padding=256)
        x = x[..., :self.n_samples]

        x = self.net(x)
        context, indices = torch.max(x, dim=-1)

        x = self.to_atoms(x)

        x = F.dropout(x, p=0.05)

        output, indices, values = sparsify(
            x, self.k_sparse, return_indices=True)

        atom_indices = indices // self.n_samples
        time_indices = indices % self.n_samples

        output = torch.zeros(batch, 1, self.n_samples * 2)

        for i in range(batch):
            for e in range(self.k_sparse):
                atom = self.atoms[atom_indices[i, e]] * values[i, e]
                time_index = time_indices[i, e]

                output[:, :, time_index: time_index + self.atom_size] += atom

        output = output[..., :self.n_samples]

        verb = self.verb.forward(context, output)
        return verb


# TODO: Multi-band/samplerate version of this
loss_model = PerceptualAudioModel(exp, norm_second_order=False).to(device)


def experiment_loss(a, b):
    return loss_model.loss(a, b)


model = TransferFunctionModel(
    n_samples=exp.n_samples,
    channels=exp.model_dim,
    n_atoms=512,
    atom_size=512,
    n_transfers=512,
    transfer_size=8192,
    n_events=32).to(device)

optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    # loss = loss_model.loss(recon, batch)
    loss = experiment_loss(recon, batch)
    loss.backward()
    optim.step()

    return loss, recon


@readme
class KSparse(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
