
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.dilated import DilatedStack
from modules.fft import fft_convolve, simple_fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, unit_norm
from modules.perceptual import PerceptualAudioModel
from modules.pos_encode import ExpandUsingPosEncodings
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import VectorwiseSparsity, sparsify, sparsify_vectors, to_sparse_vectors_with_context
from modules.stft import stft
from modules.transfer import PosEncodedImpulseGenerator
from scalar_scheduling import pos_encoded
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from torch import nn
import torch
from torch.nn import functional as F
from upsample import PosEncodedUpsample

from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def softmax(x):
    return torch.softmax(x, dim=-1)
    # return sparse_softmax(x)
    # return hard_softmax(x)


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

        scale = zounds.MelScale(zounds.FrequencyBand(
            20, exp.samplerate.nyquist), self.n_transfers)
        bank = morlet_filter_bank(
            exp.samplerate, self.transfer_size, scale, 0.1, normalize=True)

        self.atoms = nn.Parameter(torch.zeros(
            self.n_atoms, self.atom_size).uniform_(-1, 1))
        self.transfer = nn.Parameter(torch.from_numpy(bank.real).float())

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        # TODO: In order to make the locations where atoms are placed valid,
        # this should probably be the opposite of causal convolutions
        self.net = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1], dropout=0.1),
        )

        self.norm = ExampleNorm()
        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)

        self.to_atoms = LinearOutputStack(
            channels, 3, out_channels=self.n_atoms)
        self.to_transfer = LinearOutputStack(
            channels, 3, out_channels=self.n_transfers)
        self.to_amp = LinearOutputStack(channels, 3, out_channels=1)

        self.apply(lambda x: exp.init_weights(x))

    def give_atoms_unit_norm(self):
        pass

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

        latents, indices = sparsify_vectors(
            x, attn, self.n_events, normalize=False, dense=False)

        atoms = self.to_atoms.forward(latents)
        transfers = self.to_transfer.forward(latents)
        amps = self.to_amp.forward(latents) ** 2

        atoms = softmax(atoms)
        transfers = softmax(transfers)

        atoms = atoms @ self.atoms
        transfers = transfers @ self.transfer

        atoms = F.pad(atoms, (0, self.transfer_size - self.atom_size))

        x = fft_convolve(atoms, transfers) + atoms
        x = x * amps

        output = torch.zeros(batch, 1, self.n_samples * 2, device=x.device)

        for i in range(batch):
            for e in range(self.n_events):
                index = indices[i, e]
                output[:, :, index: index + self.transfer_size] += x[i, e]

        dry = output[..., :self.n_samples]

        verb = self.verb.forward(context, dry)
        return verb


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_samples = exp.n_samples

        self.k_sparse = 16
        self.n_atoms = 2048
        self.atom_size = 8192

        scale = zounds.MelScale(zounds.FrequencyBand(
            20, 3000), self.n_atoms)
        bank = morlet_filter_bank(
            exp.samplerate, self.atom_size, scale, 0.1, normalize=True)

        self.reduce = LinearOutputStack(
            exp.model_dim + 33, 3, out_channels=exp.model_dim)

        encoder = nn.TransformerEncoderLayer(
            exp.model_dim, 4, exp.model_dim, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 4)

        self.to_atom_vectors = PosEncodedUpsample(
            exp.model_dim,
            exp.model_dim,
            size=self.k_sparse,
            out_channels=exp.model_dim,
            layers=self.k_sparse)

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=self.k_sparse, dense=False, normalize=True, time=exp.n_frames)

        self.to_atom_choice = LinearOutputStack(
            exp.model_dim, 3, out_channels=self.n_atoms)
        self.to_atom_pos = LinearOutputStack(exp.model_dim, 3, out_channels=33)
        self.to_amp = LinearOutputStack(exp.model_dim, 3, out_channels=1)

        self.to_context = LinearOutputStack(exp.model_dim, 3)

        self.atoms = nn.Parameter(torch.from_numpy(bank.real).float())

        self.impulse = PosEncodedImpulseGenerator(
            self.n_samples, self.n_samples, softmax=hard_softmax)
        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)
        

        self.norm = ExampleNorm()

        self.apply(lambda x: exp.init_weights(x))

    def give_atoms_unit_norm(self):
        self.atoms.data[:] = unit_norm(self.atoms)

    def forward(self, x):
        batch = x.shape[0]

        spec = exp.pooled_filter_bank(x).permute(0, 2, 1)
        pos = pos_encoded(batch, spec.shape[-1], 16, device=spec.device)

        x = torch.cat([spec, pos], dim=-1)

        x = self.reduce(x)
        x = self.encoder(x)

        x = self.norm(x)

        context = self.to_context(x[:, -1, :])

        x, _ = self.sparse(x)


        pos = torch.sigmoid(self.to_atom_pos(x))
        impulses, _ = self.impulse.forward(pos.view(-1, 33))
        impulses = impulses.view(batch, self.k_sparse, self.n_samples)


        amp = self.to_amp(x) ** 2
        choice = sparse_softmax(self.to_atom_choice(x))
        atoms = choice @ self.atoms
        atoms = atoms * amp

        atoms = F.pad(atoms, (0, self.n_samples - self.atom_size))

        atoms = fft_convolve(impulses, atoms)
        atoms = atoms[..., :self.n_samples]

        signal = torch.mean(atoms, dim=1, keepdim=True)

        verb = self.verb.forward(context, signal)
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

        self.k_sparse = 64
        self.n_atoms = 2048
        self.atom_size = 1024

        # scale = zounds.MelScale(zounds.FrequencyBand(20, 3000), self.n_atoms)
        # bank = morlet_filter_bank(exp.samplerate, self.atom_size, scale, 0.1, normalize=True)

        self.atoms = nn.Parameter(torch.zeros(
            self.n_atoms, self.atom_size).uniform_(-1, 1))

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        # TODO: In order to make the locations where atoms are placed valid,
        # this should probably be the opposite of causal convolutions
        self.net = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1], dropout=0.25),
        )

        self.to_atoms = nn.Conv1d(exp.model_dim, self.n_atoms, 1, 1, 0)

        self.apply(lambda x: exp.init_weights(x))

    def give_atoms_unit_norm(self):
        self.atoms.data[:] = unit_norm(self.atoms)

    def forward(self, orig):
        batch = orig.shape[0]

        # x = F.conv1d(orig, self.bank, padding=256)
        # x = x[..., :self.n_samples]
        # x = torch.relu(x)
        # x = 20 * torch.log(x + 1e-8)

        # x = self.net(x)
        # context, indices = torch.max(x, dim=-1)

        # x = self.to_atoms(x)

        # atoms = F.pad(self.atoms[None, ...], (0, self.n_samples - self.atom_size))
        # x = fft_convolve(atoms, orig)
        x = F.conv1d(orig, self.atoms.view(self.n_atoms, 1,
                     self.atom_size), padding=self.atom_size // 2)
        x = x[..., :self.n_samples]
        x = F.dropout(x, p=0.05)

        output, indices, values = sparsify(
            x, self.k_sparse, return_indices=True)

        atom_indices = indices // self.n_samples
        time_indices = indices % self.n_samples

        output = torch.zeros(batch, 1, self.n_samples * 2, device=x.device)

        for i in range(batch):
            for e in range(self.k_sparse):
                atom = self.atoms[atom_indices[i, e]] * values[i, e]
                time_index = time_indices[i, e]

                output[:, :, time_index: time_index + self.atom_size] += atom

        output = output[..., :self.n_samples]
        return output

        verb = self.verb.forward(context, output)
        return verb


# TODO: Multi-band/samplerate version of this
loss_model = PerceptualAudioModel(exp, norm_second_order=False).to(device)
# loss_model = PsychoacousticFeature()


def experiment_loss(a, b):
    # return loss_model.loss(a, b)
    return F.mse_loss(a, b)


model = TransferFunctionModel(
    n_samples=exp.n_samples,
    channels=exp.model_dim,
    n_atoms=512,
    atom_size=512,
    n_transfers=512,
    transfer_size=exp.n_samples,
    n_events=128).to(device)

# model = Model().to(device)

# model = TransformerModel().to(device)

optim = optimizer(model, lr=1e-3)


def train(batch):
    model.give_atoms_unit_norm()
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
