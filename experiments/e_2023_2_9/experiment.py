
import numpy as np
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.dilated import DilatedStack
from modules.fft import fft_convolve, fft_shift
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, unit_norm
from modules.perceptual import PerceptualAudioModel
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.softmax import sparse_softmax
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from torch import nn
import torch
from torch.nn import functional as F
from modules import fft_frequency_decompose, fft_frequency_recompose

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
    return sparse_softmax(x, normalize=True)
    # return hard_softmax(x, invert=True)


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


class AnalysisBand(nn.Module):
    def __init__(
            self,
            n_atoms,
            atom_size,
            channels,
            kernel_size,
            k_sparse,
            vectorwise=False,
            gain=1):

        super().__init__()
        self.n_atoms = n_atoms
        self.channels = channels
        self.kernel_size = kernel_size
        self.atom_size = atom_size
        self.k_sparse = k_sparse
        self.vectorwise = vectorwise

        scale = zounds.LinearScale(zounds.FrequencyBand(
            1, exp.samplerate.nyquist), channels)
        bank = morlet_filter_bank(
            exp.samplerate, self.kernel_size, scale, 0.1, normalize=True)
        self.register_buffer('bank', torch.from_numpy(
            bank.real).float().view(channels, 1, kernel_size))

        scale = zounds.LinearScale(zounds.FrequencyBand(
            1, exp.samplerate.nyquist), self.n_atoms)
        bank = morlet_filter_bank(
            exp.samplerate, self.atom_size, scale, 0.1, normalize=True)

        self.atoms = nn.Parameter(torch.zeros(self.n_atoms, self.atom_size).uniform_(-0.01, 0.01))
        # self.atoms = nn.Parameter(torch.from_numpy(bank.real).float())

        # time-frequency mask
        mask = -torch.ones(1, 1, 3, self.atom_size // 2)
        mask[:, :, 1, self.atom_size // 4] = 1
        self.register_buffer('masking', mask)

        self.reduce = nn.Conv1d(self.channels * 2, channels, 1, 1, 0, bias=False)

        self.gain = nn.Parameter(torch.zeros(1).fill_(gain))
        self.norm = ExampleNorm()

        self.net = nn.Sequential(
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 3),
            DilatedBlock(channels, 9),
            DilatedBlock(channels, 27),
            DilatedBlock(channels, 81),
            DilatedBlock(channels, 1),
        )

        # self.net = UNet(self.channels)

        if self.vectorwise:
            self.attn = nn.Conv1d(self.channels, 1, 1, 1, 0)
            self.to_atoms = LinearOutputStack(
                self.channels, 3, out_channels=self.n_atoms)
            self.to_amps = LinearOutputStack(self.channels, 3, out_channels=1)
            self.to_shifts = LinearOutputStack(
                self.channels, 3, out_channels=1)
        else:
            self.reduce = nn.Conv1d(self.channels + 33, self.channels, 1, 1, 0, bias=False)
            self.to_atoms = nn.Conv1d(self.channels, self.n_atoms, 1, 1, 0, bias=False)


    def forward(self, x):

        batch = x.shape[0]
        x = x.view(batch, 1, -1)
        n_samples = x.shape[-1]

        x = F.conv1d(x, self.bank, padding=self.kernel_size // 2)[..., :n_samples]

        # x = self.norm(x)

        pos = pos_encoded(batch, n_samples, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)
        features = x = self.net(x)

        context = torch.max(features, dim=-1)[0]

        if self.vectorwise:
            attn = torch.sigmoid(self.attn(x))
            latents, indices = sparsify_vectors(
                x, attn, n_to_keep=self.k_sparse)

            shifts = torch.sigmoid(self.to_shifts(latents))
            amps = torch.relu(self.to_amps(latents))
            atoms = (softmax(self.to_atoms(latents)) @ self.atoms) * amps
            atoms = torch.cat([atoms, torch.zeros(
                batch, self.k_sparse, n_samples - self.atom_size, device=x.device)], dim=-1)
            atoms = fft_shift(atoms, shifts)
            atoms = torch.sum(atoms, dim=1, keepdim=True)
            return atoms, context
        else:

            x = self.to_atoms(x)
            atom_feat = x
            x = torch.relu(x)

            # perform time frequency masking
            # masked = x.view(batch, 1, self.n_atoms, n_samples)
            # masked = F.conv2d(masked, self.masking, padding=(1, self.atom_size // 4))
            # masked = masked[..., :n_samples]
            # masked = masked.view(batch, self.n_atoms, n_samples)

            x = F.dropout(x, p=0.01)

            # x = self.gain * x

            x, indices, values = sparsify(
                x, self.k_sparse, return_indices=True)
            
            # sparse = x

            # atom_indices = indices // n_samples
            # time_indices = indices % n_samples

            # print('------------------------------------------')
            # print(n_samples, atom_feat.std().item())
            # print('unique atoms', len(set(atom_indices.view(-1).data.cpu().numpy())))
            # print('unique times', len(set(time_indices.view(-1).data.cpu().numpy())))

            # output = torch.zeros(batch, 1, n_samples * 2, device=x.device)

            # half_atom = self.atom_size // 2


            # for i in range(batch):
            #     for e in range(self.k_sparse):
            #         atom = self.atoms[atom_indices[i, e]] * values[i, e]
            #         time_index = time_indices[i, e]

            #         start = time_index - half_atom
            #         end = time_index + half_atom

            #         signal_start = torch.clamp(start, 0, n_samples)
            #         signal_end = torch.clamp(end, 0, n_samples)

            #         atom_start = torch.clamp(-start, 0, self.atom_size)
            #         atom_end = self.atom_size - \
            #             torch.clamp((end - n_samples), 0, self.atom_size)


            #         output[:, :, signal_start: signal_end] += atom[atom_start: atom_end]

            # output = output[..., :n_samples]

            x = F.pad(x, (0, 1))
            output = F.conv_transpose1d(x, self.atoms[:, None, :], padding=self.atom_size // 2)

            return output, context


class TransferFunctionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.lowest_band = 512

        self.channels = 128
        self.kernel_size = 128

        self.atoms_to_keep = 8
        self.n_atoms = 1024

        start = int(np.log2(self.lowest_band))
        end = int(np.log2(exp.n_samples))
        self.n_bands = end - start

        self.bands = nn.ModuleDict({str(2**k): AnalysisBand(
            self.n_atoms,
            atom_size=(2**k) // 8,
            channels=self.channels,
            kernel_size=self.kernel_size,
            k_sparse=self.atoms_to_keep,
            vectorwise=False) for k in range(start, end)})

        self.to_context = LinearOutputStack(
            self.channels, 3, out_channels=self.channels, in_channels=self.channels * self.n_bands)

        self.verb = ReverbGenerator(
            exp.model_dim, 3, exp.samplerate, exp.n_samples)

        self.apply(lambda p: exp.init_weights(p))


    def forward(self, x):
        batch = x.shape[0]
        n_samples = x.shape[-1]
        bands = fft_frequency_decompose(x, self.lowest_band)


        audio = {}
        feat = []
        for k, layer in self.bands.items():
            signal = bands[int(k)]
            a, f = layer.forward(signal)
            audio[int(k)] = a
            feat.append(f)

        f = torch.cat(feat, dim=-1)
        context = self.to_context(f)

        dry = fft_frequency_recompose(audio, n_samples)

        verb = self.verb.forward(context, dry)
        return verb


# loss_model = PerceptualAudioModel(exp, norm_second_order=False).to(device)
loss_model = PsychoacousticFeature([128] * 6)


def multiband_features(x):
    bands = fft_frequency_decompose(x, 512)
    return torch.cat([b for b in bands.values()], dim=-1)


def experiment_loss(a, b):
    # a = stft(a)
    # b = stft(b)
    # return F.mse_loss(a, b)

    # return exp.perceptual_loss(a, b)

    # return loss_model.loss(a, b)

    a, _ = loss_model.forward(a)
    b, _ = loss_model.forward(b)
    return F.mse_loss(a, b)

    # a = multiband_features(a)
    # b = multiband_features(b)
    # return F.l1_loss(a, b)


model = TransferFunctionModel().to(device)


optim = optimizer(model, lr=1e-3)

feature_maps = {}


def train(batch):

    optim.zero_grad()
    recon = model.forward(batch)

    # loss = 0
    # for s in sp:

    #     nps = s.data.cpu().numpy()
    #     indices = np.nonzero(nps)
    #     nps[indices] = nps[indices] + (1 - nps[indices])
    #     feature_maps[s.shape[-1]] = nps

    #     channel_means = torch.mean(s, dim=(0, 2))
    #     overall_mean = torch.mean(s)
    #     loss = loss + torch.sum(channel_means + overall_mean)

    loss = experiment_loss(recon, batch)


    loss.backward()
    optim.step()

    return loss, recon


@readme
class MultibandKSparse(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.feature_maps = feature_maps
