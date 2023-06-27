
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from experiments.e_2023_3_30.experiment import dirac_impulse
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.activation import unit_sine
from modules.fft import fft_convolve
from modules.matchingpursuit import sparse_feature_map
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.softmax import hard_softmax
from modules.sparse import soft_dirac, sparsify_vectors
from modules.transfer import PosEncodedImpulseGenerator
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from upsample import PosEncodedUpsample
from util import device
from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

d_size = 256
kernel_size = 256
sparse_coding_iterations = 16

band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, d_size)
d = morlet_filter_bank(exp.samplerate, kernel_size, scale,
                       np.linspace(0.25, 0.01, d_size)).real
d = torch.from_numpy(d).float().to(device)
d.requires_grad = True


def generate(batch_size):
    total_events = batch_size * sparse_coding_iterations
    amps = torch.zeros(total_events, device=device).uniform_(0.9, 1)
    positions = torch.zeros(total_events, device=device).uniform_(0, 1)
    atom_indices = (torch.zeros(total_events).uniform_(0, 1) * d_size).long()
    output = _inner_generate(
        batch_size, total_events, amps, positions, atom_indices)
    return output


def _inner_generate(batch_size, total_events, amps, positions, atom_indices):
    output = torch.zeros(total_events, exp.n_samples, device=device)
    for i in range(total_events):
        index = atom_indices[i]
        pos = positions[i]
        amp = amps[i]
        signal = torch.zeros(exp.n_samples, device=device)
        signal[:kernel_size] = unit_norm(d[index]) * amp
        signal = fft_shift(signal, pos)[..., :exp.n_samples]
        output[i] = signal

    output = output.view(batch_size, sparse_coding_iterations, exp.n_samples)
    output = torch.sum(output, dim=1, keepdim=True)
    return output


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        channels = 512

        self.embed = nn.Linear(128 + 33, channels)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            encoder, 
            6, 
            norm=nn.LayerNorm((128, 512))
        )
    

        self.to_amp = nn.Linear(channels, 1)
        self.to_pos = nn.Linear(channels, 33)
        self.to_atom = nn.Linear(channels, d_size)

        self.impulse = PosEncodedImpulseGenerator(
            512, 
            exp.n_samples, 
            softmax=lambda x: torch.softmax(x, dim=-1))

        self.up = PosEncodedUpsample(
            channels, channels, size=sparse_coding_iterations, layers=3, out_channels=channels)
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):
        x = x.view(-1, 1, exp.n_samples)
        x = exp.pooled_filter_bank(x).permute(0, 2, 1)
        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.embed(x)
        x = self.encoder(x)
        x = torch.sum(x, dim=1)
        # x = x[:, -1, :]

        x = self.up(x).permute(0, 2, 1)

        amp = torch.abs(self.to_amp(x))

        # pos = torch.sigmoid(self.to_pos(x))

        pos = self.to_pos(x)
        spike, _ = self.impulse(pos.view(-1, 33))
        spike = spike.view(-1, sparse_coding_iterations, exp.n_samples)

        atom = torch.softmax(self.to_atom(x), dim=-1)

        atoms = (atom @ d) * amp
        atoms = F.pad(atoms, (0, exp.n_samples - kernel_size))

        # output = fft_shift(atoms, pos)[..., :exp.n_samples]
        output = fft_convolve(spike, atoms)
        output = torch.sum(output, dim=1, keepdim=True)
        output = max_norm(output)
        return output


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def mse_loss(a, b):
    # return torch.abs(a - b).sum()
    return F.mse_loss(a, b)


def fft_loss(a, b):
    a = torch.fft.rfft(a, dim=-1)
    b = torch.fft.rfft(b, dim=-1)
    mag_loss = F.mse_loss(torch.abs(a), torch.abs(b))
    phase_loss = F.mse_loss(torch.angle(a), torch.angle(b))
    return mag_loss + phase_loss


def     xself_similarity_loss(a: torch.Tensor, b: torch.Tensor):
    a = a.unfold(-1, 128, 64)
    b = b.unfold(-1, 128, 64)
    a_sim = torch.cdist(a, b)
    b_sim = torch.cdist(b, b)
    loss = torch.abs(a_sim - b_sim).sum()
    return loss

def spec_similarity_loss(a: torch.Tensor, b: torch.Tensor):
    a = exp.pooled_filter_bank(a)
    b = exp.pooled_filter_bank(b)

    diff = torch.abs(a[:, :, None, :] - b[:, :, :, None])

    # a_sim = torch.cdist(a, b)
    # b_sim = torch.cdist(b, b)
    # loss = torch.abs(a_sim - b_sim).sum()

    loss = diff.sum()
    return loss


def sparse_loss(a, b):
    a = sparse_feature_map(a, d, n_steps=sparse_coding_iterations, device=device)
    b = sparse_feature_map(b, d, n_steps=sparse_coding_iterations, device=device)
    orig = F.mse_loss(a, b)

    
    a1 = a.unfold(-1, 512, 256).sum(dim=-1)
    b1 = b.unfold(-1, 512, 256).sum(dim=-1)
    small = F.mse_loss(a1, b1)

    a2 = a.unfold(-1, 4096, 2048).sum(dim=-1)
    b2 = b.unfold(-1, 4096, 2048).sum(dim=-1)
    med = F.mse_loss(a2, b2)

    a3 = a.unfold(-1, 8192, 4096).sum(dim=-1)
    b3 = b.unfold(-1, 8192, 4096).sum(dim=-1)
    lg = F.mse_loss(a3, b3)

    a4 = a.unfold(-1, 16384, 8192).sum(dim=-1)
    b4 = b.unfold(-1, 16384, 8192).sum(dim=-1)
    xlg = F.mse_loss(a4, b4)

    return orig# + small + med + lg + xlg

def experiment_loss(a, b):
    # return spec_similarity_loss(a, b)
    return self_similarity_loss(a, b)
    # return mse_loss(a, b)
    # return sparse_loss(a, b)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = experiment_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class SchedulingExperiment(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)

    def run(self):

        g = generate(self.batch_size)

        for i, item in enumerate(self.iter_items()):
            self.real = g
            l, r = train(g.clone().detach(), i)
            self.fake = r
            print('ITER', i, l.item())
            self.after_training_iteration(l)
            g = generate(self.batch_size)
