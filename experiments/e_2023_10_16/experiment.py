
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from experiments.e_2023_10_12.experiment import Contextual
from fft_shift import fft_shift
from modules.activation import unit_sine
from modules.fft import fft_convolve
from modules.mixer import MixerStack
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.stft import stft
from modules.transfer import ImpulseGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 32
atom_size = 4096

def loss_func(
        target: torch.Tensor, 
        stems: torch.Tensor, 
        amplitudes: torch.Tensor, 
        positions: torch.Tensor):
    
    batch, _, n_samples = target.shape
    _, n_events, atom_samples = stems.shape

    # TODO: I should find best position, amplitude and samples
    # one channel at a time, instead of all at once

    # Find the best matching positions for each atom produced
    # padded = F.pad(stems, (0, n_samples - atom_samples))
    padded = stems
    fm = fft_convolve(padded, target)[..., :n_samples]
    values, indices = torch.max(fm, dim=-1)
    actual_positions = indices / n_samples

    # shift the stems into the best possible positions
    shifted = fft_shift(padded, actual_positions[..., None])[..., :n_samples]
    # give them the best possible norms
    shifted = shifted * values[..., None]
    full = torch.sum(shifted, dim=1, keepdim=True)

    start_norm = torch.norm(target, dim=-1).mean()
    residual = target - full
    end_norm = torch.norm(residual, dim=-1).mean()
    norm_loss = start_norm - end_norm

    sample_loss = 0
    for channel in range(n_events):
        ch = shifted[:, channel: channel + 1, :]
        t = residual + ch

        # ch = stft(ch, 2048, 256, pad=True)
        # t = stft(t, 2048, 256, pad=True)

        sample_loss = sample_loss + F.mse_loss(ch, t.clone().detach())


    amp_loss = F.mse_loss(amplitudes.view(*values.shape), values.clone().detach())
    pos_loss = F.mse_loss(positions.view(*actual_positions.shape), actual_positions.clone().detach())

    return sample_loss + amp_loss + pos_loss + (-norm_loss)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
        
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x
    
class Contextual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.down = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 6, norm=nn.LayerNorm((128, channels)))

    def forward(self, x):
        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.down(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return x

# def single_channel_loss(target: torch.Tensor, recon: torch.Tensor):
#     ws = 2048
#     ss = 256

#     target = stft(target, ws, ss, pad=True)
#     full = torch.sum(recon, dim=1, keepdim=True)
#     full = stft(full, ws, ss, pad=True)

#     residual = target - full

#     loss = 0

    
#     for i in range(n_events):
#         ch = recon[:, i: i + 1, :]
#         ch = stft(ch, ws, ss, pad=True)
#         t = residual + ch
#         loss = loss + F.mse_loss(ch, t.clone().detach())
    
#     return loss


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1025, 1024)
        self.pos = nn.Parameter(torch.zeros(1, 128, 1024).uniform_(-0.1, 0.1))

        self.stack = nn.Sequential(
            DilatedBlock(1024, 1),
            DilatedBlock(1024, 3),
            DilatedBlock(1024, 9),
            DilatedBlock(1024, 1),
            DilatedBlock(1024, 3),
            DilatedBlock(1024, 9),
            DilatedBlock(1024, 1),
        )
        # self.stack = MixerStack(1025, 1024, 128, 6, attn_blocks=4, channels_last=True)


        
        n_atoms = 1024
        resolution = 1024
        self.resolution = resolution

        self.atom_size = atom_size
        self.atoms = nn.Parameter(torch.zeros(n_atoms, atom_size).uniform_(-1, 1))
        self.to_amps = nn.Linear(1024, 1)
        self.to_positions = nn.Linear(1024, 1)
        self.to_atoms = nn.Linear(1024, n_atoms)

        self.imp = ImpulseGenerator(exp.n_samples)

        self.apply(lambda x: exp.init_weights(x))

    
    def forward(self, x):
        spec = stft(x, 2048, 256, pad=True).view(-1, 128, 1025)

        x = self.embed(spec)
        x = x + self.pos

        x = x.permute(0, 2, 1)
        x = self.stack(x)
        x = x.permute(0, 2, 1)

        # x = self.stack(spec)

        # information should be "shipped" to the last N events
        x = x[:, -n_events:, :]

        amps = torch.abs(self.to_amps(x))
        positions = self.to_positions(x)

        sel = torch.softmax(self.to_atoms(x), dim=-1)
        atoms = sel @ self.atoms

        padded = F.pad(atoms, (0, exp.n_samples - self.atom_size))
        normed = padded = unit_norm(padded)

        scaled = padded * amps

        # shifted = fft_convolve(scaled, imps)[..., :exp.n_samples]

        shifted = fft_shift(scaled, positions)[..., :exp.n_samples]
        full = torch.sum(shifted, dim=1, keepdim=True)

        return normed, amps, positions, full, sel


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()

    atoms, amps, positions, recon, selection = model.forward(batch)

    # model should commit to a single atom as much as possible
    values, indices = torch.max(selection, dim=-1)
    confidence_loss = torch.abs(1 - values).mean()

    loss = loss_func(batch, atoms, amps, positions) + confidence_loss
    
    loss.backward()
    optim.step()

    recon = max_norm(recon)
    return loss, recon

@readme
class DecomposedLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    