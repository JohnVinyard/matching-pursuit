from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.ddsp import NoiseModel, noise_bank2
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.pos_encode import pos_encoded
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

        self.spectral_shapes = nn.Parameter(torch.zeros(self.n_atoms, self.n_coeffs, self.n_frames).uniform_(-1, 1))
    
    def forward(self):
        # normed = unit_norm(torch.sin(self.spectral_shapes), axis=1)
        # normed = normed.repeat(1, 1, self.n_frames)
        # with_envelope = normed * (self.envelopes ** 2)
        # params = self.spectral_shapes
        as_samples = noise_bank2(self.spectral_shapes)
        return as_samples.view(1, self.n_atoms, self.n_samples)

class TransferFunctionDictionary(nn.Module):
    def __init__(self, n_atoms, n_samples, n_frequencies, n_frames):
        super().__init__()
        self.n_atoms = n_atoms
        self.n_samples = n_samples
        self.n_frequencies = n_frequencies
        self.n_frames = n_frames

        self.osc = nn.Parameter(torch.zeros(self.n_atoms, self.n_frequencies).uniform_(0, 1))
        self.amp = nn.Parameter(torch.zeros(self.n_atoms, self.n_frequencies).uniform_(0, 1))


        self.decay = nn.Parameter(torch.zeros(self.n_atoms, self.n_frequencies).uniform_(0, 1))
    
    def forward(self):
        osc = torch.sigmoid(self.osc.view(self.n_atoms, self.n_frequencies, 1)) ** 2
        freq = (osc * np.pi).repeat(1, 1, self.n_samples)
        osc = torch.sin(torch.cumsum(freq, dim=-1))
        osc = osc * (self.amp.view(self.n_atoms, self.n_frequencies, 1) ** 2)

        dec = 0.7 + (torch.sigmoid(self.decay.view(self.n_atoms, self.n_frequencies, 1).repeat(1, 1, self.n_frames)) * 0.299999)
        dec = torch.exp(torch.cumsum(torch.log(dec), dim=-1))
        # dec = torch.cumprod(dec, dim=-1)
        dec = F.interpolate(dec, size=self.n_samples, mode='linear')

        osc = osc * dec
        osc = torch.mean(osc, dim=1, keepdim=True)
        return osc.view(1, self.n_atoms, self.n_samples)


class Model(nn.Module):
    def __init__(self, channels, n_heads, n_layers, atom_size, n_atoms, k_sparse):
        super().__init__()

        self.k_sparse = k_sparse
        self.n_atoms = n_atoms
        self.atom_size = atom_size

        # self.encoder = DilatedStack(channels, [1, 3, 9, 1], dropout=0.1)

        self.encoder = nn.Sequential(
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 3),
            DilatedBlock(channels, 9),
            DilatedBlock(channels, 1),
        )
        
        self.reduce = nn.Conv1d(channels + 33, channels, 1, 1, 0)

        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)

        self.to_impulse_choice = LinearOutputStack(channels, 3, out_channels=self.n_atoms)
        self.to_transfer_choice = LinearOutputStack(channels, 3, out_channels=self.n_atoms)
        self.to_mixture = LinearOutputStack(channels, 3, out_channels=1)

        self.impulse_dict = ImpulseDictionary(n_atoms, self.atom_size)
        self.transfer_dict = TransferFunctionDictionary(n_atoms, exp.n_samples, 16, exp.n_samples // exp.step_size)

        self.seq_generator = ConvUpsample(
            channels,
            channels,
            start_size=128,
            end_size=exp.n_samples,
            from_latent=False,
            mode='learned',
            out_channels=self.k_sparse,
            batch_norm=False)
        

        self.to_impulse = nn.Conv1d(channels, self.n_atoms, 1, 1, 0)
        self.to_transfer = nn.Conv1d(channels, self.n_atoms, 1, 1, 0)

        self.verb = ReverbGenerator(channels, 3, exp.samplerate, exp.n_samples)

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, x):

        batch, _, n_samples = x.shape

        spec = exp.pooled_filter_bank(x)
        pos = pos_encoded(batch, spec.shape[-1], 16, device=x.device).permute(0, 2, 1)
        spec = torch.cat([pos, spec], dim=1)
        spec = self.reduce(spec)

        x = self.encoder(spec)

        context = torch.mean(x, dim=-1)

        # d = torch.cat([self.d, torch.zeros(1, self.n_atoms, exp.n_samples - self.atom_size, device=x.device)], dim=-1)

        # upsample to the correct sampling rate
        seq = self.seq_generator.forward(x)

        attn = torch.sigmoid(self.attn.forward(x))
        events, indices = sparsify_vectors(x, attn, self.k_sparse, normalize=False, dense=False)

        impulse_choice = self.to_impulse_choice.forward(events)
        impulse_choice = choice_softmax(impulse_choice)

        d = self.impulse_dict.forward()
        t = self.transfer_dict.forward()

        impulses = impulse_choice @ d
        impulses = torch.cat([
            impulses, 
            torch.zeros(batch, self.k_sparse, exp.n_samples - self.atom_size, device=impulses.device)], dim=-1)

        transfer_choice = self.to_transfer_choice.forward(events)
        transfer_choice = choice_softmax(transfer_choice)
        transfers = transfer_choice @ t

        mixture = (torch.sin(self.to_mixture.forward(events)) + 1) * 0.5

        d = ((1 - mixture) * fft_convolve(impulses, transfers)) + (mixture * impulses)
        # d = fft_convolve(impulses, transfers) + impulses

        seq = F.dropout(seq, 0.01)
        seq, indices, values = sparsify(
            seq, self.k_sparse, return_indices=True)
    
        output = fft_convolve(seq, d)[..., :exp.n_samples]
        output = torch.sum(output, dim=1, keepdim=True)        

        verb = self.verb.forward(context, output)
        return verb


model = Model(
    exp.model_dim,
    n_heads=4,
    n_layers=6,
    atom_size=4096,
    n_atoms=512,
    k_sparse=8).to(device)

optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class DynamicDictionary(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
