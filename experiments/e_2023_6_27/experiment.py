
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from fft_shift import fft_shift
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.sparse import soft_dirac
from modules.transfer import ImpulseGenerator, PosEncodedImpulseGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import PosEncodedUpsample
from util import device
from util.readmedocs import readme
from itertools import count


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

d_size = 256
kernel_size = 256
sparse_coding_iterations = 16

band = zounds.FrequencyBand(20, 2000)
scale = zounds.MelScale(band, d_size)

d = morlet_filter_bank(
    exp.samplerate, 
    kernel_size, 
    scale, 
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
    output = max_norm(output)
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


class Scheduler(nn.Module):
    def __init__(self, schedule_type='impulse', n_frames=512):
        super().__init__()
        self.schedule_type = schedule_type
        self.n_frames = n_frames

        if schedule_type == 'impulse':
            self.params = nn.Parameter(
                torch.zeros(sparse_coding_iterations, n_frames).uniform_(-1, 1))
            self.gen = ImpulseGenerator(exp.n_samples, softmax=lambda x: torch.softmax(x, dim=-1))
        else:
            self.params = nn.Parameter(
                torch.zeros(sparse_coding_iterations, 33).uniform_(-1, 1))
            self.gen = PosEncodedImpulseGenerator(
                n_frames, exp.n_samples, softmax=lambda x: torch.softmax(x, dim=-1), scale_frequencies=True)
    
    def forward(self, x, softmax):
        if self.schedule_type == 'impulse':
            inp = x if x is not None else self.params
            impulses = self.gen.forward(inp.view(-1, self.n_frames), softmax=softmax)
            impulses = impulses.view(1, sparse_coding_iterations, exp.n_samples)
        else:
            impulses, _ = self.gen.forward(x if x is not None else self.params, softmax=softmax)
            impulses = impulses.view(1, sparse_coding_iterations, exp.n_samples)
        
        return impulses


class Encoder(nn.Module):
    def __init__(self, channels, impulse_frames):
        super().__init__()
        self.channels = channels
        self.impulse_frames = impulse_frames
        self.embed = nn.Linear(exp.n_bands + 33, channels)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 5, norm=nn.LayerNorm((128, channels)))
        self.up = PosEncodedUpsample(channels, channels, sparse_coding_iterations, channels, 5)

        self.to_amps = nn.Linear(channels, 1)
        self.to_positions = nn.Linear(channels, impulse_frames)
        self.to_atoms = nn.Linear(channels, d_size)
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        batch, channels, frames = x.shape
        x = x.permute(0, 2, 1)
        pos = pos_encoded(batch, frames, n_freqs=16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.embed(x)
        x = self.encoder(x)
        # x = torch.sum(x, dim=1)
        x = x[:, -1, :]
        x = self.up(x).permute(0, 2, 1) # (batch, n_atoms, channels)

        amps = self.to_amps(x) ** 2
        pos = self.to_positions(x)
        atoms = self.to_atoms(x)

        return amps, pos, atoms



class Model(nn.Module):
    def __init__(
            self, 
            n_scheduling_frames=512, 
            training_softmax=lambda x: soft_dirac(x, dim=-1), 
            inference_softmax=lambda x: soft_dirac(x, dim=-1),
            scheduler_type='impulse',
            encode=False):
        
        super().__init__()

        self.encode = encode
        self.training_softmax = training_softmax
        self.inference_softmax = inference_softmax

        self.scheduler_type = scheduler_type
        self.n_scheduling_frames = n_scheduling_frames

        self.encoder = Encoder(1024, n_scheduling_frames)

        self.amps = nn.Parameter(
            torch.zeros(sparse_coding_iterations, 1).uniform_(0, 1))
        
        self.atom_selection = nn.Parameter(
            torch.zeros(sparse_coding_iterations, d_size).uniform_(-1, 1))
        
        self.scheduler = Scheduler(self.scheduler_type, self.n_scheduling_frames)

        self.apply(lambda x: exp.init_weights(x))

    
    @property
    def training_atom_softmax(self):
        return self.training_softmax
    
    @property
    def training_schedule_softmax(self):
        return self.training_softmax
    
    @property
    def inference_atom_softmax(self):
        return self.inference_softmax
    
    @property
    def inference_schedule_softmax(self):
        return self.inference_softmax
    
    def _core_forward(
            self, 
            x, 
            atom_softmax, 
            schedule_softmax, 
            atom_selection=None, 
            amps=None, 
            positions=None):
        
        sel = atom_softmax(atom_selection if atom_selection is not None else self.atom_selection)
        atoms = (sel @ d)
        with_amp = atoms * (amps if amps is not None else self.amps)
        with_amp = with_amp.view(1, sparse_coding_iterations, kernel_size)
        atoms = F.pad(with_amp, (0, exp.n_samples - kernel_size))

        impulses = self.scheduler.forward(positions, schedule_softmax)

        final = fft_convolve(impulses, atoms)
        final = torch.sum(final, dim=1, keepdim=True)
        final = max_norm(final)
        return final
    
    

    def forward(self, x):
        if self.encode:
            amps, pos, atoms = self.encoder.forward(x)
        else:
            amps, pos, atoms = None, None, None

        result = self._core_forward(
            x, 
            self.training_atom_softmax, 
            self.training_schedule_softmax,
            atom_selection=atoms,
            amps=amps,
            positions=pos)
        
        return result

model = Model(
    n_scheduling_frames=512, 
    training_softmax=lambda x: soft_dirac(x, dim=-1),
    inference_softmax=lambda x: soft_dirac(x, dim=-1),
    scheduler_type='impulse',
    encode=True
).to(device)

optim = optimizer(model, lr=1e-3)

def exp_loss(a, b):
    return exp.perceptual_loss(a, b)

def train(batch, i):
    optim.zero_grad()

@readme
class NoGridExperiment(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        item = generate(1)
        self.real = item

        for i in count():
            item = item.clone().detach()

            optim.zero_grad()
            recon = model.forward(item)
            self.fake = recon
            loss = exp_loss(recon, item)
            loss.backward()
            optim.step()

            print(i, loss.item())
            self.after_training_iteration(loss)
    