
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.decompose import fft_frequency_recompose
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import soft_dirac, sparsify_vectors
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

d_size = 2048
kernel_size = 4096
sparse_coding_iterations = 32


verb = ReverbGenerator(128, 3, samplerate=exp.samplerate, n_samples=exp.n_samples).to(device)




def generate(batch_size):
    total_events = batch_size * sparse_coding_iterations
    amps = torch.zeros(total_events, device=device).uniform_(0.9, 1)
    positions = torch.zeros(total_events, device=device).uniform_(0, 1)
    atom_indices = (torch.zeros(total_events).uniform_(0, 1) * d_size).long()

    output = _inner_generate(
        batch_size, total_events, amps, positions, atom_indices)
    
    rm = torch.zeros((batch_size, verb.n_rooms)).to(device).uniform_(0, 1) ** 2
    rm = torch.softmax(rm, dim=-1)

    mx = torch.zeros((batch_size, 2)).to(device).normal_(0, 1)
    mx = torch.softmax(mx, dim=-1)

    output = verb.precomputed(output, mx, rm)
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


class ContextModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(

            nn.Conv1d(channels, channels, 3, 1, padding=1, dilation=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=3, dilation=3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=9, dilation=9),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=27, dilation=27),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

            nn.Conv1d(channels, channels, 3, 1, padding=1, dilation=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(channels),

        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x

class Scheduler(nn.Module):
    def __init__(self, n_frames=512):
        super().__init__()
        self.n_frames = n_frames

        

        self.params = nn.Parameter(
            torch.zeros(sparse_coding_iterations, n_frames).uniform_(-1, 1))
        self.gen = ImpulseGenerator(exp.n_samples, softmax=lambda x: torch.softmax(x, dim=-1))
        
    def forward(self, x, softmax):
        inp = x if x is not None else self.params
        impulses = self.gen.forward(inp.view(-1, self.n_frames), softmax=softmax)
        impulses = impulses.view(-1, sparse_coding_iterations, exp.n_samples)
        return impulses


class Encoder(nn.Module):
    def __init__(self, channels, impulse_frames, reduction=False):
        super().__init__()
        self.channels = channels
        self.impulse_frames = impulse_frames
        self.embed = nn.Linear(exp.n_bands + 33, channels)


        self.encoder = ContextModule(channels)
        self.reduction = reduction

        
        self.verb_params = nn.Linear(channels, channels)

        self.project_spec = nn.Linear(128, channels)
        self.project_pos = nn.Linear(33, channels)

        self.attn = nn.Linear(channels, 1)
        self.to_amps = nn.Linear(channels, 1)
        self.to_positions = nn.Linear(channels, impulse_frames)
        self.to_atoms = nn.Linear(channels, d_size)
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)

        

        batch, channels, frames = x.shape
        x = x.permute(0, 2, 1)

        spec = self.project_spec(x)

        pos = pos_encoded(batch, frames, n_freqs=16, device=x.device)
        pos = self.project_pos(pos)

        x = pos + spec
        
        x = self.encoder(x)

        # bring the position back in using a skip connection
        x = x + pos

        
        attn = self.attn(x).view(batch, frames)
        attn = torch.softmax(attn, dim=-1)
        x = x.permute(0, 2, 1)
        x, indices = sparsify_vectors(x, attn, n_to_keep=sparse_coding_iterations)
        
        agg = torch.sum(x, dim=1)
        verb_params = self.verb_params.forward(agg)
        verb_params = verb_params + agg

        amps = self.to_amps(x) ** 2
        pos = self.to_positions(x).view(batch, sparse_coding_iterations, self.impulse_frames)
        atoms = self.to_atoms(x)

        return amps, pos, atoms, verb_params



class MultibandAtoms(nn.Module):
    def __init__(self, size, bands, n_atoms):
        super().__init__()
        self.size = size
        self.bands = bands
        self.n_atoms = n_atoms

        bands = {}
        start = int(np.log2(size))
        for i in range(start, start - self.bands, -1):
            size = 2 ** i
            bands[str(size)] = nn.Parameter(torch.zeros(n_atoms, 1, size).normal_(0, 0.1))
        
        self.bands = nn.ParameterDict(bands)
    
    def forward(self):
        d = {int(k): v for k, v in self.bands.items()}
        atoms = fft_frequency_recompose(d, self.size)
        window = torch.ones(self.size, device=atoms.device)
        window[:10] = torch.linspace(0, 1, 10, device=atoms.device)
        window[-10:] = torch.linspace(1, 0, 10, device=atoms.device)
        atoms = atoms.view(self.n_atoms, self.size) * window[None, ...]
        return unit_norm(atoms, dim=-1)

class Model(nn.Module):
    def __init__(
            self, 
            n_scheduling_frames=512, 
            training_softmax=lambda x: soft_dirac(x, dim=-1), 
            inference_softmax=lambda x: soft_dirac(x, dim=-1),
            encode=False,
            reduction=False):
        
        super().__init__()

        self.atoms = nn.Parameter(torch.zeros(d_size, kernel_size).uniform_(-1, 1))
        # self.atoms = MultibandAtoms(kernel_size, 6, d_size)

        self.encode = encode
        self.training_softmax = training_softmax
        self.inference_softmax = inference_softmax

        self.n_scheduling_frames = n_scheduling_frames

        self.channels = 1024

        self.encoder = Encoder(self.channels, n_scheduling_frames, reduction=reduction)

        
        self.amps = nn.Parameter(
            torch.zeros(sparse_coding_iterations, 1).uniform_(0, 1))
        
        self.atom_selection = nn.Parameter(
            torch.zeros(sparse_coding_iterations, d_size).uniform_(-1, 1))
    
        
        self.scheduler = Scheduler(self.n_scheduling_frames)

        self.verb = ReverbGenerator(1024, 2, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((1024,)))

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
            positions=None,
            atom_dict=None):
        
        ad = d if atom_dict is None else atom_dict
        sel = atom_softmax(atom_selection if atom_selection is not None else self.atom_selection)
        atoms = (sel @ ad)
        with_amp = atoms * (amps if amps is not None else self.amps)
        with_amp = with_amp.view(-1, sparse_coding_iterations, kernel_size)
        atoms = F.pad(with_amp, (0, exp.n_samples - kernel_size))

        # env = self.noise(atom_selection)

        # atoms = self.res(env, atom_selection)
        # atoms = F.pad(atoms, (0, exp.n_samples - kernel_size)).view(-1, sparse_coding_iterations, exp.n_samples)

        impulses = self.scheduler.forward(positions, schedule_softmax)


        final = fft_convolve(impulses, atoms)
        final = torch.sum(final, dim=1, keepdim=True)
        return final
    

    def forward(self, x, training=True):
        if self.encode:
            amps, pos, atoms, verb_params = self.encoder.forward(x)
        else:
            amps, pos, atoms = None, None, None

        t = training

        result = self._core_forward(
            x, 
            self.training_atom_softmax if t else self.inference_atom_softmax, 
            self.training_schedule_softmax if t else self.inference_schedule_softmax,
            atom_selection=atoms,
            amps=amps,
            positions=pos,
            atom_dict=unit_norm(self.atoms))
        
        result = self.verb.forward(verb_params, result)

        
        return result

def training_softmax(x):
    """
    Produce a random mixture of the soft and hard functions, such
    that softmax cannot be replied upon.  This _should_ cause
    the model to gravitate to the areas where the soft and hard functions
    are near equivalent
    """
    mixture = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device).uniform_(0, 1)
    sm = torch.softmax(x, dim=-1)
    d = soft_dirac(x)
    return (d * mixture) + (sm * (1 - mixture))

model = Model(
    n_scheduling_frames=512, 
    training_softmax=lambda x: training_softmax(x),
    inference_softmax=lambda x: soft_dirac(x, dim=-1),
    encode=True,
    reduction=False
).to(device)

optim = optimizer(model, lr=1e-3)


def exp_loss(a, b):
    p_loss = exp.perceptual_loss(a, b)
    return p_loss

def train(batch, i):
    optim.zero_grad()

@readme
class NoGridExperiment(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            
            optim.zero_grad()
            recon = model.forward(item)
            
            loss = exp_loss(recon, item)
            loss.backward()
            optim.step()

            with torch.no_grad():
                recon = model.forward(item, training=False)
                self.fake = recon

            print(i, loss.item())
            self.after_training_iteration(loss)
    