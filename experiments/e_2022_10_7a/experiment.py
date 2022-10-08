
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
import zounds
from torch import nn
import torch
from torch.nn import functional as F
from modules.fft import fft_shift
from modules.stft import stft

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    model_dim=128,
    weight_init=0.1
)


class Branch(nn.Module):
    def __init__(self, factor=2, range=1):
        super().__init__()
        self.to_time = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.transform = LinearOutputStack(
            exp.model_dim, 2, out_channels=exp.model_dim * factor)
        self.factor = factor
        self.range = range
        self.norm = ExampleNorm()
    
    def forward(self, x, base_time=None):
        batch = x.shape[0]
        
        x = x.view(batch, -1, exp.model_dim)

        x = self.norm(x)

        x = self.transform(x)
        x = x.view(batch, -1, exp.model_dim)
        t = torch.sigmoid(self.to_time(x)) * self.factor

        if base_time is None:
            time = t
        else:
            base_time = base_time.view(batch, -1, 1).repeat(1, t.shape[1] // base_time.shape[1], 1)
            time = t + base_time
        
        return time, x

class Leaf(nn.Module):
    def __init__(self, n_atoms=2048, atom_size=512, factor=1):
        super().__init__()
        self.to_time = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_selection = LinearOutputStack(exp.model_dim, 2, out_channels=n_atoms)
        self.to_amp = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.factor = factor
        self.n_atoms = n_atoms
        self.atom_size = atom_size

        
        self.atoms = nn.Parameter(torch.zeros(n_atoms, atom_size).uniform_(-1, 1))
    
    @property
    def normed_atoms(self):
        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-8)
        return atoms
    
    def forward(self, x, times):
        time = (torch.sigmoid(self.to_time(x)) * self.factor) + times

        amp = self.to_amp(x) ** 2
        selections = F.gumbel_softmax(self.to_selection(x), tau=1, dim=-1, hard=True)

        atoms = (selections @ self.normed_atoms)
        atoms = atoms * amp

        atoms = F.pad(atoms, (0, exp.n_samples - self.atom_size))

        final = fft_shift(atoms, time)
        return torch.sum(final, dim=1, keepdim=True)

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = DilatedStack(128, [1, 3, 9, 27, 1])

        self.levels = nn.Sequential(
            Branch(factor=4, range=1),
            Branch(factor=4, range=0.5),
            Branch(factor=4, range=0.25),
            Branch(factor=4, range=0.125),
        )

        self.final = Leaf(n_atoms=2048, atom_size=512, factor=0.05)

        self.apply(exp.init_weights)
    
    def forward(self, x):
        batch = x.shape[0]

        x = exp.pooled_filter_bank(x)
        x = self.encoder(x)
        x, _ = torch.max(x, dim=-1)

        times = torch.zeros(batch, 1, device=x.device)
        vectors = x

        for level in self.levels:
            times, vectors = level.forward(vectors, times)
        
        x = self.final.forward(vectors, times)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)

    loss.backward()
    optim.step()
    print(loss.item())
    return recon

@readme
class GraphRepresentationExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.recon = None
        self.real = None
    
    def listen(self):
        return playable(self.recon, exp.samplerate)
    
    def orig(self):
        return playable(self.real, exp.samplerate)
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            self.recon = train(item)