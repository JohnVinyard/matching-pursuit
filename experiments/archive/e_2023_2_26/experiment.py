import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.dilated import DilatedStack
from modules.normalization import ExampleNorm, unit_norm
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify
from modules.phase import morlet_filter_bank
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import Tensor, nn
from util import device
from torch.nn import functional as F
import torch

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_bands = 6

scattering_window_size = 64
n_log_samples = int(np.log2(exp.n_samples))
band_sizes = [2**i for i in range(n_log_samples, n_log_samples - n_bands, -1)][::-1]

# atom sizes are constant in time btut vary in size
# TODO: try the opposite, constant in size but variable in time
atom_sizes = {k: k // 4 for k in band_sizes}

k_sparse = 64
n_atoms = 1024

n_filters_per_band = exp.model_dim
kernel_size = 128

scale = zounds.LinearScale(
    zounds.FrequencyBand(1, exp.samplerate.nyquist), n_filters_per_band)

filter_bank = torch.from_numpy(morlet_filter_bank(
    exp.samplerate, 
    kernel_size, 
    scale=scale, 
    scaling_factor=0.1,
    normalize=True).real).float().view(1, n_filters_per_band, kernel_size).to(device)


class SpectralBand(nn.Module):
    def __init__(self, band_size):
        super().__init__()
        self.band_size = band_size
    
    @property
    def dict_key(self):
        return str(self.band_size)
    
    @property
    def size(self):
        return self.band_size
    
    def forward(self, x):
        x = x.view(-1, 1, self.band_size)
        x = F.conv1d(x, filter_bank.view(n_filters_per_band, 1, kernel_size), padding=kernel_size // 2)[..., :self.band_size]
        x = torch.relu(x)
        x = 20 * torch.log(x + 1e-8)
        return x

class Scattering(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor):
        x = F.pad(x, (0, scattering_window_size))
        x = x.unfold(-1, scattering_window_size, scattering_window_size // 2)
        x = torch.fft.rfft(x, dim=-1, norm='ortho')
        x = torch.abs(x)
        return x

class AnalysisBand(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.filters = SpectralBand(self.size)
        self.scatter = Scattering()
    
    def forward(self, x):
        x = self.filters.forward(x)
        x = self.scatter.forward(x)
        return x

class SynthesisBand(nn.Module):
    def __init__(self, band_size, atom_size):
        super().__init__()
        self.band_size = band_size
        self.atom_size = atom_size
        self.atoms = nn.Parameter(torch.zeros(n_atoms, 1, atom_size).uniform_(-1, 1))
        self.gain = nn.Parameter(torch.zeros(1).fill_(1))
    
    def forward(self, x):
        full_size = x.shape[-1]

        if full_size != self.band_size:
            window = full_size // self.band_size
            pool = nn.MaxPool1d(window, window, padding=0)
            down = pool.forward(x)
        else:
            down = x
        
        down = F.pad(down, (0, self.atoms.shape[-1] + 1))
        output = F.conv_transpose1d(down, self.atoms, stride=1)
        output = output[..., :self.band_size]
        output = output * self.gain
        return output

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bands = nn.ModuleDict({str(k): AnalysisBand(k) for k in band_sizes})
    
    def loss(self, a, b):
        a = self.forward(a)
        b = self.forward(b)
        losses = {k: F.mse_loss(a[k], b[k]) for k in band_sizes}
        
        loss = 0
        for k, v in losses.items():
            loss = loss + v
        
        return loss
    
    def forward(self, x):
        x = fft_frequency_decompose(x, band_sizes[0])
        # analysis = {k: self.bands[str(k)].forward(x[k]) for k in x}
        analysis = x
        return analysis

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        return x


class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # self.encoder = DilatedStack(
        #     channels, 
        #     [1, 3, 9, 27, 81, 1], 
        #     dropout=0.1, 
        #     padding='only-future', 
        #     soft_sparsity=False, 
        #     internally_sparse=False, 
        #     sparsity_amt=1)

        self.encoder = nn.Sequential(*[
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 3),
            DilatedBlock(channels, 9),
            DilatedBlock(channels, 27),
            DilatedBlock(channels, 81),
            DilatedBlock(channels, 243),
            DilatedBlock(channels, 1),
        ])
    
        self.synth_bands = nn.ModuleDict({str(k): SynthesisBand(k, atom_sizes[k]) for k in band_sizes})
    
        self.up = nn.Conv1d(self.channels, n_atoms, 1, 1, 0)
        self.mask = nn.Conv1d(self.channels, n_atoms, 1, 1, 0)
        self.verb = ReverbGenerator(channels, 3, exp.samplerate, exp.n_samples)
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x, debug=False):
        batch = x.shape[0]
        x = F.conv1d(x, filter_bank.view(n_filters_per_band, 1, kernel_size), padding=kernel_size // 2)
        x = x[..., :exp.n_samples]
        encoded = self.encoder(x)

        context = torch.mean(encoded, dim=-1)

        # saliency map
        encoded = F.dropout(encoded, p=0.05)
        mask = self.mask(encoded)
        mask = torch.tanh(mask)
        encoded = self.up(encoded)

        encoded = encoded * mask


        # TODO: Generate queries, keys and values (normalized) in order
        # to choose the most salient moments, regardless of raw 
        # amplitude
        encoded = sparsify(
            encoded, k_sparse, return_indices=False, soft=False, sharpen=False)

        atoms = (encoded > 0).sum().item()
        if debug:
            print('NON-ZERO', atoms / batch)
        assert (atoms / batch) <= k_sparse

        
        bands = {k: self.synth_bands[str(k)].forward(encoded) for k in band_sizes}
        
        final = fft_frequency_recompose(bands, exp.n_samples)

        final = self.verb.forward(context, final)
        return final, encoded

model = Model(exp.model_dim).to(device)

# try:
#     model.load_state_dict(torch.load('model.dat'))
#     print('loaded weights from model')
# except IOError:
#     print('failed to load weights')

optim = optimizer(model, lr=1e-3)

loss_model = Loss().to(device)

def train(batch, i):
    optim.zero_grad()
    recon, encoded = model.forward(batch, debug=False)
    loss = loss_model.loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon, encoded


@readme
class SparseMultibandSynthesis(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.model = model
        self.encoded = None
    
    @property
    def enc(self):
        return self.encoded[0].data.cpu().numpy()
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            l, r, e = train(item, i)
            self.fake = r
            self.real = item
            self.encoded = e
            print(i, l.item())
    