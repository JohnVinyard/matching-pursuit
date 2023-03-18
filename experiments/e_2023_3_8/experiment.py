from typing import List
import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.matchingpursuit import dictionary_learning_step, sparse_code, compare_conv
from modules.normalization import unit_norm
from modules.pointcloud import decode_events, encode_events
from train.experiment_runner import BaseExperimentRunner
from torch import nn
from util.readmedocs import readme
import zounds
from util import device, playable
import torch


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

class BandSpec(object):
    def __init__(self, size, n_atoms, atom_size, slce=None, device=None):
        super().__init__()
        self.size = size
        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.slce = slce
        self.device = device

        d = torch.zeros(
            n_atoms, atom_size, requires_grad=False).uniform_(-1, 1).to(device)
        self.d = unit_norm(d)
    
    @property
    def filename(self):
        return f'band_{self.size}.dat'
    
    def load(self):
        try:
            d = torch.load(self.filename)
            self.d = d
            print(f'loaded {self.filename}')
        except IOError:
            print(f'failed to load ${self.filename}')
    
    def store(self):
        torch.save(self.d, self.filename)
    
    def learn(self, batch, steps=16):
        d = dictionary_learning_step(
            batch, self.d, steps, device=self.device, approx=self.slce)
        self.d = unit_norm(d)
        return d
    
    def encode(self, batch, steps=16):
        instances, scatter = sparse_code(
            batch, self.d, steps, device=self.device, approx=self.slce)
        all_instances = []
        for k, v in instances.items():
            all_instances.extend(v)
        return all_instances, scatter, batch.shape
    
    def decode(self, shape, all_instances, scatter):
        return scatter(shape, all_instances)
    
    def recon(self, batch, steps=16):
        # instances, scatter = sparse_code(
        #     batch, self.d, steps, device=self.device, approx=self.slce)
        # all_instances = []
        # for k, v in instances.items():
        #     all_instances.extend(v)

        all_instances, scatter, shape = self.encode(batch, steps)

        # recon = scatter(batch.shape, all_instances)
        recon = self.decode(shape, all_instances, scatter)
        return recon, all_instances, scatter


class MultibandDictionaryLearning(object):
    def __init__(self, specs: List[BandSpec]):
        super().__init__()
        self.bands = {spec.size: spec for spec in specs}
        self.min_size = min(map(lambda spec: spec.size, specs))
        self.n_samples = exp.n_samples
    
    @property
    def band_dicts(self):
        return {size: band.d for size, band in self.bands.items()}
    
    def store(self):
        for band in self.bands.values():
            band.store()
    
    def load(self):
        for band in self.bands.values():
            band.load()
    
    def learn(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)
        for size, band in bands.items():
            self.bands[size].learn(band, steps)
    
    def encode(self, batch, steps):
        bands = fft_frequency_decompose(batch, self.min_size)
        return {size: band.encode(bands[size], steps) for size, band in self.bands.items()}
    
    def decode(self, d):
        output = {}
        for size, tup in d.items():
            all_instances, scatter, shape = tup
            recon = self.bands[size].decode(shape, all_instances, scatter)
            output[size] = recon
        recon = fft_frequency_recompose(output, self.n_samples)
        return recon
    
    def recon(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)

        recon_bands = {}
        events = {}
        scatter = {}
        for size, spec in self.bands.items():
            r, e, s = self.bands[size].recon(bands[size], steps)
            recon_bands[size] = r
            events[size] = e
            scatter[size] = s

        recon = fft_frequency_recompose(recon_bands, batch.shape[-1])
        return recon, events



class Predictor(nn.Module):
    """
    Analyze a sequence of atoms with absolute positions
    and magnitudes.

    Output a new atom, relative position and relative magnitude
    """
    def __init__(self, channels, n_atoms=512 * 7):
        super().__init__()

        self.embed = nn.Embedding(n_atoms, embedding_dim=channels)
        self.pos_amp = nn.Linear(2, channels)

        self.net = DilatedStack(channels, [1, 3, 9, 27, 81, 1], dropout=0.1, padding='only-past')


        self.to_atom = LinearOutputStack(channels, 3, out_channels=n_atoms)
        self.to_pos_amp_pred = LinearOutputStack(channels, 3, out_channels=2)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        atoms = x[:, :, 0].long()
        atoms = self.embed.forward(atoms)

        pos_amp = x[:, :, 1:3]
        pos_amp = self.pos_amp.forward(pos_amp)

        x = torch.cat([atoms, pos_amp], dim=-1)
        x = x.permute(0, 2, 1)
        x = self.net(x)
        x = x.permute(0, 2, 1)

        a = self.to_atom(x)
        pa = self.to_pos_amp_pred(x)
        return a, pa



def to_slice(n_samples, percentage):
    n_coeffs = n_samples // 2 + 1
    start = n_coeffs // 2
    total = n_coeffs - start
    size = int(percentage * total)
    end = start + size
    return slice(start, end)

n_atoms = 512
steps = 64

model = MultibandDictionaryLearning([
    BandSpec(512,   n_atoms, 128,  slce=None, device=device),
    BandSpec(1024,  n_atoms, 128,  slce=None, device=device),
    BandSpec(2048,  n_atoms, 128,  slce=None, device=device),
    BandSpec(4096,  n_atoms, 128, slce=None, device=device),
    BandSpec(8192,  n_atoms, 128, slce=None, device=device),
    BandSpec(16384, n_atoms, 128, slce=None, device=device),
    BandSpec(32768, n_atoms, 128, slce=None, device=device),
])
model.load()


def train():
    pass

@readme
class BasicMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None
    
    def recon(self, steps=steps):
        recon, events = model.recon(self.real[:1, ...], steps=steps)
        return playable(recon, exp.samplerate)
    
    def encode_for_transformer(self, batch, steps):
        encoding = model.encode(batch, steps=steps) # size -> (all_instances, scatter, shape)
        e = {k: v[0] for k, v in encoding.items()} # size -> all_instances
        events = encode_events(e, steps) # tensor (batch, 4, N)
        return events
    
    def round_trip(self, steps):
        encoding = model.encode(self.real[:1, ...], steps=steps) # size -> (all_instances, scatter, shape)
        e = {k: v[0] for k, v in encoding.items()} # size -> all_instances
        events = encode_events(e, steps) # tensor
        d = decode_events(events, model.band_dicts, steps) # size -> all_instances
        d = {k: (d[k], *encoding[k][1:]) for k in d.keys()} # size -> (all_instances, scatter, shape)
        recon = model.decode(d)
        return playable(recon, exp.samplerate)

    def store(self):
        model.store()

    def spec(self, steps=steps):
        return np.abs(zounds.spectral.stft(self.recon(steps)))
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            with torch.no_grad():
                print('====================================')
                model.learn(item, steps=steps)
            
            
            
            
