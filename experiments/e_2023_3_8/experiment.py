from typing import List
import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.matchingpursuit import dictionary_learning_step, sparse_code, compare_conv
from modules.normalization import unit_norm
from modules.pointcloud import encode_events
from train.experiment_runner import BaseExperimentRunner
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
    
    def recon(self, batch, steps=16):
        instances, scatter = sparse_code(
            batch, self.d, steps, device=self.device, approx=self.slce)
        all_instances = []
        for k, v in instances.items():
            all_instances.extend(v)

        recon = scatter(batch.shape, all_instances)
        return recon, all_instances, scatter


class MultibandDictionaryLearning(object):
    def __init__(self, specs: List[BandSpec]):
        super().__init__()
        self.bands = {spec.size: spec for spec in specs}
        self.min_size = min(map(lambda spec: spec.size, specs))
    
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
        x = encode_events(events, steps)
        print(x.shape)
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
            
            self.recon()
            
            
            
