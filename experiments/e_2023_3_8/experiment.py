from typing import List
import numpy as np
from config.experiment import Experiment
from modules import stft
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.matchingpursuit import dictionary_learning_step, sparse_code, compare_conv
from modules.normalization import unit_norm
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
        return recon


class MultibandDictionaryLearning(object):
    def __init__(self, specs: List[BandSpec]):
        super().__init__()
        self.bands = {spec.size: spec for spec in specs}
        self.min_size = min(map(lambda spec: spec.size, specs))
    
    def learn(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)
        for size, band in bands.items():
            self.bands[size].learn(band, steps)
    
    def recon(self, batch, steps=16):
        bands = fft_frequency_decompose(batch, self.min_size)
        recon_bands = {
            size: self.bands[size].recon(bands[size], steps) 
            for size, spec in self.bands.items()
        }
        recon = fft_frequency_recompose(recon_bands, batch.shape[-1])
        return recon


def to_slice(n_samples, percentage):
    n_coeffs = n_samples // 2 + 1
    start = n_coeffs // 2
    total = n_coeffs - start
    size = int(percentage * total)
    end = start + size
    return slice(start, end)

model = MultibandDictionaryLearning([
    BandSpec(512,   512, 128,  slce=None, device=device),
    BandSpec(1024,  512, 128,  slce=None, device=device),
    BandSpec(2048,  512, 128,  slce=None, device=device),
    BandSpec(4096,  512, 128, slce=None, device=device),
    BandSpec(8192,  512, 128, slce=None, device=device),
    BandSpec(16384, 512, 128, slce=None, device=device),
    BandSpec(32768, 512, 128, slce=None, device=device),
])

steps = 64

def train():
    pass

@readme
class BasicMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.encoded = None
    
    def recon(self, steps=steps):
        recon = model.recon(self.real[:1, ...], steps=steps)
        return playable(recon, exp.samplerate)

    def spec(self, steps=steps):
        return np.abs(zounds.spectral.stft(self.recon(steps)))
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            with torch.no_grad():
                print('====================================')
                model.learn(item, steps=steps)
            
            
            
