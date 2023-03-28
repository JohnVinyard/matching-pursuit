
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.ddsp import overlap_add
from modules.phase import AudioCodec, MelScale
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from util import device, playable
from util.readmedocs import readme
from librosa.decompose import hpss
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


scale = MelScale()
codec = AudioCodec(scale)

def train(batch, i):
    pass

def istft(spec):
    windowed = torch.fft.irfft(spec, dim=-1, norm='ortho')
    signal = overlap_add(windowed[:, None, :, :], apply_window=False)
    return signal

def to_mag_phase(x):
    mag = torch.abs(x)
    phase = torch.angle(x)
    return mag, phase

def to_complex(mag, phase):
    return mag * torch.exp(1j * phase)

@readme
class ComplexValuedMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.recon = None
        self.harm = None
        self.perc = None
    
    def orig(self):
        return playable(self.recon, exp.samplerate)

    def h(self):
        x = istft(self.harm)
        return playable(x, exp.samplerate)

    def p(self):
        perc = self.perc

        # if destroy:
        #     perc.imag = torch.zeros(*perc.imag.shape).normal_(perc.real.mean(), perc.real.std())
        m, p = to_mag_phase(perc)
        p = torch.zeros(*p.shape, device=device).uniform_(-np.pi, np.pi)
        p = torch.cumsum(p, dim=1)
        perc = to_complex(m, p)
            
        x = istft(perc)
        return playable(x, exp.samplerate)
    
    def combine(self):
        return playable(
            self.h() + self.p(), 
            exp.samplerate)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            spec = stft(item.view(-1, exp.n_samples), 1024, 512, pad=True, return_complex=True)
            self.recon = istft(spec)

            spec = spec.data.cpu().numpy()

            harms = []
            percs = []

            for b in spec:
                harm, perc = hpss(b.T)
                harms.append(harm.T[None, ...])
                percs.append(perc.T[None, ...])
            
            harm = np.concatenate(harms, axis=0)
            perc = np.concatenate(percs, axis=0)

            harm = torch.from_numpy(harm).to(device)
            perc = torch.from_numpy(perc).to(device)

            self.harm = harm
            self.perc = perc

            

        
    