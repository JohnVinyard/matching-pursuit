
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.decompose import fft_frequency_decompose
from modules.normalization import max_norm
from modules.stft import stft
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

def generate_sine(f0: torch.Tensor, n_samples: int):
    n_freqs = f0.shape[0]
    freq = torch.zeros(n_freqs, n_samples, device=f0.device)
    freq[:, :] = f0
    signal = torch.sin(torch.cumsum(freq * np.pi, dim=-1))
    signal = max_norm(signal)
    return signal


class BinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_powers = int(np.floor(np.log2(exp.samplerate.nyquist)))
        powers = np.array(list([2**i for i in range(n_powers)])).astype(np.float32)
        self.register_buffer('powers', torch.from_numpy(powers))
        self.f0 = nn.Parameter(torch.zeros(1, n_powers).uniform_(-1, 1))
    
    def forward(self, x):
        f0 = torch.sigmoid(self.f0)
        f0 = f0 @ self.powers[:, None]
        f0 = f0 / exp.samplerate.nyquist

        hz = radians_to_hz(f0, exp.samplerate)
        print('hz', hz.item())

        sig = generate_sine(f0, exp.n_samples)
        return sig

class ScalarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.f0 = nn.Parameter(torch.zeros(1, 1).uniform_(-1, 1))
    
    def forward(self, x):
        f0 = torch.sigmoid(self.f0)
        print('hz', radians_to_hz(f0, exp.samplerate).item())
        sig = generate_sine(f0, exp.n_samples)
        return sig

def hz_to_radians(hz, samplerate):
    return hz / samplerate.nyquist

def radians_to_hz(radians, samplerate):
    return radians * samplerate.nyquist

model = BinaryModel().to(device)
optim = optimizer(model, lr=1e-4)


def exp_loss(recon, target):
    recon = recon.view(-1, 1, exp.n_samples)
    target = target.view(-1, 1, exp.n_samples)

    # recon = stft(recon, 512, 256, pad=True)
    # target = stft(target, 512, 256, pad=True)

    # recon = torch.abs(torch.fft.rfft(recon, dim=-1, norm='ortho'))
    # target = torch.abs(torch.fft.rfft(target, dim=-1, norm='ortho'))

    # loss = F.mse_loss(recon, target)

    loss = exp.perceptual_loss(recon, target)


    # recon = fft_frequency_decompose(recon, 512)
    # target = fft_frequency_decompose(target, 512)

    # loss = 0
    # for key in recon.keys():
    #     loss = loss + F.mse_loss(recon[key], target[key])
    
    return loss

def train(batch, i):
    # batch_size = batch.shape[0]
    

    target = batch
    recon = model.forward(None)
    recon = max_norm(recon)

    loss = exp_loss(recon, target)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class F0Optim(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        i = 0
        while True:
            radians = hz_to_radians(440, exp.samplerate)
            freq = torch.zeros(1, 1, device=device)
            freq[:, :] = radians
            target = generate_sine(freq, exp.n_samples)
            self.real = target
            l, r = self.train(target, i)
            i += 1
            self.fake = r
            print(l.item())
            self.after_training_iteration(l)