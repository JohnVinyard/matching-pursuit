import zounds
import torch
from kymatio.torch import Scattering1D
from modules.ddsp import NoiseModel
from torch import nn
import torch
from modules.overfitraw import OverfitRawAudio
from modules.pif import AuditoryImage
from modules.stft import stft
from train.optim import optimizer
from torch.nn import functional as F
from data import AudioIterator
import argparse
from modules.upsample import ConvUpsample
from util.playable import playable, viewable
from util.weight_init import make_initializer
import numpy as np


"""
In this experiment, we'll pick a set of N audio files and produce
two variations of each, one with phase perturbed, and another produced
only with band-limited noise.

Ideally, a loss/distance function will place the phase-perturbed audio closer
to the original than the band-limited noise version
"""

init = make_initializer(0.1)

class STFTLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, recon, target):
        r = stft(recon, 512, 256, pad=True)
        t = stft(target, 512, 256, pad=True)
        return F.mse_loss(r, t)


class AIMLoss(nn.Module):
    def __init__(self, samplerate: zounds.SampleRate, n_bands: int, kernel_size=512, residual=False, twod=False):
        super().__init__()
        band = zounds.FrequencyBand(20, samplerate.nyquist)
        scale = zounds.MelScale(band, n_bands)
        self.fb = zounds.learn.FilterBank(
            samplerate, kernel_size, scale, 0.25, normalize_filters=True)
        self.aim = AuditoryImage(
            512, 128, do_windowing=False, check_cola=False, residual=residual, twod=twod)
    
    def forward(self, recon, target):
        r = self.aim(self.fb.forward(recon, normalize=True))
        t = self.aim(self.fb.forward(target, normalize=True))
        return F.mse_loss(r, t)



class ScatteringLoss(nn.Module):
    def __init__(self, n_samples):
        super().__init__()

        self.T = n_samples
        self.J = 6
        self.Q = 16
        self.scattering = Scattering1D(self.J, self.T, self.Q)
    
    def _features(self, x):
        x = self.scattering(x.view(-1))
        meta = self.scattering.meta()
        order1 = np.where(meta['order'] == 1)
        order2 = np.where(meta['order'] == 2)
        return torch.cat([
            torch.log(1e-8 + x[order1].view(-1)), 
            torch.log(1e-8 + x[order2].view(-1))
        ])
    
    def forward(self, recon, target):
        r = self._features(recon)
        t = self._features(target)
        return F.mse_loss(r, t)

class BandLimitedNoiseModel(nn.Module):
    def __init__(self, dim=128, n_samples=2*15, window_size=512):
        super().__init__()
        self.n_frames = 64

        self.latent = nn.Parameter(torch.zeros(1, dim).normal_(0, 1))
        self.up = ConvUpsample(dim, dim, 4, self.n_frames, mode='learned', out_channels=dim)
        self.model = NoiseModel(
            dim, self.n_frames, self.n_frames * 2, n_samples, dim, squared=True, mask_after=1)
        self.apply(init)
    
    def forward(self, x):
        x = self.up(self.latent)
        return self.model.forward(x)

def scaled(x):
    return x / np.abs(x).max()


def stft_mag_phase(signal: zounds.AudioSamples):
    spec = zounds.spectral.stft(
        signal, 
        zounds.HalfLapped(), 
        zounds.OggVorbisWindowingFunc())
    
    mag = np.abs(spec)
    phase = np.angle(spec)
    return mag, phase

def istft_mag_phase(mag: np.ndarray, phase: np.ndarray):
    real = mag
    imag = phase

    spec = real * np.exp(1j * imag)
    synth = zounds.FFTSynthesizer()
    audio = synth.synthesize(spec)
    return audio

def randomize_phase(signal):
    mag, phase = stft_mag_phase(signal)
    perturbation = np.random.uniform(-np.pi, np.pi, phase.shape) * 0.5
    phase[:, 1:] += perturbation[:, 1:]
    audio = istft_mag_phase(mag, phase)
    return audio

def shift_bin_phases(signal, shape=(1, 2)):
    mag, phase = stft_mag_phase(signal)
    perturbation = np.random.uniform(-np.pi, np.pi, shape) * 0.5
    phase[:, 1:] += perturbation[:, 1:]
    audio = istft_mag_phase(mag, phase)
    return audio


def produce_rendering(audio, n_iterations=1000, loss_func=STFTLoss()):
    model = OverfitRawAudio((1, audio.shape[-1]), std=0.01)
    optim = optimizer(model, lr=1e-3)

    for _ in range(n_iterations):
        optim.zero_grad()
        recon = model.forward(None)
        loss = loss_func(recon, audio)
        loss.backward()
        print(loss.item())
        optim.step()
    
    return recon

def produce_band_limited_noise_rendering(audio, n_iterations=1000, loss_func=STFTLoss()):
    model = BandLimitedNoiseModel(dim=128, n_samples=audio.shape[-1], window_size=512)
    optim = optimizer(model, lr=1e-3)

    for _ in range(n_iterations):
        optim.zero_grad()
        recon = model.forward(None)
        loss = loss_func(recon, audio)
        loss.backward()
        print(loss.item())
        optim.step()
    
    return recon



class LossEvaluator(object):
    def __init__(self, name, loss_func, samplerate, n_samples, n_iterations=1000):
        super().__init__()
        self.name = name
        self.loss_func = loss_func
        self.samplerate = samplerate
        self.n_iterations = n_iterations
        self.n_samples = n_samples
    
    def evaluate(self, audio):
        x = audio.data.cpu().numpy().reshape(-1)
        x = zounds.AudioSamples(x, self.samplerate)

        randomized = randomize_phase(x)[:self.n_samples]
        shifted = shift_bin_phases(x, shape=(1, 2))[:self.n_samples]

        randomized = torch.from_numpy(randomized).float()
        shifted = torch.from_numpy(shifted).float()

        r_loss = self.loss_func(randomized, audio)
        s_loss = self.loss_func(shifted, audio)

        rendered = produce_rendering(audio, self.n_iterations, self.loss_func)
        rendered = playable(rendered, self.samplerate)

        return r_loss, s_loss, rendered

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--n_samples', type=int)
    parser.add_argument('--noise-iterations', type=int, default=1000)

    args = parser.parse_args()

    n_samples = 2 ** args.n_samples
    samplerate = zounds.SR22050()
    n_iterations = 500

    audio_iter = AudioIterator(
        args.batch_size, n_samples, samplerate, normalize=True)
    batch = next(audio_iter.__iter__()).view(-1, 1, n_samples)

    experiments = [
        # LossEvaluator('stft', STFTLoss(), samplerate, n_samples, n_iterations=n_iterations),
        # LossEvaluator('scatter', ScatteringLoss(n_samples), samplerate, n_samples, n_iterations),
        LossEvaluator(
            'aim_residual', 
            AIMLoss(samplerate, 128, residual=True), samplerate, n_samples, n_iterations=n_iterations),

        LossEvaluator(
            'aim_2D', 
            AIMLoss(samplerate, 128, residual=False, twod=True), samplerate, n_samples, n_iterations=n_iterations),


        LossEvaluator(
            'aim', 
            AIMLoss(samplerate, 128, residual=False), samplerate, n_samples, n_iterations=n_iterations),
        
        
        
    ]


    all_results = []

    current_render = None

    for i, item in enumerate(batch):
        item = item.view(1, 1, n_samples)
        results = {}
        for exp in experiments:
            r_loss, s_loss, rendered = exp.evaluate(item)
            current_render = rendered
            print(f'{exp.name} - {r_loss} {s_loss}')
            results[exp.name] = (r_loss, s_loss, rendered)
        all_results.append(results)
        

        



    input('Waiting...')
