from typing import Callable, Type
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm, unit_norm
from modules.overfitraw import OverfitRawAudio
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, soft_clamp, sparse_softmax
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import morlet_filter_bank, stft
from perceptual.feature import NormalizedSpectrogram
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import Tensor, nn
from util import device, playable
from torch.nn import functional as F
import torch
import numpy as np
from torch.jit._script import ScriptModule, script_method
import time
from torch.nn.init import orthogonal_

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class DecayModel(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.n_envelopes = 512
        self.n_transfer = 512
        self.k_sparse = 16

        self.env_frames = 8
        self.n_frames = 128

        self.min_decay = 0.5

        self.reduce = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)
        self.net = DilatedStack(channels, [1, 3, 9, 27, 81, 1])

        self.envelopes = nn.Parameter(
            torch.zeros(self.n_envelopes, self.env_frames).uniform_(0, 1) \
            * torch.linspace(1, 0, self.env_frames)[None, :] ** 10)
    

        self.norm = ExampleNorm()

        bank = morlet_filter_bank(
            exp.samplerate, exp.n_samples, zounds.MelScale(zounds.FrequencyBand(20, 2000), self.n_transfer), 0.1, normalize=True).real
    
        # self.transfer = nn.Parameter(torch.zeros(self.n_transfer, exp.n_samples).uniform_(-1, 1))
        self.transfer = nn.Parameter(max_norm(torch.from_numpy(bank).float()))

        self.to_envelope = LinearOutputStack(channels, 3, out_channels=self.n_envelopes)
        self.to_transfer = LinearOutputStack(channels, 3, out_channels=self.n_transfer)
        self.to_decay = LinearOutputStack(channels, 3, out_channels=1)
        self.to_mix = LinearOutputStack(channels, 3, out_channels=2)

        self.to_context = LinearOutputStack(channels, 3, out_channels=channels)

        self.atom_map = nn.Conv1d(channels, self.k_sparse, 1, 1, 0)

        self.verb = ReverbGenerator(channels, 3, exp.samplerate, exp.n_samples)

        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x):
        batch, _, n_samples = x.shape
        x = exp.fb.forward(x, normalize=False)
        pos = pos_encoded(batch, n_samples, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([pos, x], dim=1)
        x = self.reduce(x)
        x = self.net(x)
        x = self.norm(x)
        x = F.dropout(x, p=0.1)

        context = torch.mean(x, dim=-1)
        context = self.to_context(context)

        am = self.atom_map.forward(x)
        am = F.dropout(am, p=0.1)
        am = sparsify(am, self.k_sparse, return_indices=False)

        attn = self.attn(x)

        events, indices = sparsify_vectors(
            x, attn, self.k_sparse, normalize=True, dense=False)
        
        e = self.to_envelope.forward(events)
        e = hard_softmax(e)
        e = e @ torch.abs(self.envelopes)
        e = F.pad(e, (0, self.n_frames - self.env_frames))

        t = self.to_transfer.forward(events)
        t = hard_softmax(t)
        t = t @ self.transfer

        d = self.to_decay.forward(events)
        d = self.min_decay + ((1 - self.min_decay) * torch.sigmoid(d) * 0.99)

        m = self.to_mix.forward(events)
        m = torch.softmax(m, dim=-1)

        env = torch.zeros_like(e)
        for i in range(self.n_frames):
            if i == 0:
                env[:, :, i] = e[:, :, i]
            else:
                env[:, :, i] = e[:, :, i] + (env[:, :, i - 1].clone() * d[:, :, 0])
        
        envelope = F.interpolate(e, size=exp.n_samples, mode='linear')
        with_decay = F.interpolate(env, size=exp.n_samples, mode='linear')

        noise = torch.zeros_like(envelope).uniform_(-1, 1)

        
        noise = envelope * noise
        t = with_decay * t

        mixture = torch.stack([noise, t], dim=-1) #* m[:, :, None, :]
        mixture = torch.sum(mixture, dim=-1)
        
        x = fft_convolve(mixture, am)[..., :exp.n_samples]
        x = torch.sum(x, dim=1, keepdim=True)

        x = self.verb.forward(context, x)
        return x


model = DecayModel(exp.model_dim).to(device)
optim = optimizer(model, lr=1e-3)

feat = PsychoacousticFeature().to(device)

def experiment_loss(a, b):
    # a, _ = feat.forward(a)
    # b, _ = feat.forward(b)
    return F.mse_loss(a, b)
    # return exp.perceptual_loss(a, b)

def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = experiment_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class ApproximateConvolution(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None
    