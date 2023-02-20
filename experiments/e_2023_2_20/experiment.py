from typing import Callable, Type
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, unit_norm
from modules.overfitraw import OverfitRawAudio
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax
from modules.sparse import sparsify, sparsify_vectors
from modules.stft import stft
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


class ApproximateConvolution(ScriptModule):
    def __init__(self):
        super().__init__()
    
    @script_method
    def forward(self, a, b, percent_sparse):
        n_samples = a.shape[-1]

        a = F.pad(a, (0, a.shape[-1]))
        b = F.pad(b, (0, b.shape[-1]))

        n_coeffs = ((a.shape[-1] // 2) + 1)
        n_elements = int(n_coeffs * percent_sparse)

        a_spec = torch.fft.rfft(a, dim=-1, norm='ortho')[..., :n_elements]
        b_spec = torch.fft.rfft(b, dim=-1, norm='ortho')[..., :n_elements]

        x = a_spec * b_spec

        x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], n_coeffs - n_elements, dtype=x.dtype, device=x.device)], dim=-1)
        
        x = torch.fft.irfft(x, dim=-1, norm='ortho')[..., :n_samples]

        return x


# class PatchLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
    
#     def forward(self, recon: Tensor, target: Tensor):
#         batch = recon.shape[0]
#         n_samples = recon.shape[-1]
#         window_size = 512
#         step_size = window_size // 2
#         windowed = \
#             recon.unfold(-1, window_size, step_size).view(batch, 1, -1, window_size) \
#             * torch.hamming_window(window_size, device=recon.device).view(1, 1, 1, window_size)
#         windowed = torch.cat([
#             windowed, 
#             torch.zeros(batch, 1, windowed.shape[2], n_samples - window_size, device=recon.device)
#         ], dim=-1)
#         fm = fft_convolve(windowed, target.view(batch, 1, 1, n_samples))[..., :n_samples]
#         positions = torch.argmax(fm, dim=-1, keepdim=True)
#         positions = positions / n_samples
#         positioned = fft_shift(windowed, positions)[..., :n_samples]
#         positioned = torch.sum(positioned, dim=2).view(batch, 1, n_samples)

#         return F.mse_loss(positioned, target)


class DecayModel(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.n_envelopes = 512
        self.n_transfer = 512
        self.k_sparse = 16
        self.n_frames = 128

        self.reduce = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)
        self.net = DilatedStack(channels, [1, 3, 9, 27, 81, 1])

        self.envelopes = nn.Parameter(
            torch.zeros(self.n_envelopes, self.n_frames).uniform_(0, 1) \
            * torch.linspace(1, 0, self.n_frames)[None, :] ** 15)
    
        self.transfer = nn.Parameter(torch.zeros(self.n_transfer, exp.n_samples).uniform_(-1, 1))

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

        context = torch.mean(x, dim=-1)
        context = self.to_context(context)

        am = self.atom_map.forward(x)
        am = sparsify(am, self.k_sparse, return_indices=False)

        attn = self.attn(x)

        events, indices = sparsify_vectors(
            x, attn, self.k_sparse, normalize=False, dense=False)
        
        e = self.to_envelope.forward(events)
        e = hard_softmax(e, invert=True)
        e = e @ (self.envelopes ** 2)

        t = self.to_transfer.forward(events)
        t = hard_softmax(t, invert=True)
        t = t @ self.transfer

        d = self.to_decay.forward(events)
        d = torch.sigmoid(d)

        m = self.to_mix.forward(events)
        m = torch.softmax(m, dim=-1)

        env = torch.zeros_like(e)
        for i in range(self.n_frames):
            if i == 0:
                env[:, :, i] = e[:, :, i]
            else:
                env[:, :, i] = e[:, :, i] + (e[:, :, i - 1] * d[:, :, 0])
        
        envelope = F.interpolate(e, size=exp.n_samples, mode='linear')
        noise = torch.zeros_like(envelope).uniform_(-1, 1)
        noise = envelope * noise

        with_decay = F.interpolate(env, size=exp.n_samples, mode='linear')
        t = with_decay * t

        mixture = torch.stack([noise, t], dim=-1) * m[:, :, None, :]
        mixture = torch.sum(mixture, dim=-1)
        
        x = fft_convolve(mixture, am)[..., :exp.n_samples]
        x = torch.sum(x, dim=1, keepdim=True)

        x = self.verb.forward(context, x)
        return x


# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.approx = ApproximateConvolution()

#         self.n_atoms = 1024
#         self.atom_size = 2048
#         self.k_sparse = 16

#         self.approx_conv_percent = torch.zeros(1).fill_(0.1)

#         self.weights = nn.Parameter(torch.zeros(
#             1, self.n_atoms, self.atom_size).uniform_(-1, 1))
    
#     def give_atoms_unit_norm(self):
#         self.weights.data[:] = unit_norm(self.weights)
    
#     def forward(self, x):
#         n_samples = x.shape[-1]

#         w = torch.cat([
#             self.weights, 
#             torch.zeros(1, self.n_atoms, n_samples - self.atom_size, device=x.device)
#         ], dim=-1)

#         fm = self.approx.forward(x, w, self.approx_conv_percent)
#         fm = F.dropout(fm, 0.01)
#         fm = sparsify(fm, self.k_sparse, return_indices=False)


#         fm = F.pad(fm, (0, 1))
#         x = F.conv_transpose1d(
#             fm, 
#             self.weights.view(self.n_atoms, 1, self.atom_size), 
#             stride=1, 
#             padding=self.atom_size // 2)
#         return x


model = DecayModel(exp.model_dim).to(device)
optim = optimizer(model, lr=1e-3)

feat = PsychoacousticFeature().to(device)

def experiment_loss(a, b):
    a, _ = feat.forward(a)
    b, _ = feat.forward(b)
    return F.mse_loss(a, b)

def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    # loss = F.mse_loss(recon, batch)
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
    