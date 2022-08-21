import torch
import zounds
from torch.nn import functional as F
from torch import nn
from modules.psychoacoustic import PsychoacousticFeature
from modules import LinearOutputStack
import numpy as np
from modules.sparse import sparsify_vectors
from train.optim import optimizer
from util import playable

from util import device
from util.readmedocs import readme
from util.weight_init import make_initializer
from torch import jit
import math
from modules.stft import stft

n_samples = 2 ** 15

# z = int(math.log(n_samples, 2))
# band_sizes = [2**i for i in range(z, z - 6, -1)]

samplerate = zounds.SR22050()

envelope_factor = 256
envelope_frames = n_samples // envelope_factor

model_dim = 128
n_atoms = 16
n_transfer_functions = 1024
n_envelopes = 1024

n_bands = 128
kernel_size = 512

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


pif = PsychoacousticFeature([128] * 6).to(device)

init_weights = make_initializer(0.1)


def perceptual_feature(x):
    # bands = pif.compute_feature_dict(x)
    # return torch.cat(bands, dim=-2)
    return stft(x, 512, 256, pad=True)

def perceptual_loss(a, b):
    return F.mse_loss(a, b)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)

    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        return x


class SparseTransferFunctions(jit.ScriptModule):
    def __init__(self, n_transfer_functions, n_envelopes, channels):
        super().__init__()

        self.n_samples = n_samples
        self.channels = channels
        self.n_transfer_functions = n_transfer_functions
        self.n_envelopes = n_envelopes

        x = torch.zeros(n_transfer_functions, n_samples).uniform_(-1, 1) \
            * (torch.linspace(1, 0, n_samples)[None, :] ** torch.zeros(1, n_samples).uniform_(1, 20))
        # self.transfer_functions = nn.Parameter(
            # torch.zeros(n_transfer_functions, n_samples).uniform_(-1., 1.))
        self.transfer_functions = nn.Parameter(x)
        
        # t = torch.linspace(1, 0, n_samples // envelope_factor)
        # p = torch.linspace(1, 20, n_envelopes)
        # x = p[None, :] ** t[:, None]

        x = torch.zeros(n_envelopes, envelope_frames).uniform_(0, 1) \
            * (torch.linspace(1, 0, envelope_frames)[None, :] ** torch.zeros(1, envelope_frames).uniform_(1, 20))
        # self.envelopes = nn.Parameter(
            # torch.zeros(n_envelopes, n_samples // envelope_factor).uniform_(0, 1))
        self.envelopes = nn.Parameter(x)

        self.to_transfer_function = LinearOutputStack(
            channels, 3, out_channels=n_transfer_functions, activation=None)
        self.to_envelopes = LinearOutputStack(
            channels, 3, out_channels=n_envelopes, activation=None)
        self.to_amp = LinearOutputStack(channels, 3, out_channels=1, activation=None)


    def normalized(self):
        tf = self.transfer_functions
        mx, _ = torch.max(tf, dim=-1, keepdim=True)
        tf = tf / (mx + 1e-8)

        e = self.envelopes
        mx, _ = torch.max(e, dim=-1, keepdim=True)
        e = e / (mx + 1e-8)

        return tf, e

    @jit.script_method
    def forward(self, x):
        batch, channels = x.shape

        transfer_functions, envelope = self.normalized()

        tf = torch.softmax(self.to_transfer_function(x), dim=-1)
        tm, ti = torch.max(tf, dim=1, keepdim=True)
        tf_values = torch.gather(tf, dim=1, index=ti)
        tf_values = tf_values + (1 - tf_values)
        funcs = transfer_functions[ti]
        tf = (funcs * tf_values[..., None]).view(batch, 1, self.n_samples)

        e = torch.softmax(self.to_envelopes(x), dim=-1)
        em, ei = torch.max(e, dim=1, keepdim=True)
        env_values = torch.gather(e, dim=1, index=ei)
        env_values = env_values + (1 - env_values)
        envs = envelope[ei]
        env = (envs * env_values[..., None]).view(batch, 1, envs.shape[-1])
        env = torch.abs(env)
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        env = env * torch.zeros_like(env).uniform_(-1., 1.)
        env = env.view(batch, 1, self.n_samples)

        a = torch.abs(self.to_amp(x))

        tf = torch.fft.rfft(tf, dim=-1, norm='ortho')
        env = torch.fft.rfft(env, dim=-1, norm='ortho')
        sig = tf * env
        sig = torch.fft.irfft(sig, dim=-1, norm='ortho')
        sig = sig * a[..., None]

        return sig

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 1)
        )

        self.fb = fb
        self.attn = nn.Conv1d(model_dim, 1, 1, 1, 0)

        self.transfer_functions = SparseTransferFunctions(
            n_transfer_functions, n_envelopes, model_dim)
        
        self.apply(init_weights)

    def forward(self, x):
        batch, _, time = x.shape

        x = self.fb.forward(x, normalize=False)
        x = self.net(x)

        attn = torch.softmax(self.attn(x), dim=-1)

        x, indices = sparsify_vectors(x, attn, n_atoms)

        x = x.view(-1, model_dim)

        atoms = self.transfer_functions(x).view(batch, -1, n_samples)

        output = torch.zeros(batch, 1, n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(n_atoms):
                atom = atoms[b, i]
                start = indices[b, i]
                end = start + n_samples
                output[b, :, start: end] += atom
        
        return output[..., :n_samples]
        

model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class SparseTransferFunctionExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None
        self.real = None

    def orig(self):
        return playable(self.real, samplerate)
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))
    
    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss, self.fake = train(item)
            print(loss.item())
