import torch
import zounds
from torch.nn import functional as F
from torch import nn
from modules.ddsp import overlap_add
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules import LinearOutputStack
import numpy as np
from modules.sparse import sparsify_vectors
from modules.stft import stft
from train.optim import optimizer
from util import playable
from scipy.signal import square, sawtooth

from util import device
from util.readmedocs import readme
from util.weight_init import make_initializer
from torch import jit

n_samples = 2 ** 15

samplerate = zounds.SR22050()

normalize_softmax = True

step_size = 256
n_frames = n_samples // step_size


envelope_factor = 256
envelope_frames = n_samples // envelope_factor

model_dim = 128
n_atoms = 16
n_transfer_functions = 384
n_envelopes = 64

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


pif = PsychoacousticFeature().to(device)

init_weights = make_initializer(0.1)


def perceptual_feature(x):
    bands = pif.compute_feature_dict(x)
    return torch.cat(bands, dim=-2)

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
        self.normalize_softmax = normalize_softmax


        # freqs = torch.from_numpy(np.geomspace(20, 3520, n_transfer_functions)).float()
        # t = torch.linspace(0, np.pi, n_samples)
        # x = torch.sin(t[None, :] * freqs[:, None]) * (torch.linspace(1, 0, n_samples) ** 10)[None, :]
        # self.transfer_functions = nn.Parameter(x)

        tfs = []
        start = 20 / samplerate.nyquist
        stop = 4000 / samplerate.nyquist
        f0s = torch.linspace(start, stop, n_transfer_functions // 3)

        for i in range(n_transfer_functions // 3):
            rps = f0s[i] * np.pi
            radians = np.linspace(0, rps * n_samples, n_samples)
            sin = np.sin(radians)[None, ...]
            sq = square(radians)[None, ...]
            st = sawtooth(radians)[None, ...]
            tfs.extend([sin, sq, st])

        tfs = tfs * (np.linspace(1, 0, n_samples) ** 10)[None, ...]
        tfs = np.concatenate(tfs, axis=0).astype(np.float32)
        tfs = torch.from_numpy(tfs)
        # self.transfer_functions = nn.Parameter(torch.from_numpy(tfs))

        # tfs = torch.zeros(n_transfer_functions, n_samples).uniform_(-1, 1) * (torch.linspace(1, 0, n_samples) ** 20)[None, ...]
        tfs = stft(tfs.view(n_transfer_functions, 1, n_samples), 512, 256, pad=True)
        self.transfer_functions = nn.Parameter(tfs)
        
        
        x = torch.zeros(n_envelopes, envelope_frames)\
            .uniform_(0, 1) \
            * (torch.linspace(1, 0, envelope_frames)[None, :] ** np.random.randint(10, 50))
        self.envelopes = nn.Parameter(x)

        self.to_transfer_function = LinearOutputStack(
            channels, 3, out_channels=n_transfer_functions, activation=None)
        self.to_envelopes = LinearOutputStack(
            channels, 3, out_channels=n_envelopes, activation=None)
        self.to_amp = LinearOutputStack(channels, 3, out_channels=1, activation=None)


    def normalized(self):
        tf = self.transfer_functions
        tf = torch.fft.irfft(tf, dim=-1, norm='ortho')
        tf = overlap_add(tf)[..., :n_samples].view(-1, n_samples)

        mx, _ = torch.max(tf, dim=-1, keepdim=True)
        tf = tf / (mx + 1e-8)

        e = self.envelopes
        mx, _ = torch.max(e, dim=-1, keepdim=True)
        e = e / (mx + 1e-8)

        return tf, e

    # @jit.script_method
    def forward(self, x):
        batch, channels = x.shape

        x = F.dropout(x, 0.1)

        transfer_functions, envelope = self.normalized()

        tf = self.to_transfer_function(x)
        tf = torch.softmax(tf, dim=-1)
        tm, ti = torch.max(tf, dim=1, keepdim=True)
        tf_values = torch.gather(tf, dim=1, index=ti)
        if self.normalize_softmax:
            tf_values = tf_values + (1 - tf_values)
        funcs = transfer_functions[ti]
        tf = (funcs * tf_values[..., None]).view(batch, 1, self.n_samples)

        e = self.to_envelopes(x)
        e = torch.softmax(e, dim=-1)
        em, ei = torch.max(e, dim=1, keepdim=True)
        env_values = torch.gather(e, dim=1, index=ei)
        if self.normalize_softmax:
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


class Norm(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        norms = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norms + 1e-12)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()


        encoder = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True, activation=lambda x: F.gelu(x))
        self.context = nn.TransformerEncoder(encoder, 6, norm=Norm())
        self.to_env = nn.Conv1d(model_dim, 1, 1, 1, 0)
        self.embed = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)

        self.fb = fb
        self.attn = nn.Conv1d(model_dim, 1, 1, 1, 0)

        self.transfer_functions = SparseTransferFunctions(
            n_transfer_functions, n_envelopes, model_dim)
        
        self.apply(init_weights)

    def forward(self, x):
        batch, _, time = x.shape

        x = torch.abs(self.fb.convolve(x))
        x = self.fb.temporal_pooling(x, 512, 256)[..., :n_frames]

        pos = pos_encoded(batch, n_frames, n_freqs=16, device=x.device).permute(0, 2, 1)


        n = torch.cat([pos, x], dim=1)

        n = self.embed(n)
        n = n.permute(0, 2, 1)
        x = self.context(n)
        x = x.permute(0, 2, 1)

        vectors = x

        attn = torch.softmax(self.attn(x), dim=-1)

        x, indices = sparsify_vectors(x, attn, n_atoms, normalize=normalize_softmax)

        x = x.view(-1, model_dim)
        norms = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norms + 1e-12)

        atoms = self.transfer_functions(x).view(batch, -1, n_samples)

        output = torch.zeros(batch, 1, n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(n_atoms):
                atom = atoms[b, i]
                start = indices[b, i] * step_size
                end = start + n_samples
                output[b, :, start: end] += atom
        
        output = output[..., :n_samples]

        mx, _ = torch.max(torch.abs(output), dim=-1, keepdim=True)
        output = output / (mx + 1e-12)
        return output, indices, vectors
        

model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon, indices, vectors = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon, indices, vectors

@readme
class SparseTransferFunctionExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None
        self.real = None
        self.indices = None
        self.vectors = None
        self.model = model
    
    def events(self):
        indices = self.indices.data.cpu().numpy()[0]
        x = np.zeros((n_frames, 10))
        x[indices, :] = 1
        return x
    
    def envelopes(self):
        return self.model.transfer_functions.envelopes.data.cpu().numpy().squeeze()
    
    def transfer_functions(self):
        x = self.model.transfer_functions.transfer_functions.data.cpu().numpy().squeeze()
        return x

    def orig(self):
        return playable(self.real, samplerate)
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))
    
    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def vec(self):
        return self.vectors.data.cpu().numpy()[0].T

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss, self.fake, self.indices, self.vectors = train(item)
            print(loss.item())
