from torch import nn
from config.dotenv import Config
from config.experiment import Experiment
import zounds
import torch
import numpy as np
from torch.nn import functional as F
from modules.fft import fft_convolve
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
from modules.sparse import VectorwiseSparsity
from modules.stft import stft

from train.optim import optimizer
from upsample import PosEncodedUpsample
from util import playable
from util.readmedocs import readme
from util import device

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 16
n_atom_samples = 2048
learning_rate = 1e-4


def activation(x):
    return torch.sin(x * 30)

def softmax(x):
    # return torch.softmax(x, dim=-1)
    return F.gumbel_softmax(x, dim=-1, hard=True)

def amp_activation(x):
    return torch.relu(x)

class AtomSynth(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = PosEncodedUpsample(
            exp.model_dim, 
            exp.model_dim, 
            n_atom_samples, 
            out_channels=1, 
            layers=6, 
            activation=activation,
            concat=True)
        
        self.amp = LinearOutputStack(
            exp.model_dim, 3, out_channels=1, activation=activation)
        
        self.time = PosEncodedUpsample(
            exp.model_dim,
            exp.model_dim,
            exp.n_frames,
            out_channels=1,
            layers=6,
            activation=activation,
            concat=True)
    
    def forward(self, x):
        amp = amp_activation(self.amp(x))
        x = self.net(x)
        x = torch.tanh(x) * amp[:, None, :]

        t = self.time(x)

        t = softmax(t)

        full = torch.zeros(x.shape[0], 1, exp.n_samples, device=x.device)
        factor = exp.n_samples // exp.n_frames
        full[..., ::factor] = t

        x = F.pad(x, (0, exp.n_samples - n_atom_samples))

        x = fft_convolve(x, full)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)
        self.n_rooms = self.verb.n_rooms


        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])

        self.reduce = nn.Conv1d(exp.scale.n_bands + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(
            exp.model_dim, 2, out_channels=self.n_rooms)
        
        self.atom_synth = AtomSynth()

        self.apply(lambda p: exp.init_weights(p))


    def forward(self, x):
        batch = x.shape[0]
        x = exp.fb.forward(x, normalize=False)
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        x = self.norm(x)
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)

        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        x = self.context(x)

        x = self.norm(x)

        x, indices = self.sparse(x)

        orig_verb_params, _ = torch.max(x, dim=1)
        verb_params = orig_verb_params

        # expand to params
        mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
        rm = softmax(self.to_room(verb_params))

        x = x.view(-1, exp.model_dim)

        x = self.atom_synth(x)
        x = x.view(-1, n_events, exp.n_samples)
        samples = torch.sum(x, dim=1, keepdim=True)

        wet = self.verb.forward(samples, rm)

        samples = (mx * wet) + ((1 - mx) * samples)
        samples = samples

        samples = max_norm(samples)
        return samples
        
model = Model().to(device)
optim = optimizer(model, lr=learning_rate)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    # loss = exp.perceptual_loss(recon, batch)

    fake = stft(recon, 512, 256, pad=True)
    real = stft(batch, 512, 256, pad=True)

    loss = F.mse_loss(fake, real)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class SparseEvents(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def orig(self):
        return playable(self.real, exp.samplerate)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, recon = train(item)
            self.fake = recon
            print('R', i, l.item())

