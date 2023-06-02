
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_shift
from modules.linear import LinearOutputStack
from modules.reverb import ReverbGenerator
from modules.sparse import sparsify_vectors
from scalar_scheduling import pos_encoded
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


n_events = 8
min_decay = 0.8


class Resonance(nn.Module):
    def __init__(self, n_samples, n_frames, filter_size, n_filters):
        super().__init__()

        self.n_samples = n_samples
        self.n_frames = n_frames
        self.filter_size = filter_size
        self.n_filters = n_filters

        self.filters = nn.Parameter(torch.zeros(n_filters, filter_size).uniform_(-1, 1))

        self.to_mixture = LinearOutputStack(exp.model_dim, 3, out_channels=n_filters)
    
    def forward(self, signal, x):

        batch, n_events, _, samples = signal.shape
        signal = signal.view(batch * n_events, 1, samples)
        x = x.view(batch * n_events, exp.model_dim)

        filtered = F.conv1d(signal, self.filters.view(self.n_filters, 1, self.filter_size), padding=self.filter_size // 2)
        filtered = filtered[..., :exp.n_samples]

        mx = self.to_mixture(x) ** 2

        x = filtered * mx[..., None] 

        x = x.view(batch, n_events, self.n_filters, samples)
        x = torch.sum(x, dim=2, keepdim=True)
        return x


class NoisePulseGenerator(nn.Module):

    def __init__(self, n_samples, n_frames):
        super().__init__()
        self.n_samples = n_samples
        self.n_frames = n_frames

        self.up = ConvUpsample(
            exp.model_dim, 
            exp.model_dim, 
            8, 
            end_size=n_frames, 
            mode='learned', 
            out_channels=1)
        
        self.to_decay = LinearOutputStack(exp.model_dim, 3, out_channels=1)

    
    def forward(self, x):
        batch, n_events, latent_dim = x.shape

        x = x.view(-1, exp.model_dim)
        decay = min_decay + (torch.sigmoid(self.to_decay(x)).view(batch * n_events, 1) * (1 - min_decay))

        x = self.up(x) ** 2
        x = x.view(batch * n_events, -1, self.n_frames)

        values = []

        # apply decay before upsampling
        for i in range(self.n_frames):
            if i == 0:
                values.append(x[:, :, i])
            else:
                decayed = (decay * values[i - 1])
                values.append((x[:, :, i] + decayed))
        
        x = torch.stack(values, dim=-1)

        x = F.interpolate(x, size=self.n_samples, mode='linear')

        noise = torch.zeros(x.shape[0], 1, self.n_samples, device=device).uniform_(-10, 10)
        x = x * noise
        x = x.view(batch, n_events, 1, self.n_samples)

        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.net = DilatedStack(exp.model_dim, [1, 3, 9, 27, 81, 1], dropout=0.1)
        self.attn = nn.Conv1d(exp.model_dim, 1, 1, 1, 0)
        self.to_positions = LinearOutputStack(exp.model_dim, 3, out_channels=1)

        self.noise = NoisePulseGenerator(exp.n_samples, 128)
        self.res = Resonance(
            exp.n_samples, 
            128, 
            filter_size=256, 
            n_filters=512
        )

        self.verb = ReverbGenerator(exp.model_dim, 3, exp.samplerate, exp.n_samples)

        self.apply(lambda x: exp.init_weights(x))
        
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)
        x = self.net(x)

        g = torch.mean(x, dim=-1)

        attn = self.attn(x)

        events, indices = sparsify_vectors(x, attn, n_to_keep=n_events)

        t = torch.sigmoid(self.to_positions(events))

        env = self.noise(events)

        final = self.res(env, events)

        # learnings from today indicate that this is only helpful for small shifts,
        # if a somewhat-matching segment overlaps a generated event.
        # For events separated by silence, this won't help
        final = fft_shift(
            final.view(-1, n_events, exp.n_samples), 
            t.view(-1, n_events, 1)
        )

        final = torch.sum(final, dim=1, keepdim=True)

        final = self.verb.forward(g, final)
        
        return final

model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class BandFilteredImpulseResponse(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    