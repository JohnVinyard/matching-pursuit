
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from modules.mixer import MixerStack
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.softmax import sparse_softmax
from modules.stft import stft
from modules.transfer import ImpulseGenerator
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


class UNet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.down = nn.Sequential(
            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 4
            nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.up = nn.Sequential(
            # 8
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 16
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 32
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 64
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 128
            nn.Sequential(
                nn.Dropout(0.1),
                nn.ConvTranspose1d(channels, channels, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
        )

        self.proj = nn.Conv1d(1024, 4096, 1, 1, 0)
    
    def forward(self, x):
        # Input will be (batch, 1024, 128)
        context = {}

        for layer in self.down:
            x = layer(x)
            context[x.shape[-1]] = x
        
        for layer in self.up:
            x = layer(x)
            size = x.shape[-1]
            if size in context:
                x = x + context[size]
            
        x = self.proj(x)
        return x


    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(1025, 1024)

        self.net = UNet(1024)


        self.n_atoms = 1024
        self.atom_size = 1024
        self.resolution = 4096

        self.atoms = nn.Parameter(torch.zeros(1, self.n_atoms, self.atom_size).uniform_(-0.01, 0.01))

        self.pos = nn.Parameter(torch.zeros(1, 1024, 128).uniform_(-1, 1))

        self.embed_pos = nn.Conv1d(1024, 1024, 1, 1, 0)
        self.embed_spec = nn.Conv1d(1024, 1024, 1, 1, 0)

        self.to_selection = nn.Linear(1024, self.n_atoms)
        self.to_position = nn.Linear(1024, self.resolution)
        self.to_amp = nn.Linear(1024, 1)

        self.imp = ImpulseGenerator(
            exp.n_samples, softmax=lambda x: torch.softmax(x, dim=-1))
        
        self.apply(lambda x: exp.init_weights(x))
    
    @property
    def normed_atoms(self):
        return unit_norm(self.atoms)
    
    def forward(self, x):
        batch_size = x.shape[0]

        x = stft(x, 2048, 256, pad=True)
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2).reshape(-1, 1024, 128)
        pos = self.embed_pos(self.pos)
        spec = self.embed_spec(x)
        x = pos + spec
        x = self.net(x)

        # now, generate the events
        x = x.view(-1, 1024, 128).permute(0, 2, 1)

        # just 16 "events"
        x = x[:, 64 - 8: 64 + 8, :]

        sel = torch.softmax(self.to_selection(x), dim=-1)

        atoms = self.normed_atoms.permute(0, 2, 1) @ sel.permute(0, 2, 1)
        atoms = atoms.permute(0, 2, 1)

        atoms = F.pad(atoms, (0, exp.n_samples - self.atom_size))
        amps = torch.abs(self.to_amp(x))
        atoms = atoms * amps
        atoms = atoms.view(batch_size, -1, exp.n_samples)

        pos = self.to_position(x)

        impulses, logits = self.imp.forward(pos.view(-1, self.resolution), return_logits=True)
        impulses = impulses.view(batch_size, -1, exp.n_samples)

        final = fft_convolve(atoms, impulses)[..., :exp.n_samples]
        # final = torch.sum(final, dim=1, keepdim=True)

        return final, logits, sel

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def single_channel_loss(recon: torch.Tensor, target: torch.Tensor):

    target = stft(target, 512, 256, pad=True)

    everything = torch.sum(recon, dim=1, keepdim=True)
    everything = stft(everything, 512, 256, pad=True)

    residual = target - everything

    loss = 0

    for channel in range(recon.shape[1]):
        just_channel = recon[:, channel: channel + 1, :]
        just_channel = stft(just_channel, 512, 256, pad=True)
        t = residual + just_channel
        loss = loss + F.mse_loss(just_channel, t.clone().detach())
    

    # everything_removed = torch.relu(target - everything.clone().detach())

    # residual = everything_removed + just_channel

    # loss = F.mse_loss(just_channel, residual)

    return loss

# TODO: Consider trying out single-channel loss
def train(batch, i):
    optim.zero_grad()
    recon, logits, sel = model.forward(batch)

    sel = sel.view(-1, 1024)
    highest, indices = torch.max(sel, dim=-1)
    conf_loss = torch.abs(1 - highest).mean() * 0.01

    logits = logits.view(-1, 4096)
    highest, indices = torch.max(logits, dim=-1)
    print(indices // (exp.n_samples // 4096))

    # TODO: Also push atom selection to be very confident
    confidence_loss = torch.abs(1 - highest).mean() * 0.01

    # loss = exp.perceptual_loss(recon, batch) + confidence_loss

    # real_spec = stft(batch, 2048, 256, pad=True)
    # fake_spec = stft(recon, 2048, 256, pad=True)
    # loss = F.mse_loss(fake_spec, real_spec) + confidence_loss

    loss = single_channel_loss(recon, batch) + confidence_loss + conf_loss
    loss.backward()
    optim.step()

    recon = max_norm(torch.sum(recon, dim=1, keepdim=True))
    return loss, recon

@readme
class WaveCollapse(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    