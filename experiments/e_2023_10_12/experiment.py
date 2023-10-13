
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


class Contextual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.down = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))

    def forward(self, x):
        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.down(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return x
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(257, 8)

        # self.net = MixerStack(1024, 1024, 128, 6, 4, channels_last=False)
        self.net = Contextual(1024)

        # self.net = nn.Sequential(
        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 2), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=1),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),

        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 6), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=3),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),

        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 18), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=9),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),

        #     nn.Sequential(
        #         nn.Dropout(0.05),
        #         nn.ConstantPad1d((0, 2), 0),
        #         nn.Conv1d(1024, 1024, 3, 1, dilation=1),
        #         nn.LeakyReLU(0.2),
        #         nn.BatchNorm1d(1024)
        #     ),


        #     nn.Conv1d(1024, 1024, 1, 1, 0)
        # )

        self.n_atoms = 1024
        self.atom_size = 1024
        self.resolution = 4096

        self.atoms = nn.Parameter(torch.zeros(1, self.n_atoms, self.atom_size).uniform_(-1, 1))

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

        x = exp.perceptual_feature(x)
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2).reshape(-1, 1024, 128)

        pos = self.embed_pos(self.pos)
        spec = self.embed_spec(x)

        x = pos + spec

        x = self.net(x)

        # now, generate the events
        x = x.view(-1, 1024, 128).permute(0, 2, 1)

        sel = torch.softmax(self.to_selection(x), dim=-1)

        atoms = self.normed_atoms.permute(0, 2, 1) @ sel.permute(0, 2, 1)
        atoms = atoms.permute(0, 2, 1)

        atoms = F.pad(atoms, (0, exp.n_samples - self.atom_size))
        amps = torch.abs(self.to_amp(x))
        atoms = atoms * amps

        pos = self.to_position(x)

        impulses, logits = self.imp.forward(pos.view(-1, self.resolution), return_logits=True)
        impulses = impulses.view(batch_size, -1, exp.n_samples)

        final = fft_convolve(atoms, impulses)
        final = torch.sum(final, dim=1, keepdim=True)

        return final, logits

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def single_channel_loss(recon: torch.Tensor, target: torch.Tensor):
    channel = np.random.randint(0, 128)

    target = stft(target, 512, 256, pad=True)

    just_channel = recon[:, channel: channel + 1, :]
    just_channel = stft(just_channel, 512, 256, pad=True)
    mask = torch.norm(just_channel, dim=-1, keepdim=True)
    mask = mask != 0

    everything = torch.sum(recon, dim=1, keepdim=True)
    everything = stft(everything, 512, 256, pad=True)

    everything_removed = torch.relu(target - everything.clone().detach())

    residual = everything_removed + just_channel

    # loss = F.mse_loss(just_channel, residual)
    loss = (((residual - just_channel) ** 2) * mask).mean()

    return loss

# TODO: Consider trying out single-channel loss
def train(batch, i):
    optim.zero_grad()
    recon, logits = model.forward(batch)
    
    logits = logits.view(128, -1)
    highest, indices = torch.max(logits, dim=-1)

    # TODO: Also push atom selection to be very confident
    confidence_loss = torch.abs(1 - highest).mean()

    loss = exp.perceptual_loss(recon, batch) + confidence_loss
    loss.backward()
    optim.step()

    recon = max_norm(torch.sum(recon, dim=1, keepdim=True))
    return loss, recon

@readme
class WaveCollapse(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    