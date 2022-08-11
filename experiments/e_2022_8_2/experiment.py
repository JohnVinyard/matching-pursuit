import numpy as np
import torch
from experiments.e_2022_4_10_a.experiment import DownsamplingBlock
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from modules.sparse import SparseAudioModel
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
import zounds
from torch import nn
from torch.nn import functional as F
from math import log2, log
from util.weight_init import make_initializer

n_samples = 2 ** 15
samplerate = zounds.SR22050()

start = int(log2(n_samples))
n_resolutions = 6
resolutions = [2**x for x in range(start, start - n_resolutions, -1)][::-1]

model_dim = 64

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

aim = AuditoryImage(512, 128, do_windowing=False, check_cola=False)

init_weights = make_initializer(0.1)


class BandConfig(object):
    def __init__(self, n_atoms, atom_size, atoms_to_keep):
        super().__init__()
        self.atom_size = atom_size
        self.n_atoms = n_atoms
        self.atoms_to_keep = atoms_to_keep


configs = {
    1024: BandConfig(512, 512, 128),
    2048: BandConfig(512, 512, 128),
    4096: BandConfig(512, 512, 128),
    8192: BandConfig(512, 512, 128),
    16384: BandConfig(512, 512, 128),
    32768: BandConfig(512, 512, 128)
}

print('TOTAL ATOMS', sum([c.atoms_to_keep for c in configs.values()]))

def perceptual_feature(x):
    # x = fb.forward(x, normalize=False)
    # x = aim.forward(x)
    # bands = fft_frequency_decompose(x, resolutions[0])

    bands = x
    x = torch.cat([v for v in bands.values()], dim=-1)
    return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    # loss = F.mse_loss(a, b)
    loss = torch.abs(a - b).sum()
    return loss


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
        x = F.leaky_relu(x, 0.2)
        return x

class BandDownSample(nn.Module):
    def __init__(self, start_size, end_size, n_atoms):
        super().__init__()
        self.start_size = start_size
        self.end_size = end_size
        self.n_atoms = n_atoms

        start = int(log2(start_size))
        stop = int(log2(end_size))
        n_layers = start - stop

        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(model_dim, model_dim, 3, 2, 1),
                nn.LeakyReLU(0.2)
            ) for _ in range(n_layers)
        ])
        self.final = nn.Conv1d(model_dim, n_atoms, 1, 1, 0)
    
    def forward(self, x):
        x = self.net(x)
        x = self.final(x)
        return x


class MultibandSparseModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.context = nn.Sequential(
            nn.Conv1d(n_bands, model_dim, 1, 1, 0),
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 243),
            DilatedBlock(model_dim, 1),
        )

        self.bands = nn.ModuleDict({
            str(k): nn.Sequential(
                BandDownSample(n_samples, k, configs[k].n_atoms),
                nn.Dropout(0.05),
                SparseAudioModel(k, configs[k].n_atoms, configs[k].atom_size, configs[k].atoms_to_keep)
            ) 
            for k in configs.keys()
        })

        self.apply(init_weights)

    def forward(self, x):
        x = fb.forward(x, normalize=True)
        x = self.context(x)

        bands = {k: self.bands[str(k)].forward(x) for k in configs.keys()}

        return bands


model = MultibandSparseModel().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)

    b = fft_frequency_decompose(batch, resolutions[0])
    loss = perceptual_loss(recon, b)

    loss.backward()
    optim.step()

    recon = fft_frequency_recompose(recon, n_samples)
    return loss, recon


@readme
class MultibandSparseExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.fake = None

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))

    def run(self):
        print(resolutions)

        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            loss, self.fake = train(item)
            print(loss.item())
