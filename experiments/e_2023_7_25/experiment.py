
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.activation import unit_sine
from modules.decompose import fft_frequency_decompose
from modules.fft import fft_convolve
from modules.floodfill import flood_fill_loss
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import soft_dirac, sparsify
from modules.transfer import ImpulseGenerator
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from torch.distributions import Normal

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    a_weighting=True)


# total number of atoms
d_size = 1024

# length (in samples) of each atom
kernel_size = 1024

# the number of events per segment
n_atoms = 512



def training_softmax(x):
    """
    Produce a random mixture of the soft and hard functions, such
    that softmax cannot be replied upon.  This _should_ cause
    the model to gravitate to the areas where the soft and hard functions
    are near equivalent
    """
    mixture = torch.zeros(x.shape[0], x.shape[1], 1, device=x.device).uniform_(0, 1)
    sm = torch.softmax(x, dim=-1)
    d = soft_dirac(x)
    return (d * mixture) + (sm * (1 - mixture))


class AudioAnalysis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Conv1d(exp.n_bands, channels, 7, 1, 3)
    
    def forward(self, x):
        x = exp.fb.forward(x, normalize=False)
        x = self.net(x)
        return x



class AudioSynthesis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.atoms = nn.Parameter(torch.zeros(1, d_size, kernel_size).uniform_(-1, 1))

        self.to_context = nn.Linear(d_size, channels)
        self.verb = ReverbGenerator(channels, 2, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((channels)))
    
    def forward(self, x):
        batch, n_atoms, time = x.shape

        context = torch.sum(x, dim=-1)
        context = self.to_context(context)

        atoms = unit_norm(self.atoms, dim=-1)
        x = F.pad(x, (0, kernel_size))
        x = F.conv1d(x, atoms)
        x = x[..., :exp.n_samples]        

        x = self.verb.forward(context, x)
        return x


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, 3, 1, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x

class Stack(nn.Module):
    def __init__(self, channels, dilations):
        super().__init__()
        self.channels = channels
        self.dilation = dilations
        self.transformer_mode = False

        if self.transformer_mode:
            self.proj = nn.Linear(33, channels)
            encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
            self.net = nn.TransformerEncoder(encoder, 5, norm=nn.LayerNorm((128, channels)))
        else:
            self.net = nn.Sequential(*[Block(channels, d) for d in dilations])
        
    
    def forward(self, x):
        if self.transformer_mode:
            pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device)
            pos = self.proj(pos)

            x = x.permute(0, 2, 1)
            x = pos + x
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.net(x)
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels, encoding_channels):
        super().__init__()

        self.k_sparse = n_atoms
        self.in_channels = in_channels
        self.encoding_channels = encoding_channels
        self.up = nn.Conv1d(in_channels, encoding_channels, 1, 1, 0)
        self.mask = nn.Conv1d(in_channels, encoding_channels, 1, 1, 0)

        # self.summary = nn.Linear(encoding_channels, in_channels)
        self.down = nn.Conv1d(encoding_channels, in_channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(in_channels)
    
    def forward(self, x):
        batch = x.shape[0]

        x = F.dropout(x, 0.02)

        orig = x

        x = self.up(x)

        mask = self.mask(orig)
        sm = torch.softmax(mask.view(batch, -1), dim=-1)
        sm = sm.view(x.shape)
        sm = sparsify(sm, n_to_keep=self.k_sparse)

        # multiply mask by computed values
        return sm * x
        

class Model(nn.Module):
    def __init__(self, channels, encoding_channels):
        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.analyze = AudioAnalysis(channels)
        self.encoder = Stack(channels, [1, 3, 9, 27, 81, 243, 1])
        self.bottleneck = Bottleneck(channels, encoding_channels)
        # self.decoder = Stack(channels, [1, 3, 9, 27, 1])
        self.synthesize = AudioSynthesis(channels)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = self.analyze(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.synthesize(x)
        x = max_norm(x)
        return x

def extract_windows(x, kernel, step):
    kw, kh = kernel
    sw, sh = step

    x = x.unfold(-1, kw, sw)
    x = x.unfold(1, kh, sh)

    # win = torch.hamming_window(kw, device=x.device)[:, None] * torch.hamming_window(kh, device=x.device)[None, :]
    # x = x * win[None, None, None, :, :]

    x = x.reshape(*x.shape[:-2], np.product(x.shape[-2:]))
    x = unit_norm(x, dim=-1)


    return x

def extract_feature(x):
    x = exp.pooled_filter_bank(x)
    return x


def exp_loss(a, b):

    atoms = unit_norm(model.synthesize.atoms[0])
    atoms = atoms @ atoms.T
    atoms = torch.triu(a)
    ortho_loss = torch.abs(atoms).mean()
    print('ortho', ortho_loss.item())

    a = extract_feature(a)
    b = extract_feature(b)
    spec_loss = F.mse_loss(a, b)
    return spec_loss + ortho_loss

model = Model(512, d_size).to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    batch_size = batch.shape[0]


    optim.zero_grad()
    recon = model.forward(batch)
    
    
    # loss = exp.perceptual_loss(recon, batch)
    loss = exp_loss(recon, batch)
    

    loss.backward()
    optim.step()

    recon = torch.sum(recon, dim=1, keepdim=True)

    return loss, max_norm(recon)

@readme
class NeuralMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    