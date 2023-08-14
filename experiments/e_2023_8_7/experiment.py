
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.activation import unit_sine
from modules.ddsp import AudioModel
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import soft_dirac, sparsify, sparsify_vectors
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


n_events = 512
d_size = 512
kernel_size = 512

    

# class Context(nn.Module):
#     def __init__(self, channels, channels_last=True):
#         super().__init__()
#         self.channels = channels
#         self.channels_last = channels_last
#         encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
#         self.net = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))
    
#     def forward(self, x):
#         if not self.channels_last:
#             x = x.permute(0, 2, 1)
        
#         x = self.net(x)

#         if not self.channels_last:
#             x = x.permute(0, 2, 1)
        
#         return x


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
        self.net = nn.Sequential(*[Block(channels, d) for d in dilations])
    
    def __iter__(self):
        return iter(self.net)
    
    def forward(self, x):
        x = self.net(x)
        return x

class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Conv1d(exp.n_bands, channels, 1, 1, 0)
        self.context = Stack(channels, [1, 3, 9, 1, 3, 9, 1])
        self.am = AudioModel(exp.n_samples, channels, exp.samplerate, 128, 1024, batch_norm=True)

        self.apply(lambda x: exp.init_weights(x))

    

    def forward(self, x):
        batch = x.shape[0]
        x = exp.pooled_filter_bank(x)
        x = self.up(x)
        x = self.context(x)
        x = self.am(x)
        return x
        


gen = Generator(1024).to(device)
g_optim = optimizer(gen, lr=1e-3)


def train(batch, i):
    g_optim.zero_grad()
    recon = gen.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    g_optim.step()
    return loss, recon




@readme
class TryingStuffOut(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    