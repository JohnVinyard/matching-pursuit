
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.ddsp import AudioModel
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded
from modules.sparse import sparsify, sparsify2
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    a_weighting=False)


k_sparse = 128
model_channels = 128
encoding_channels = 512
event_latent = 16
kernel_size = 1024


transformer_encoder = False

def sharpen(x):
    orig_shape = x.shape
    x = x.view(-1, 1, x.shape[1], x.shape[-1])
    pooled = F.avg_pool2d(x, (3, 3), stride=(1, 1), padding=(1, 1))
    sharpened = x - pooled
    sharpened = sharpened.view(orig_shape)
    return sharpened


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

class AudioAnalysis(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.up = nn.Conv1d(exp.n_bands, channels, 1, 1, 0)
        self.context = Stack(channels, [1, 3, 9, 1, 3, 9, 1])

        self.down = nn.Conv1d(exp.n_bands + 33, channels, 1, 1, 0)
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        if transformer_encoder:
            pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device).permute(0, 2, 1)
            x = torch.cat([x, pos], dim=1)
            x = self.down(x)
            x = x.permute(0, 2, 1)
            x = self.encoder(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.up(x)
            x = self.context(x)
        
        return x


class Expand(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.up = ConvUpsample(
            channels, 
            channels, 
            128, 
            end_size=exp.n_samples, 
            mode='nearest', 
            out_channels=channels, 
            from_latent=False, 
            batch_norm=True)
        
    
    def forward(self, x):
        x = self.up(x)
        return x

class Contract(nn.Module):
    def __init__(self, encoding_channels, channels):
        super().__init__()
        self.encoding_channels = encoding_channels
        self.channels = channels
        self.conv = nn.Conv1d(encoding_channels, channels, 1, 1, 0)
    
    def forward(self, x):
        return self.conv(x)

class SparseBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, k_sparse, sparsify2 = False):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.k_sparse = k_sparse
        self.sparsify2 = sparsify2

        self.up = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)
        self.salience = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)

    
    def forward(self, x):
        batch, channels, time = x.shape

        x = F.dropout(x, 0.02)
        sig = self.up(x)


        sal = self.salience(x)
        # sal = sharpen(sal)
        sal = torch.softmax(sal.view(batch, -1), dim=-1).view(*sal.shape)

        if self.sparsify2:
            x = sig * sal
            s, p, c = sparsify2(x, n_to_keep=self.k_sparse)
            return s, p, c
        else:
            s = sparsify(sal, n_to_keep=self.k_sparse)
            return s * sig

    

class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, 3, 1, dilation=dilation, padding=dilation)
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
    def __init__(self, channels, encoding_channels, k_sparse, kernel_size, latent_dim):
        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.k_sparse = k_sparse
        self.kernel_size = kernel_size
        self.latent_dim = latent_dim

        self.analysis = AudioAnalysis(channels)
        self.expand = Expand(channels)

        self.sparse = SparseBottleneck(
            channels, encoding_channels, k_sparse, sparsify2=False)
        self.contract = Contract(encoding_channels, channels)

        self.embed_global = nn.Linear(encoding_channels, latent_dim)

        self.latents = nn.Parameter(torch.zeros(encoding_channels, latent_dim).uniform_(-1, 1))

        self.to_dict = ConvUpsample(
            latent_dim,
            channels,
            start_size=4,
            out_channels=1,
            end_size=kernel_size,
            mode='nearest',
            batch_norm=True
        )
        self.apply(lambda x: exp.init_weights(x))
        
    
    def forward(self, x):
        batch = x.shape[0]

        x = self.analysis(x)
        x = self.expand(x)
        # s, p, c = self.sparse(x)
        s = self.sparse(x)

        agg = torch.sum(s, dim=-1)


        g = self.embed_global(agg)

        latents = self.latents[None, ...] + g[:, None, :]
        latents = latents.view(batch, encoding_channels, self.latent_dim)
        latents = latents.view(batch * encoding_channels, self.latent_dim)



        d = self.to_dict(latents)
        d = d.view(batch, encoding_channels, kernel_size)
        d = unit_norm(d, dim=-1)
        d = F.pad(d, (0, exp.n_samples - kernel_size))

        x = fft_convolve(s, d)[..., :exp.n_samples]
        x = torch.sum(x, dim=1, keepdim=True)

        x = max_norm(x)

        return x


model = Generator(
    model_channels, 
    encoding_channels, 
    k_sparse=128, 
    kernel_size=kernel_size,
    latent_dim=event_latent).to(device)

optim = optimizer(model, lr=1e-3)



def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class SparseStreamingCodec(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    