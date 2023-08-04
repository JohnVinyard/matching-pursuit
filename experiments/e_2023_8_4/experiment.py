
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.linear import LinearOutputStack
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme



exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=512,
    kernel_size=512)

n_frames = 128
sparse_frames = 8192
sparse_dim = 4096
sparsity = 512


class Context(nn.Module):
    def __init__(self, channels, channels_last=True):
        super().__init__()
        self.channels = channels
        self.channels_last = channels_last
        encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        self.net = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))
    
    def forward(self, x):
        if not self.channels_last:
            x = x.permute(0, 2, 1)
        
        x = self.net(x)

        if not self.channels_last:
            x = x.permute(0, 2, 1)

class AudioAnalysis(nn.Module):
    def __init__(self, spec_size, channels):
        super().__init__()
        self.channels = channels
        self.embed = nn.Linear(spec_size + 33, channels)
        self.context = Context(channels, channels_last=True)
        
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x).permute(0, 2, 1)
        pos = pos_encoded(x.shape[0], x.shape[1], 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.embed(x)
        x = self.context(x)
        return x


class SparseBottleneck(nn.Module):
    def __init__(self, in_channels, hidden_channels, k_sparse):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.k_sparse = k_sparse

        self.up = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)
        self.salience = nn.Conv1d(in_channels, hidden_channels, 1, 1, 0)

        self.agg = nn.Conv1d(hidden_channels, in_channels, 1, 1, 0)
        self.down = nn.Conv1d(hidden_channels, in_channels, 1, 1, 0)
    
    def forward(self, x):
        batch = x.shape[0]

        sig = self.up(x)
        sal = self.salience(x)
        sal = torch.softmax(sal.view(batch, -1), dim=-1).view(*sal.shape)

        x = sparsify(sal, self.k_sparse)
        x = sig * sal


        # agg = torch.sum(x, dim=-1, keepdim=True)

        x = self.down(x)
        # agg = self.agg(agg)

        # x = x + agg
        return x


class ExpandAndContractBottleneck(nn.Module):
    def __init__(self, start_frames, sparse_frames, channels, sparse_channels, k_sparse):
        super().__init__()
        self.start_frames = start_frames
        self.sparse_frames = sparse_frames
        self.channels = channels
        self.sparse_channels = channels
        self.k_sparse = k_sparse

        self.up = ConvUpsample(
            channels, 
            channels, 
            start_frames, 
            sparse_frames, 
            mode='nearest', 
            out_channels=channels,
            from_latent=False,
            batch_norm=True)

        self.bottleneck = SparseBottleneck(
            channels, sparse_channels, k_sparse=k_sparse)
        
        self.down = nn.Sequential(
            # 2048
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 512
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            # 128
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            
        )
    
    def forward(self, x):
        x = self.up(x)
        x = self.bottleneck(x)
        x = self.down(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis = AudioAnalysis(128, 512)
        self.bottleneck = ExpandAndContractBottleneck(128, 8192, 512, 4096, sparsity)
        self.context = Context(512, channels_last=False)
        self.to_samples = ConvUpsample(
            512, 
            512, 
            128, 
            end_size=exp.n_samples, 
            mode='nearest', 
            out_channels=512, 
            from_latent=False, 
            batch_norm=True)
    
    def forward(self, x):
        x = self.analysis(x)
        x = x.permute(0, 2, 1)
        x = self.bottleneck(x)
        x = self.context(x)
        x = self.to_samples(x)
        x = F.pad(x, (0, 1))
        x = exp.fb.transposed_convolve(x)[..., :exp.n_samples]
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.analysis = AudioAnalysis(128, 512)
        self.disc = LinearOutputStack(512, 4, out_channels=1, norm=nn.LayerNorm((128, 512)))
    
    def forward(self, x):
        x = self.analysis(x)
        x = self.disc(x)
        return x

gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)

def train(batch, i):
    if i % 2 == 0:
        recon = gen.forward(batch)

    else:
        pass

@readme
class SparseAdversarialLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    