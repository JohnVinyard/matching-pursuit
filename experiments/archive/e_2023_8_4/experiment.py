
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.linear import LinearOutputStack
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
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
    model_dim=128,
    kernel_size=512)

n_frames = 128
sparse_dim = 4096
sparsity = 64

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

        self.norm = nn.BatchNorm1d(in_channels)
    
    def forward(self, x):
        batch = x.shape[0]

        x = F.dropout(x, 0.02)

        sig = self.up(x)

        sal = self.salience(x)
        sal = torch.softmax(sal.view(batch, -1), dim=-1).view(*sal.shape)
        sal = sparsify(sal, self.k_sparse)
        x = sig * sal


        agg = F.avg_pool1d(x, 7, 1, 3)

        x = self.down(x)
        agg = self.agg(agg)

        x = x + agg
        # x = self.norm(x) 
        return x


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
        self.net = nn.Sequential(*[Block(channels, d) for d in dilations])
    
    def __iter__(self):
        return iter(self.net)
    
    def forward(self, x):
        x = self.net(x)
        return x

class AudioAnalysis(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.embed = nn.Linear(257, 8)
        self.channels = channels
        self.to_hidden = nn.Conv1d(1024, channels, 1, 1, 0)
        # self.norm = nn.LayerNorm((channels, 128))

    
    def forward(self, x):
        batch = x.shape[0]
        x = exp.perceptual_feature(x)
        x = self.embed(x)
        x = x.permute(0, 1, 3, 2).reshape(batch, 8 * exp.n_bands, -1)
        x = self.to_hidden(x)
        # x = self.norm(x)
        return x
    

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # self.net = nn.ConvTranspose1d(channels, channels, 8, 4, 2)
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode='nearest')
        # return self.net(x)



class Discriminator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.analyze = AudioAnalysis(channels)
        # self.context = Stack(channels, [1, 3, 9, 1, 3, 9, 1])
        # self.judge = LinearOutputStack(channels, 3, out_channels=1, norm=nn.LayerNorm((channels,)))

        self.net = nn.Sequential(
            # 32
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 8
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 4
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 2
            nn.Sequential(
                nn.Conv1d(channels, channels, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),
            # 2
            nn.Sequential(
                nn.Conv1d(channels, channels, 2, 2, 0),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(channels)
            ),

            nn.Conv1d(channels, 1, 1, 1, 0)
        )
        self.apply(lambda x: exp.init_weights(x))
        

    def forward(self, x):
        x = self.analyze(x)
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        
        return x, features


class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.analyze = AudioAnalysis(channels)
        self.context = Stack(channels, [1, 3, 9, 1, 3, 9, 1])
        self.bottleneck = SparseBottleneck(channels, sparse_dim, k_sparse=sparsity)
        self.context2 = Stack(channels, [1, 3, 9, 1, 3, 9, 1])
        
        self.up = nn.Sequential(

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            UpsampleBlock(512),

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            UpsampleBlock(256),

            nn.Conv1d(256, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            UpsampleBlock(128),

            nn.Conv1d(128, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            UpsampleBlock(64),

            nn.Conv1d(64, 1, 7, 1, 3),
        )        

        self.verb = ReverbGenerator(1024, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm(1024,))

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = self.analyze(x)
        x = self.context(x)
        x = self.bottleneck(x)
        x = self.context2(x)

        agg = torch.sum(x, dim=-1)

        x = self.up(x)

        # x = self.verb.forward(agg, x)

        return x


gen = Generator(1024).to(device)
g_optim = optimizer(gen, lr=1e-3)

disc = Discriminator(1024).to(device)
d_optim = optimizer(disc, lr=1e-3)

def train(batch, i):
    print(batch.shape)

    g_optim.zero_grad()
    d_optim.zero_grad()
    # latent = torch.zeros(batch.shape[0], 1024, device=device).uniform_(-1, 1)


    if i % 2 == 0:
        print('G')
        recon = gen.forward(batch)

        _, rf = disc.forward(batch)
        j, ff = disc.forward(recon)

        g_loss = torch.abs(1 - j).mean()

        feat_loss = 0
        for a, b in zip(ff, rf):
            feat_loss = feat_loss + F.mse_loss(a, b)
        
        g_loss = g_loss + feat_loss

        g_loss.backward()
        g_optim.step()
        loss = g_loss
    else:
        print('D')
        with torch.no_grad():
            recon = gen.forward(batch)
        
        fj, ff = disc.forward(recon)
        rj, rf = disc.forward(batch)
        d_loss = torch.abs(0 - fj).mean() + torch.abs(1 - rj).mean()
        d_loss.backward()
        d_optim.step()
        loss = d_loss
    
    return loss, recon

@readme
class SparseAdversarialLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    