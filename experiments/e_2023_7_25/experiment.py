
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import soft_dirac, sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    a_weighting=True)



channels = 256
encoding_channels = 512
kernel_size = 512
sparsity = 512


# TODO: Maybe this should be the PIF feature
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

        # self.to_samples = nn.Conv1d(channels, 1, 25, 1, 12)

        self.atoms = nn.Parameter(torch.zeros(1, encoding_channels, kernel_size).uniform_(-1, 1))

        self.to_context = nn.Linear(channels, channels)
        self.verb = ReverbGenerator(channels, 2, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((channels)))
    
    def forward(self, x):
        batch, n_atoms, time = x.shape

        # context = torch.sum(x, dim=-1)
        # context = self.to_context(context)

        # atoms = unit_norm(self.atoms, dim=-1)
        # x = F.pad(x, (0, kernel_size))
        # x = F.conv1d(x, atoms)
        # x = x[..., :exp.n_samples]        

        # x = self.to_samples(x)
        # x = F.pad(x, (0, 1))
        # x = exp.fb.transposed_convolve(x * 0.001)[..., :exp.n_samples]


        # x = self.verb.forward(context, x)

        atoms = F.pad(self.atoms, (0, exp.n_samples - kernel_size))
        x = fft_convolve(atoms, x)
        print(x.shape)
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
    
    def forward(self, x):
        x = self.net(x)
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

        self.norm = nn.BatchNorm1d(in_channels)
    
    def forward(self, x):
        batch = x.shape[0]

        x = F.dropout(x, 0.02)

        sig = self.up(x)

        sal = self.salience(x)
        sal = torch.softmax(sal.view(batch, -1), dim=-1).view(*sal.shape)
        sal = sparsify(sal, self.k_sparse)
        x = sig * sal
        return x

        # agg = F.avg_pool1d(x, 25, 1, 12)

        # x = self.down(x)
        # agg = self.agg(agg)

        # x = x + agg
        # x = self.norm(x) 
        # return x
        

class Model(nn.Module):
    def __init__(self, channels, encoding_channels):
        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.analyze = AudioAnalysis(channels)
        self.encoder = Stack(channels, [1, 3, 9, 27, 81, 243, 1])
        self.bottleneck = SparseBottleneck(channels, encoding_channels, k_sparse=sparsity)
        self.decoder = Stack(channels, [1, 3, 9, 27, 1])
        self.synthesize = AudioSynthesis(channels)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = self.analyze(x)
        x = self.encoder(x)
        x = self.bottleneck(x)
        # x = self.decoder(x)
        x = self.synthesize(x)
        return x



class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # 32
            nn.Sequential(
                nn.Conv1d(128, 256, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 8
            nn.Sequential(
                nn.Conv1d(256, 512, 7, 4, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(512)
            ),


            # 4
            nn.Sequential(
                nn.Conv1d(512, 1024, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            # 2
            nn.Sequential(
                nn.Conv1d(1024, 1024, 3, 2, 1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Conv1d(1024, 1, 2, 2, 0)
        )
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        
        return x, features


model = Model(channels, encoding_channels).to(device)
optim = optimizer(model, lr=1e-3)


disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()

    loss = 0
    recon = model.forward(batch)
    residual = batch.clone()

    for i in range(sparsity):
        start_norm = torch.norm(residual, dim=-1)
        residual = residual - recon[:, i, :]
        end_norm = torch.norm(residual, dim=-1)
        diff = (start_norm - end_norm).mean()
        loss = loss - diff

    
    loss.backward()
    return loss, torch.sum(recon, dim=1, keepdim=True)


    if i % 2 == 0:
        recon = model.forward(batch)
        fj, ff = disc.forward(recon)
        rj, rf = disc.forward(batch)
        adv_loss = torch.abs(1 - fj).mean()
        feat_loss = 0
        for a, b in zip(ff, rf):
            feat_loss = feat_loss + F.mse_loss(a, b)
        loss = adv_loss + feat_loss
        loss.backward()
        optim.step()
    else:
        with torch.no_grad():
            recon = model.forward(batch)
        rj, _ = disc.forward(batch)
        fj, _ = disc.forward(recon)
        loss = torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()
        loss.backward()
        disc_optim.step()
    
    return loss, recon

@readme
class NeuralMatchingPursuit(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    