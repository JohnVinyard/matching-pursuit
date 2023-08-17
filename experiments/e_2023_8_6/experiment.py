
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_shift import fft_shift
from modules.linear import LinearOutputStack
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.reverb import ReverbGenerator
from modules.sparse import soft_dirac, sparsify_vectors
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


n_events = 64

    

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

class Generator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Linear(exp.n_bands + 33, channels)
        # self.context = Context(channels, channels_last=True)
        self.context = Stack(channels, [1, 3, 9, 1, 3, 9, 1])
        self.proj = nn.Linear(channels, channels)
        self.attn = nn.Linear(channels, 1)

        self.atoms = nn.Parameter(torch.zeros(2048, 2048).uniform_(-0.01, 0.01))

        self.to_amp = LinearOutputStack(
            channels, 3, out_channels=1, norm=nn.LayerNorm((n_events, channels)))
        self.to_atom = LinearOutputStack(
            channels, 3, out_channels=2048, norm=nn.LayerNorm((n_events, channels)))
        self.to_pos = LinearOutputStack(
            channels, 3, out_channels=1, norm=nn.LayerNorm((n_events, channels)))
        
        self.verb = ReverbGenerator(
            channels, 3, exp.samplerate, n_samples=exp.n_samples, norm=nn.LayerNorm((channels,)))

        self.apply(lambda x: exp.init_weights(x))

    @property
    def normalized_atoms(self):
        return unit_norm(self.atoms)
        # return self.atoms
    

    def forward(self, x):
        batch = x.shape[0]

        x = exp.pooled_filter_bank(x).permute(0, 2, 1)
        pos = pos_encoded(x.shape[0], x.shape[1], 16, device=x.device)
        x = torch.cat([x, pos], dim=-1)
        x = self.down(x)

        x = x.permute(0, 2, 1)
        x = self.context(x)
        x = x.permute(0, 2, 1)

        attn = torch.softmax(self.attn(x).view(batch, -1), dim=-1)

        x = self.proj(x)

        context = torch.sum(x, dim=1)


        x = x.permute(0, 2, 1)
        attn = attn.view(batch, 1, -1)
        x, indices = sparsify_vectors(x, attn, n_to_keep=n_events, normalize=False)

        amps = torch.abs(self.to_amp(x))
        atom_selection = soft_dirac(self.to_atom(x), dim=-1)


        atoms = atom_selection @ self.normalized_atoms
        pos = self.to_pos(x)

        atoms = F.pad(atoms, (0, exp.n_samples - 8192)) * amps
        shifted = fft_shift(atoms, pos)[..., :exp.n_samples]

        x = shifted
        # x = torch.sum(shifted, dim=1, keepdim=True)

        x = self.verb.forward(context, x)


        return x



class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, 4, 3),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels),
        )
    
    def forward(self, x):
        return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            DownsamplingBlock(1, 16), # 8192
            DownsamplingBlock(16, 32), # 2048
            DownsamplingBlock(32, 64), # 512
            DownsamplingBlock(64, 128), # 128
            DownsamplingBlock(128, 256), # 32
            DownsamplingBlock(256, 512), # 8
            DownsamplingBlock(512, 1024), # 2
        )

        self.final = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(1024, 1024, 2, 2, 0),
                nn.LeakyReLU(0.2),
            ),
            nn.Conv1d(1024, 1, 1, 1, 0)
        )

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        
        x = self.final(x)
        return x, features


gen = Generator(1024).to(device)
g_optim = optimizer(gen, lr=1e-3)

disc = Discriminator().to(device)
d_optim = optimizer(disc, lr=1e-3)

def train(batch, i):

    g_optim.zero_grad()
    d_optim.zero_grad()


    recon = gen.forward(batch)

    loss = 0
    residual = batch.clone()
    for i in range(n_events):
        start_norm = torch.norm(residual, dim=-1)
        residual = residual - recon[:, i: i + 1, :]
        end_norm = torch.norm(residual, dim=-1)
        # maximize the change in norm for each atom individually
        diff = (start_norm - end_norm).sum()
        loss = loss - diff

    # loss = exp.perceptual_loss(recon, batch)

    loss.backward()
    g_optim.step()
    return loss, torch.sum(recon, dim=1, keepdim=True)



    if i % 2 == 0:
        print('G')
        recon = gen.forward(batch)

        _, rf = disc.forward(batch)
        j, ff = disc.forward(recon)

        g_loss = torch.abs(1 - j).mean()

        feat_loss = 0
        for a, b in zip(ff, rf):
            feat_loss = feat_loss + F.mse_loss(a, b)
        
        g_loss = (g_loss * 0) + feat_loss

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
    
    return loss, max_norm(recon)


@readme
class AdversarialSparseScheduler(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    