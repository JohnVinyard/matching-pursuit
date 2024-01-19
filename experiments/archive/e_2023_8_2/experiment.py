
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.ddsp import AudioModel
from modules.normalization import max_norm
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

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, padding=None):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel, stride, padding if padding is not None else kernel // 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        return self.net(x)


class UpsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'),
            nn.Conv1d(in_channels, out_channels, 7, 1, 3),
            nn.BatchNorm1d(out_channels)
        )
    
    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        kernel_size = 25
        stride = 4

        self.net = nn.Sequential(
            # 8192
            DownsamplingBlock(1, 32, kernel_size, stride),

            # 2048
            DownsamplingBlock(32, 64, kernel_size, stride),

            # 512
            DownsamplingBlock(64, 128, kernel_size, stride),

            # 128
            DownsamplingBlock(128, 256, 7, 4),

            # 32
            DownsamplingBlock(256, 512, 7, 4),

            # 8
            DownsamplingBlock(512, 1024, 7, 4),

            # 4
            DownsamplingBlock(1024, 2048, 3, 2),

            nn.Conv1d(2048, 1024, 4, 4, 0)
        )
    
    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        return x, features


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()


        self.up = nn.Conv1d(1024, 512 * 4, 1, 1, 0)
        self.net = nn.Sequential(
            UpsamplingBlock(512, 256), # 16
            UpsamplingBlock(256, 128), # 64
            UpsamplingBlock(128, 64), # 256
            UpsamplingBlock(64, 32), # 1024
            UpsamplingBlock(32, 16), # 4096
            UpsamplingBlock(16, 8), # 16384
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv1d(8, 4, 7, 1, 3),
            nn.BatchNorm1d(4),
            nn.Conv1d(4, 1, 7, 1, 3)
        )

        self.am = AudioModel(exp.n_samples, 64, exp.samplerate, 256, 256, batch_norm=True)
    def forward(self, x):
        x = self.up(x)
        x = x.reshape(-1, 512, 4)
        x = self.net(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        encoded, features = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = Decoder()
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x = x.view(-1, 1024, 1)
        x = self.decoder(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.judge = nn.Conv1d(1024, 1, 1, 1, 0)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        x, features = self.encoder(x)
        x = self.judge(x)
        return x, features

# gen = Generator().to(device)
# g_optim = optimizer(gen, lr=1e-3)
gen = Model().to(device)
g_optim = optimizer(gen, lr=1e-3)

disc = Discriminator().to(device)
d_optim = optimizer(disc, lr=1e-3)


# model = Model().to(device)
# optim = optimizer(model, lr=1e-3)

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
class FrameBasedModel(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    