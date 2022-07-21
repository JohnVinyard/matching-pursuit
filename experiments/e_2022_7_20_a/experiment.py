import numpy as np
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.reverb import NeuralReverb
from train.optim import optimizer
from modules.latent_loss import latent_loss

from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer

n_samples = 2 ** 14
samplerate = zounds.SR22050()
model_dim = 128

n_frames = 128
n_noise_frames = 512

n_rooms = 8

init_weights = make_initializer(0.1)


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
        x = F.leaky_relu(x + orig, 0.2)
        return x


class EmbeddingBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 7, 1, 3),
            nn.LeakyReLU(0.2),
            DilatedBlock(out_channels, 9),
            nn.MaxPool1d(7, 4, 3),
            nn.Conv1d(out_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            EmbeddingBlock(1, 16),  # 4096
            EmbeddingBlock(16, 32),  # 1024
            EmbeddingBlock(32, 64),  # 256
            EmbeddingBlock(64, 128),  # 64
            EmbeddingBlock(128, 256),  # 16
            EmbeddingBlock(256, 512),  # 4
            nn.Conv1d(512, 128, 1, 1, 0),
            nn.Conv1d(128, 128, 4, 4, 0)
        )
        self.apply(init_weights)

    def forward(self, x):
        x = x.view(-1, 1, n_samples)
        x = self.net(x)
        x = x.view(x.shape[0], 128)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = EmbeddingModule()
        self.cond = LinearOutputStack(128, 3)
        self.judge = LinearOutputStack(128, 3, out_channels=1)
        self.apply(init_weights)
    
    def forward(self, x, e):
        x = self.embedding(x)
        e = self.cond(e)
        x = self.judge(x + e)
        return x

class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim, 
            model_dim, 
            n_samples, 
            constrain=True, 
            lowest_freq=40 / samplerate.nyquist,
            amp_activation=lambda x: x ** 2,
            complex_valued=False)
        
        self.noise = NoiseModel(
            model_dim,
            n_frames,
            n_noise_frames,
            n_samples,
            model_dim,
            squared=True,
            mask_after=1)
        
        self.verb = NeuralReverb(n_samples, n_rooms)

        self.to_rooms = LinearOutputStack(model_dim, 3, out_channels=n_rooms)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)

    
    def forward(self, x):
        x = x.view(-1, model_dim, n_frames)

        agg = x.mean(dim=-1)
        room = self.to_rooms(agg)
        mix = torch.sigmoid(self.to_mix(agg)).view(-1, 1, 1)

        harm = self.osc.forward(x)
        noise = self.noise(x)

        dry = harm + noise
        wet = self.verb(dry, room)
        signal = (dry * mix) + (wet * (1 - mix))
        return signal


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial = nn.Linear(128, 256 * 8)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='nearest'), # 32
            nn.Conv1d(256, 128, 7, 1, 3),

            nn.Upsample(scale_factor=4, mode='nearest'), # 128
            nn.Conv1d(128, 64, 7, 1, 3),

            nn.Upsample(scale_factor=4, mode='nearest'), # 512
            nn.Conv1d(64, 32, 7, 1, 3),

            nn.Upsample(scale_factor=4, mode='nearest'), # 2048
            nn.Conv1d(32, 16, 7, 1, 3),

            nn.Upsample(scale_factor=4, mode='nearest'), # 8192
            nn.Conv1d(16, 8, 7, 1, 3),

            nn.Upsample(scale_factor=2, mode='nearest'), # 16384
            nn.Conv1d(8, 1, 7, 1, 3),
        )
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.initial(x).view(-1, 256, 8)
        x = self.up(x)
        return x

embedding = EmbeddingModule().to(device)
embedding_optim = optimizer(embedding, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)

gen = Generator().to(device)
gen_optim = optimizer(gen, lr=1e-3)

def train_gen(batch):
    gen_optim.zero_grad()

    batch = batch[:, :, :n_samples]

    with torch.no_grad():
        z = embedding.forward(batch)
    
    audio = gen.forward(z)
    embedded = embedding.forward(audio)
    j = disc.forward(audio, z)

    adv_loss = least_squares_generator_loss(j)
    embedding_loss = F.mse_loss(embedded, z)

    loss = adv_loss + embedding_loss
    loss.backward()
    gen_optim.step()
    return loss, audio

def train_disc(batch):
    disc_optim.zero_grad()
    batch = batch[:, :, :n_samples]

    with torch.no_grad():
        z = embedding.forward(batch)
    
    audio = gen.forward(z)

    fj = disc.forward(audio, z)
    rj = disc.forward(batch, z)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    return loss

def train_embedding(batch):
    embedding_optim.zero_grad()
    e1 = embedding.forward(batch[:, :, :n_samples])
    e2 = embedding.forward(batch[:, :, n_samples:])

    all_embeddings = torch.cat([e1, e2], dim=0)
    ll = latent_loss(all_embeddings)
    dist_loss = ((e1 - e2) ** 2).mean()
    loss = ll + dist_loss
    loss.backward()
    embedding_optim.step()
    return loss, all_embeddings


@readme
class RepresentationExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.embedding = None
        self.fake = None
    
    def listen(self):
        return playable(self.fake, samplerate)
    
    def fake_spec(self):
        return np.log(1e-4 + np.abs(zounds.spectral.stft(self.listen())))

    def e(self):
        return self.embedding.data.cpu().numpy()

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples * 2)

            print('===============================')

            embedding_loss, self.embedding = train_embedding(item)
            print('E', embedding_loss.item())

            gen_loss, self.fake = train_gen(item)
            print('G', gen_loss.item())

            disc_loss = train_disc(item)
            print('D', disc_loss.item())
