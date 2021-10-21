from itertools import cycle
import torch
from torch.optim.adam import Adam
import zounds
from scipy.fftpack import dct, idct
from torch import nn

from datastore import batch_stream
from modules import PositionalEncoding
from modules3 import LinearOutputStack
from enum import Enum



sr = zounds.SR22050()
batch_size = 4
n_samples = 8192
path = '/hdd/musicnet/train_data'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.benchmark = True

def init_weights(p):
    with torch.no_grad():
        try:
            p.weight.normal_(0, 0.035)
        except AttributeError:
            pass

        try:
            p.bias.fill_(0)
        except AttributeError:
            pass

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8192, 1024),
            nn.LeakyReLU(0.2),

            LinearOutputStack(1024, 2),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),

            LinearOutputStack(512, 2),

            nn.Linear(512, 256),
            LinearOutputStack(256, 3, out_channels=1)
        )
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.net(x)
        return x


class ConvDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = PositionalEncoding(1, 8192, 8)

        self.freq = nn.Conv1d(1, 128, 7, 1, 3)
        self.time = nn.Conv1d(17, 128, 7, 1, 3)

        self.net = nn.Sequential(
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 2, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, 4, 4, 0),
        )

        self.apply(init_weights)
    
    def forward(self, x):
        x = x.view(-1, 1, 8192)

        p = self.time(self.pos.pos_encode.permute(0, 1).view(1, 17, 8192))
        f = self.freq(x)

        x = p + f
        x = self.net(x)
        x = torch.sigmoid(x)
        return x

class ConvGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = PositionalEncoding(1, 8192, 64)
        self.freq = nn.Conv1d(17, 128, 7, 1, 3)
        self.net = nn.Sequential(
            nn.Conv1d(256, 128, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, 7, 1, 3),
        )

        self.apply(init_weights)
    
    def forward(self, x):
        x = x.view(-1, 128, 1).repeat(1, 1, 8192)
        p = self.pos.pos_encode.permute(0, 1).view(1, 129, 8192).repeat(x.shape[0], 1, 1)

        x = torch.cat([x, p[:, :128, :]], dim=1)
        x = self.net(x)
        return x.view(-1, 8192)
        


class Generator(nn.Module):
    def __init__(self):
        super().__init__()


        self.net = nn.Sequential(
            nn.Linear(128, 512),
            nn.LeakyReLU(0.2),

            LinearOutputStack(512, 2),

            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),

            LinearOutputStack(1024, 2),

            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2),

            LinearOutputStack(2048, 2),

            nn.Linear(2048, 8192),
        )
        self.apply(init_weights)
    
    def forward(self, x):
        x = self.net(x)
        return x


gen = ConvGenerator().to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = ConvDiscriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

# one-sided label smoothing
real_target = 1
fake_target = 0

class LatentGenerator(object):
    def __init__(self):
        self._fixed = self._generate()

    def _generate(self):
        return torch.FloatTensor(
            batch_size, 128).normal_(0, 1).to(device)

    def __call__(self):
        return self._generate()


latent = LatentGenerator()


def least_squares_generator_loss(j):
    return 0.5 * ((j - real_target) ** 2).mean()


def least_squares_disc_loss(r_j, f_j):
    return 0.5 * (((r_j - real_target) ** 2).mean() + ((f_j - fake_target) ** 2).mean())


def train_disc(batch):
    disc_optim.zero_grad()
    with torch.no_grad():
        z = latent()
        recon = gen.forward(z)
    rj = disc.forward(batch)
    fj = disc.forward(recon)
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('Disc: ', loss.item())
    return batch, recon


def train_gen(batch):
    gen_optim.zero_grad()
    z = latent()
    recon = gen.forward(z)
    fj = disc.forward(recon)
    loss = least_squares_generator_loss(fj)
    loss.backward()
    gen_optim.step()
    print('Gen: ', loss.item())


class Turn(Enum):
    GEN = 'gen'
    DISC = 'disc'


turn = cycle([
    Turn.GEN, 
    Turn.DISC
])

def listen():
    return zounds.AudioSamples(idct(r, norm='ortho'), sr).pad_with_silence()

def real():
    return zounds.AudioSamples(idct(o, norm='ortho'), sr).pad_with_silence()

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)


    for batch in batch_stream(path, '*.wav', batch_size, n_samples):
        coeffs = dct(batch, axis=-1, norm='ortho')
        
        
        t = next(turn)
        c = torch.from_numpy(coeffs).to(device).float()

        if t == Turn.GEN:
            train_gen(c)
            
        else:
            orig, recon = train_disc(c)
            recon = recon.data.cpu().numpy()
            orig = orig.data.cpu().numpy()
            o = orig[0]
            r = recon[0]
    
    