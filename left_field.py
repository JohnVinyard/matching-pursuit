from torch.nn.modules.conv import Conv1d
from modules import pos_encode
from train import least_squares_disc_loss, least_squares_generator_loss
from datastore import batch_stream
import zounds
import torch
from torch import nn
from torch.optim.adam import Adam
from itertools import cycle
from torch.nn import functional as F
from torch.nn.utils.clip_grad import clip_grad_value_

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_samples = 2 ** 15
sr = zounds.SR22050()
scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist), 128)
fb = zounds.learn.FilterBank(
    sr, 
    512, 
    scale, 
    0.01, 
    normalize_filters=False, 
    a_weighting=False).to(device)
fb.filter_bank = fb.filter_bank * 0.003

batch_size = 2
latent_size = 128


def init_weights(p):
    

    with torch.no_grad():
        try:
            p.weight.normal_(0, 0.02)
        except AttributeError:
            pass

        try:
            p.bias.fill_(0)
        except AttributeError:
            pass

def latent():
    return torch.FloatTensor(batch_size, latent_size, 1).normal_(0, 1).to(device)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        layers = 15
        channels = 128

        self.net = nn.Sequential(*[
            nn.Sequential(
                nn.Conv1d(channels, channels, 7, 2, 3),
                # nn.LeakyReLU(0.2)
                Activation()
            ) for _ in range(layers)
        ])

        self.final = nn.Conv1d(channels, 1, 1, 1, 0)
        self.apply(init_weights)

    def forward(self, x):
        batch, channels, time = x.shape
        x = fb.convolve(x)
        x = self.net(x)
        x = self.final(x)
        return x


class Activation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.leaky_relu(x, 0.2)

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Conv1d(in_channels, out_channels, 7, 1, 3)
    
    def forward(self, x):
        shortcut = x
        x = self.net(x)
        if shortcut.shape[1] == x.shape[1]:
            return F.leaky_relu(x + shortcut, 0.2)
        else:
            return F.leaky_relu(x, 0.2)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 128

        self.initial = nn.Conv1d(1, channels, 7, 1, 3)

        self.reduce = nn.Conv1d(channels * 2, channels, 7, 1, 3)

        self.net = nn.Sequential(

            Conv1d(channels * 2, channels, 7, 1, 3),
            Activation(),

            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
            GeneratorBlock(channels, channels),
        )

        self.register_buffer(
            'pos_encoded', 
            torch.from_numpy(pos_encode(1, n_samples, 64))[:128].float().repeat(batch_size, 1, 1))
        
        self.apply(init_weights)

    def forward(self, x):
        z = x.view(batch_size, latent_size, 1).repeat(1, 1, n_samples)
        z  = torch.cat([z, self.pos_encoded], dim=1)
        z = self.reduce(z)

        noise = torch.FloatTensor(batch_size, 1, n_samples).uniform_(-1, 1).to(x.device)
        noise = self.initial(noise)

        x = torch.cat([noise, z], dim=1)
        
        x = self.net(x)
        x = F.pad(x, (0, 1))
        x = fb.transposed_convolve(x)
        return x


disc = Discriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


gen = Generator().to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))


def train_disc(batch):
    disc_optim.zero_grad()
    batch = torch.from_numpy(batch).float().view(
        batch_size, 1, n_samples).to(device)
    rj = disc(batch)

    fake = gen(latent())
    fj = disc(fake)

    loss = least_squares_disc_loss(rj, fj)
    clip_grad_value_(disc.parameters(), 0.5)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


def train_gen(batch):
    gen_optim.zero_grad()
    z = latent()
    fake = gen(z)
    fj = disc(fake)
    loss = least_squares_generator_loss(fj)
    clip_grad_value_(gen.parameters(), 0.5)
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake.data.cpu().numpy()


def real():
    return zounds.AudioSamples(batch[0].squeeze(), sr).pad_with_silence()


def fake():
    return zounds.AudioSamples(recon[0].squeeze(), sr).pad_with_silence()


if __name__ == '__main__':

    path = '/hdd/musicnet/train_data'
    pattern = '*.wav'

    turns = cycle(['g', 'd'])

    torch.backends.cudnn.benchmark = True

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    for batch in batch_stream(path, pattern, batch_size, n_samples):
        mx = batch.max(axis=-1, keepdims=True)
        batch /= mx + 1e-12

        turn = next(turns)
        if turn == 'g':
            recon = train_gen(batch)
        else:
            train_disc(batch)
