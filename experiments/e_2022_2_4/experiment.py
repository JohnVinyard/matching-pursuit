import numpy as np
from config.dotenv import Config
from data.datastore import batch_stream
from modules import stft
from modules.ddsp import overlap_add
from modules.pos_encode import ExpandUsingPosEncodings, pos_encode_feature, pos_encoded
from modules.transformer import Transformer
from util.readmedocs import readme
import torch
from util import device
from util.weight_init import make_initializer
from torch import nn
from train import train_disc, train_gen, gan_cycle, get_latent
from torch.optim import Adam
import zounds

init_weights = make_initializer(0.05)
n_samples = 2**14
network_channels = 128

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_dim = n_samples // 256
        self.up = ExpandUsingPosEncodings(
            network_channels, self.time_dim, 16, network_channels, learnable_encodings=True)
        self.transformer = Transformer(network_channels, 5)

        self.to_samples = nn.Linear(network_channels, 512)

        self.to_real = nn.Linear(network_channels, 257)
        self.to_imag = nn.Linear(network_channels, 257)
    
    def forward(self, latent):
        batch_size = latent.shape[0]

        x = self.up(latent.view(batch_size, 1, network_channels))
        x = self.transformer(x)

        # windowed = torch.sin(self.to_samples(x))

        r = torch.sin(self.to_real(x))
        i = torch.sin(self.to_imag(x))

        coeffs = torch.complex(r, i)

        windowed = torch.fft.irfft(coeffs, dim=-1, norm='ortho')
        lapped = overlap_add(windowed[:, None, :, :])[..., :2**14]
        return lapped


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(257, network_channels)
        self.embed_pos = nn.Linear(33, network_channels)
        self.transformer = Transformer(network_channels, 5)
        self.judge = nn.Linear(network_channels, 1)
    
    def forward(self, x):

        pos = pos_encoded(x.shape[0], 64, 16, device=device)
        pos = self.embed_pos(pos)

        x = stft(x, pad=True)
        x = self.embed(x)

        x = pos + x
        x = self.transformer(x)
        x = self.judge(x)
        return x

gen = Generator().to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator().to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))


@readme
class PatchAudioTransformer(object):
    def __init__(self, batch_size=4, overfit=False):
        super().__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.overfit = overfit
        self.sr = zounds.SR22050()

        self.orig = None
        self.recon = None
    

    def fake(self):
        return zounds.AudioSamples(self.recon[0].data.cpu().numpy().squeeze(), self.sr).pad_with_silence()
    
    def fake_spec(self):
        return np.log(0.001 + np.abs(zounds.spectral.stft(self.fake())))
    

    def real(self):
        return zounds.AudioSamples(self.orig[0].squeeze(), self.sr).pad_with_silence()
    
    def real_spec(self):
        return np.log(0.001 + np.abs(zounds.spectral.stft(self.real())))
    
    def run(self):
        stream = batch_stream(
            Config.audio_path(), 
            '*.wav', 
            self.batch_size, 
            self.n_samples, 
            self.overfit)
        
        def make_latent():
            return get_latent(self.batch_size, network_channels)

        for batch in stream:
            self.orig = batch

            b = torch.from_numpy(batch).to(device)

            step = next(gan_cycle)

            if step == 'gen':
                self.recon = train_gen(b, gen, disc, gen_optim, make_latent)
            else:
                train_disc(b, disc, gen, disc_optim, make_latent)