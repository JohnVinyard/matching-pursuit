import zounds
from data.datastore import batch_stream
from modules.atoms import Atoms
from modules.linear import LinearOutputStack
from modules.multiresolution import BandEncoder, EncoderShell
from modules.psychoacoustic import PsychoacousticFeature
from modules.transformer import Transformer
from util import device
from config import Config
import torch
from torch import nn
from torch.optim import Adam
from itertools import chain
from torch.nn import functional as F
import numpy as np
from data import feature, compute_feature_dict

from util.readmedocs import readme
from util.weight_init import make_initializer

torch.backends.cudnn.benchmark = True

network_channels = 64
n_samples = 2 ** 14


init_weights = make_initializer(0.1)


class EncoderBranch(nn.Module):
    def __init__(self, band_size, periodicity_feature_size):
        super().__init__()
        self.encoder = BandEncoder(
            network_channels, periodicity_feature_size, band_size)
        self.linear = nn.Linear(512, network_channels)
        self.periodicity_feature_size = periodicity_feature_size
        self.band_size = band_size

    def forward(self, x):
        x = x.view(-1, 64, 32, self.periodicity_feature_size)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        x = self.linear(x)
        return x


class Summarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.reducer = nn.Linear(
            feature.n_bands * network_channels, network_channels)
        
        self.summary = nn.Sequential(
            nn.Conv1d(network_channels, network_channels, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(network_channels, network_channels, 2, 2, 0),
        )
        self.to_latent = nn.Linear(network_channels, network_channels)

    def forward(self, x):
        x = self.reducer(x)
        x = x.permute(0, 2, 1)
        x = self.summary(x)
        x = x.permute(0, 2, 1)
        x = self.to_latent(x)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.shell = EncoderShell(
            network_channels, make_band_encoder, Summarizer, feature)
        
        self.apply(init_weights)

    def forward(self, x):
        x = self.shell(x)
        return x


def make_band_encoder(periodicity, band_size):
    return EncoderBranch(band_size, periodicity)



class Decoder(nn.Module):
    def __init__(self, n_atoms, atom_latent, n_audio_samples):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_latent = atom_latent
        self.n_audio_samples = n_audio_samples

        self.atoms = Atoms(atom_latent, n_audio_samples, network_channels)

        self.decoder = LinearOutputStack(network_channels, 5, out_channels=n_atoms * atom_latent)
        self.apply(init_weights)
    
    def forward(self, x):
        # this is our latent representation
        x = x.view(-1, network_channels)
        # we need to transform it into (batch, n_atoms, atom_latent)
        x = self.decoder(x)
        x = x.view(-1, self.n_atoms, self.atom_latent)
        x, _, _, _ = self.atoms(x)
        return x

encoder = Encoder().to(device)
decoder = Decoder(n_atoms=16, atom_latent=16, n_audio_samples=n_samples).to(device)

optim = Adam(
    chain(decoder.parameters(), encoder.parameters()), lr=1e-3, betas=(0, 0.9))


@readme
class AtomsDecoder(object):
    def __init__(self, overfit=False):
        super().__init__()
        self.sr = zounds.SR22050()
        self.n_samples = n_samples
        self.batch_size = 2
        self.overfit = overfit

        self.decoded = None
        self.orig = None
        self.encoded = None
    
    def real(self):
        return zounds.AudioSamples(self.orig[0].reshape(-1), self.sr).pad_with_silence()
    
    def fake(self):
        return zounds.AudioSamples(self.decoded[0].data.cpu().numpy().reshape(-1), self.sr).pad_with_silence()
    
    def real_spec(self):
        return np.log(0.0001 + np.abs(zounds.spectral.stft(self.real())))

    def fake_spec(self):
        return np.log(0.0001 + np.abs(zounds.spectral.stft(self.fake())))
    
    def latent(self):
        return self.encoded.data.cpu().numpy().squeeze()

    def run(self):
        stream = batch_stream(
            Config.audio_path(),
            '*.wav',
            self.batch_size,
            self.n_samples,
            overfit=self.overfit)

        for batch in stream:
            optim.zero_grad()

            batch /= (batch.max(axis=-1, keepdims=True) + 1e-12)
            self.orig = batch

            
            samples = torch.from_numpy(batch).float().to(device)
            
            orig_feat = compute_feature_dict(samples)
            
            encoded = encoder.forward(orig_feat)
            self.encoded = encoded
            decoded = decoder.forward(encoded)
            self.decoded = decoded
            # recon_feat = compute_feature_dict(decoded)

            of, _ = feature.forward(samples)
            rf, _ = feature.forward(decoded)

            loss = F.mse_loss(of, rf)
            
            loss.backward()
            optim.step()

            print('AE', loss.item())

