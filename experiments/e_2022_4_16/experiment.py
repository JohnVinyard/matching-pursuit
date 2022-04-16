from modules import stft
from modules.atoms import AudioEvent
from modules.pif import AuditoryImage
from train.optim import optimizer
from util import device, playable, readme
from torch import nn
import zounds
import torch
from torch.nn import functional as F

sequence_length = 64
n_harmonics = 64
n_samples = 2 ** 14
samplerate = zounds.SR22050()
n_events = 8
latent_dim = 128


band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    samplerate, 
    512, 
    scale, 
    0.01, 
    normalize_filters=True, 
    a_weighting=False).to(device)
aim = AuditoryImage(512, 64, do_windowing=True, check_cola=True).to(device)

def perceptual_features(x):
    x = fb.forward(x, normalize=False)
    x = aim(x)
    return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.synth = Model()
    
    def forward(self, x):
        x = x.view(-1, latent_dim)
        # (batch, 128) => (batch, n_atoms, 128) => (batch, n_atoms, 70, 64)
    

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.atoms = AudioEvent(
            sequence_length=sequence_length,
            n_samples=n_samples,
            n_events=n_events,
            min_f0=20,
            max_f0=1000,
            n_harmonics=n_harmonics,
            sr=samplerate,
            noise_ws=512,
            noise_step=256)

        self.f0 = nn.Parameter(torch.zeros(
            1, n_events, sequence_length).uniform_(0, 1))
        self.osc_env = nn.Parameter(torch.zeros(
            1, n_events, sequence_length).uniform_(0, 0.1))
        self.noise_env = nn.Parameter(torch.zeros(
            1, n_events, sequence_length).uniform_(0, 0.1))
        self.overall_env = nn.Parameter(torch.zeros(
            1, n_events, sequence_length).uniform_(0, 1))
        self.noise_std = nn.Parameter(torch.zeros(
            1, n_events, sequence_length).uniform_(0, 1))

        
        self.harm_env = nn.Parameter(torch.zeros(
            1, n_events, n_harmonics, sequence_length).uniform_(0, 0.1))

        self.f0_baselines = nn.Parameter(torch.zeros(1, n_events, 1).uniform_(0, 1))

    def forward(self, x):
        x = self.atoms.forward(
            self.f0,
            self.overall_env,
            self.osc_env,
            self.noise_env,
            self.harm_env,
            self.noise_std,
            self.f0_baselines)
        x = x.sum(dim=1, keepdim=True)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon = model(None)
    # raw_loss = F.mse_loss(recon, batch)

    # real_spec = torch.log(1e-12 + stft(batch))
    # fake_spec = torch.log(1e-12 + stft(recon))
    # spec_loss = F.mse_loss(fake_spec, real_spec)

    # loss = raw_loss + spec_loss

    real_feat = perceptual_features(batch)
    fake_feat = perceptual_features(recon)

    loss = F.mse_loss(fake_feat, real_feat)
    # loss = raw_loss
    loss.backward()
    optim.step()
    print(loss.item())
    return recon


@readme
class AtomsRLExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.recon = None
        self.real = None

    def orig(self):
        return playable(self.real, samplerate)

    def listen(self):
        return playable(self.recon, samplerate)

    def run(self):
        for item in self.stream:
            self.real = item
            item = item.view(-1, 1, n_samples)
            self.recon = train_model(item)
