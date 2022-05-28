
from upsample import ConvUpsample
from util.readmedocs import readme

from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules import stft
from modules.ddsp import NoiseModel, OscillatorBank, HarmonicModel
from util.readmedocs import readme

import numpy as np
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from sklearn.cluster import MiniBatchKMeans
from modules.phase import MelScale, AudioCodec
import zounds
import torch
from torch import nn
from torch.nn import functional as F

from util.weight_init import make_initializer

n_clusters = 512
n_samples = 2 ** 14
samplerate = zounds.SR22050()
small_batch = 8

latent_dim = 128
n_rooms = 8


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256
sequence_length = n_frames


def perceptual_features(x):
    x = codec.to_frequency_domain(x.squeeze())
    x = x[..., 0]
    return x




init_weights = make_initializer(0.1)


def to_frames(batch):
    spec = codec.to_frequency_domain(batch)[..., 0]
    norms = torch.norm(spec, dim=-1, keepdim=True)
    spec = spec / (norms + 1e-8)
    return spec, norms


def update_kmeans(i, kmeans, frames):
    if i > 500:
        return
    frames = frames.view(-1, n_freq_bins).data.cpu().numpy()
    kmeans.partial_fit(frames)


def encode_batch(kmeans, frames):
    frames = frames.data.cpu().numpy().reshape(-1, n_freq_bins)
    indices = kmeans.predict(frames)
    return indices.reshape(-1, n_frames, 1)


def decode_batch(kmeans, indices, norms):
    b, length, _ = indices.shape
    indices = indices.reshape((-1))
    frames = kmeans.cluster_centers_[indices]
    frames = frames.reshape((b, length, n_freq_bins))
    frames = torch.from_numpy(frames).to(device).float()
    return frames * norms




class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation, unit_norm=False, nl=True):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.unit_norm = unit_norm
        self.nl = nl
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        if self.nl:
            x = F.leaky_relu(x, 0.2)
        if self.unit_norm:
            norms = torch.norm(x, dim=1, keepdim=True)
            x = x / (norms + 1e-12)
        return x



class Generator(nn.Module):
    def __init__(self, channels=latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_clusters, channels)
        self.channels = channels

        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, self.channels, 7, 1, 3),
        )

        c = self.channels

        self.context = nn.Sequential(
            nn.Conv1d(c * 2, c, 1, 1, 0),
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 1),
        )

        self.to_noise = nn.Conv1d(c, c, 3, 1, 1)

        self.n_voices = 16
        self.n_harmonics = 8
        self.n_profiles = 16


        self.to_fundamental = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, self.n_voices * 2, 3, 1, 1)
        )        

        self.to_harm = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, self.n_voices * self.n_profiles, 3, 1, 1)
        )
        
        self.harmonic = HarmonicModel(
            n_voices=self.n_voices,
            n_profiles=self.n_profiles,
            n_harmonics=self.n_harmonics,
            n_frames=n_frames,
            n_samples=n_samples
        )

        self.noise = NoiseModel(
            input_channels=128,
            input_size=64,
            n_noise_frames=512,
            n_audio_samples=n_samples,
            channels=128,
            activation=lambda x: x,
            squared=True,
            mask_after=1)
        

        self.to_verb_mix = nn.Conv1d(c, 1, 3, 1, 1)

        self.to_verb_params = LinearOutputStack(c, 3, out_channels=n_rooms)
        self.to_verb_mix = LinearOutputStack(c, 3, out_channels=1)

        
        self.verb = NeuralReverb(n_samples, n_rooms)

        self.apply(init_weights)

    def forward(self, indices, norms):
        # initial shape assertions
        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        x = torch.cat([embedded, amp], dim=1)
        x = self.context(x)

        verb = self.to_verb_params.forward(x.mean(dim=-1))
        mix = self.to_verb_mix.forward(x.mean(dim=-1))
        mix = torch.sigmoid(mix).view(-1, 1, 1)

        # I need (batch, n_voices, n_frames) and (batch, n_voices, n_harmonics, frames)

        f0 = self.to_fundamental(x).view(-1, self.n_voices, 2, n_frames)
        harm = self.to_harm(x).view(-1, self.n_voices, self.n_profiles, n_frames)

        osc = self.harmonic.forward(f0, harm)
        
        noise = self.to_noise(x)
        noise = self.noise(noise)


        signal = osc + noise
        wet = self.verb.forward(signal, verb)
        x = (signal * mix) + (wet * (1 - mix))
        return x


gen = Generator(channels=latent_dim).to(device)
gen_optim = optimizer(gen, lr=1e-4)



def train_gen(batch, indices, norms):
    gen_optim.zero_grad()
    fake = gen.forward(indices, norms)

    real_feat = perceptual_features(batch)
    fake_feat = perceptual_features(fake)

    loss = F.mse_loss(fake_feat, real_feat)

    loss.backward()
    gen_optim.step()
    return fake, loss


@readme
class HarmonicModelVQExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None

        self.rl = None
        self.fl = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.gen = gen

    def view_spec(self):
        return self.spec.data.cpu().numpy()[0]

    # def view_recon(self):
    #     return self.recon.data.cpu().numpy()[0]

    def view_indices(self):
        return self.indices[0].squeeze()

    def view_norms(self):
        return self.norms.data.cpu().numpy()[0].squeeze()

    def view_clusters(self):
        return self.kmeans.cluster_centers_

    def listen(self):
        return playable(self.fake, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def real_latent(self):
        return self.rl.data.cpu().numpy().squeeze()[0]
    
    def fake_latent(self):
        return self.fl.data.cpu().numpy().squeeze()[0]

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices
            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            norms = norms[:small_batch].to(device)
            real = item[:small_batch].view(-1, 1, n_samples)

            self.fake, gen_loss = train_gen(real, indices, norms)

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())


    
    