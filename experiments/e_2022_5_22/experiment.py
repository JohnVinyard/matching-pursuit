
from upsample import ConvUpsample
from util.readmedocs import readme

from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules import stft
from modules.ddsp import NoiseModel, OscillatorBank, UnconstrainedOscillatorBank
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
small_batch = 16

latent_dim = 128
band = zounds.FrequencyBand(20, samplerate.nyquist)
mel_scale = zounds.MelScale(band, latent_dim)
fb = zounds.learn.FilterBank(
    samplerate, 512, mel_scale, 0.1, normalize_filters=True, a_weighting=False).to(device)

aim = AuditoryImage(512, 64, do_windowing=False, check_cola=True).to(device)

def perceptual_features(x):
    # x = fb.forward(x, normalize=False)
    # x = fb.temporal_pooling(x, 512, 256)
    # x = aim(x)
    # x = stft(x)
    x = codec.to_frequency_domain(x.squeeze())
    x = x[..., 0]
    return x

scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256

n_events = 8
sequence_length = n_frames
n_harmonics = 64
n_rooms = 8


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

class Discriminator(nn.Module):
    def __init__(self, channels=latent_dim):
        super().__init__()
        self.channels = channels

        self.embedding = nn.Embedding(n_clusters, channels)
        self.channels = channels

        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, self.channels, 7, 1, 3),
        )

        self.embed_spec = nn.Conv1d(256, self.channels, 3, 1, 1)

        self.net = nn.Sequential(
            nn.Conv1d(self.channels * 3, self.channels, 3, 1, 1),
            DilatedBlock(self.channels, 1, unit_norm=True),
            DilatedBlock(self.channels, 3, unit_norm=True),
            DilatedBlock(self.channels, 9, unit_norm=True),
            DilatedBlock(self.channels, 1, unit_norm=True),
            DilatedBlock(self.channels, 3, unit_norm=True),
            DilatedBlock(self.channels, 9, unit_norm=True),
            DilatedBlock(self.channels, 1, unit_norm=True, nl=False),
        )

        self.final = nn.Conv1d(self.channels, 1, 1, 1, 0)

        self.apply(init_weights)
    
    def forward(self, x, indices, norms):
        spec = x = perceptual_features(x).permute(0, 2, 1)
        x = self.embed_spec(x)

        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        x = torch.cat([x, embedded, amp], dim=1)

        features = []

        for layer in self.net:
            x = layer(x)
            features.append(x)
        
        latent = x
        x = self.final(x)
        return x, features, spec, latent


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

        self.to_osc = nn.Conv1d(c, c, 3, 1, 1)
        self.to_noise = nn.Conv1d(c, c, 3, 1, 1)

        # self.osc = OscillatorBank(
        #     input_channels=128,
        #     n_osc=128,
        #     n_audio_samples=n_samples,
        #     activation=torch.sigmoid,
        #     amp_activation=torch.abs,
        #     constrain=True,
        #     log_frequency=False,
        #     lowest_freq=40 / samplerate.nyquist,
        #     sharpen=False,
        #     compete=False)

        self.osc = UnconstrainedOscillatorBank(
            input_channels=128,
            n_osc=64,
            n_audio_samples=n_samples,
            fft_upsample=False,
            baselines=True)

        self.noise = NoiseModel(
            input_channels=128,
            input_size=64,
            n_noise_frames=512,
            n_audio_samples=n_samples,
            channels=128,
            activation=lambda x: x,
            squared=False,
            mask_after=1)
        
        # self.up = ConvUpsample(c, c, 64, 16384, 'fft_learned', c, from_latent=False)

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

        osc = self.to_osc(x)
        noise = self.to_noise(x)



        osc = self.osc(osc)
        noise = self.noise(noise)


        signal = osc + noise
        # signal = self.up(x)
        # signal = F.pad(signal, (0, 1))
        # signal = fb.transposed_convolve(signal)

        wet = self.verb.forward(signal, verb)
        x = (signal * mix) + (wet * (1 - mix))
        return x


gen = Generator(channels=latent_dim).to(device)
gen_optim = optimizer(gen, lr=1e-4)

disc = Discriminator(channels=latent_dim).to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(batch, indices, norms):
    gen_optim.zero_grad()
    fake = gen.forward(indices, norms)

    rj, rf, real_spec, _ = disc(batch, indices, norms)
    fj, ff, fake_spec, _ = disc(fake, indices, norms)

    loss = 0
    for i in range(len(rf)):
        loss = loss + F.mse_loss(ff[i], rf[i])
    
    fake_norms = torch.norm(fake_spec, dim=1).view(batch.shape[0], sequence_length)
    amp_loss = F.mse_loss(fake_norms, norms.view(batch.shape[0], sequence_length))

    

    loss = (amp_loss * 0.01) + loss
    
    loss.backward()
    gen_optim.step()
    return fake, loss

def train_disc(batch, indices, norms):
    disc_optim.zero_grad()
    fake = gen.forward(indices, norms)

    rj, rf, _, real_latent = disc(batch, indices, norms)
    fj, ff, _, fake_latent = disc(fake, indices, norms)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    return loss, real_latent, fake_latent

@readme
class UnconstrainedVQGan(object):
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

            if i % 2 == 0:
                self.fake, gen_loss = train_gen(real, indices, norms)
            else:
                disc_loss, self.rl, self.fl = train_disc(real, indices, norms)

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
                print('DISC', i, disc_loss.item())


    
    