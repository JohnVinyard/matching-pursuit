from sklearn import model_selection
from sklearn.cluster import MiniBatchKMeans
from torch import nn
import torch
from fm import HarmonicModel
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.ddsp import NoiseModel, OscillatorBank, band_filtered_noise
from modules.erb import scaled_erb
import zounds
from modules.linear import LinearOutputStack
from modules.normal_pdf import pdf
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from torch.nn import functional as F
import numpy as np

from util.weight_init import make_initializer

n_clusters = 512

samplerate = zounds.SR22050()
band = zounds.FrequencyBand(30, samplerate.nyquist)
mel_scale = zounds.MelScale(band, 128)
n_samples = 2**14
fb = zounds.learn.FilterBank(samplerate, 512, mel_scale, 0.1, normalize_filters=True, a_weighting=True).to(device)
aim = AuditoryImage(512, 64, do_windowing=True, check_cola=True).to(device)
model_dim = 128
sequence_length = 64
n_frames = 128
n_noise_frames = 512
n_rooms = 8

scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
sequence_length = 64

init_weights = make_initializer(0.1)

def perceptual_feature(x):
    x = fb.forward(x, normalize=False)
    x = aim(x)
    return x


def to_frames(batch):
    spec = codec.to_frequency_domain(batch.view(-1, n_samples))[..., 0]
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
    return indices.reshape(-1, sequence_length, 1)


def decode_batch(kmeans, indices, norms):
    b, length, _ = indices.shape
    indices = indices.reshape((-1))
    frames = kmeans.cluster_centers_[indices]
    frames = frames.reshape((b, length, n_freq_bins))
    frames = torch.from_numpy(frames).to(device).float()
    return frames * norms


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 25, 1, 12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 25, 1, 12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 25, 1, 12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 25, 1, 12),
            nn.LeakyReLU(0.2),

            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 27),
            DilatedBlock(model_dim, 81),
            DilatedBlock(model_dim, 1),

            nn.Conv1d(model_dim, 1, 1, 1, 0)
        )

        self.apply(init_weights)
    
    def forward(self, x):
        return self.net(x)
    
class AudioModel(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

        self.osc = OscillatorBank(
            model_dim, 
            len(mel_scale), 
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

class EmbedClusterCenters(nn.Module):
    def __init__(self, n_clusters, dim):
        super().__init__()
        self.embedding = nn.Embedding(n_clusters, dim)

    def forward(self, x):
        return self.embedding(x)


class EmbedAmp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, dim, 7, 1, 3),
        )

    def forward(self, x):
        x = x.view(-1, 1, sequence_length)
        x = self.amp(x)
        x = x.permute(0, 2, 1)
        return x


class EmbedConditioning(nn.Module):
    def __init__(self, n_clusters, dim):
        super().__init__()
        self.embedding = EmbedClusterCenters(n_clusters, dim)
        self.amp = EmbedAmp(dim)
        self.reduce = LinearOutputStack(
            dim, 2, out_channels=dim, in_channels=dim*2)

    def forward(self, indices, norms):
        indices = self.embedding(indices).view(-1, sequence_length, model_dim)
        norms = self.amp(norms).view(-1, sequence_length, model_dim)
        x = torch.cat([indices, norms], dim=-1)
        x = self.reduce(x)
        return x


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


class ConditioningContext(nn.Module):
    def __init__(self, n_clusters, dim):
        super().__init__()
        self.cond = EmbedConditioning(n_clusters, dim)
        self.context = nn.Sequential(
            DilatedBlock(dim, 1),
            DilatedBlock(dim, 3),
            DilatedBlock(dim, 9),

        )

    def forward(self, indices, norms):
        x = self.cond.forward(indices, norms)
        x = x.permute(0, 2, 1)
        x = self.context(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond = ConditioningContext(n_clusters, model_dim)
        self.net = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 1),
            nn.ConvTranspose1d(model_dim, model_dim, 4, 2, 1)
        )
        self.audio = AudioModel(n_samples)
        self.apply(init_weights)
    
    def forward(self, indices, norms):
        cond = self.cond.forward(indices, norms)
        x = self.net(cond)
        x = self.audio(x)
        return x

model = Generator().to(device)
optim = optimizer(model, lr=1e-4)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(item, indices, norms):
    optim.zero_grad()
    item = item.view(-1, 1, n_samples)
    recon = model.forward(indices, norms)
    j = disc.forward(recon)

    loss = perceptual_loss(recon, item) + least_squares_generator_loss(j)
    loss.backward()
    optim.step()
    return loss, recon

def train_disc(item, indices, norms):
    disc_optim.zero_grad()

    item = item.view(-1, 1, n_samples)
    recon = model.forward(indices, norms)

    fj = disc.forward(recon)
    rj = disc.forward(item)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    return loss

@readme
class NoiseAndOscillatorExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None
        self.stream = stream
        self.real_feat = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.gen = model

    def view_spec(self):
        return self.spec.data.cpu().numpy()[0]

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
    
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            spec, norms = to_frames(item)
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)
            indices = encode_batch(self.kmeans, spec)
            indices = torch.from_numpy(indices).long().to(device)
            norms = norms.to(device)
            real = item

            small_batch = 2

            self.indices = indices
            self.norms = norms
            self.real = real



            if i % 2 == 0:
                loss, self.fake = train_gen(item[:small_batch], indices[:small_batch], norms[:small_batch])
            else:
                disc_loss = train_disc(item[:small_batch], indices[:small_batch], norms[:small_batch])

            if i % 10 == 0:
                try:
                    print(i)
                    print('G', loss.item())
                    print('D', disc_loss.item())
                    print('==========================')
                except:
                    pass

           