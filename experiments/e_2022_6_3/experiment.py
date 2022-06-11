
from loss.least_squares import least_squares_disc_loss
from util.readmedocs import readme

from modules.ddsp import NoiseModel, OscillatorBank
from util.readmedocs import readme

import numpy as np
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from modules.pos_encode import pos_encoded
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

batch_size = 4

latent_dim = 128


# band = zounds.FrequencyBand(20, samplerate.nyquist)
# mel_scale = zounds.MelScale(band, 128)
# fb = zounds.learn.FilterBank(
#     samplerate, 
#     512, 
#     mel_scale, 
#     0.1, 
#     normalize_filters=True, 
#     a_weighting=False).to(device)


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256
sequence_length = n_frames

n_steps = 25
means = [0] * n_steps
stds = [0.5] * n_steps


pos_embeddings = pos_encoded(batch_size, n_steps, 16, device=device)


def forward_process(audio, n_steps):
    degraded = audio
    for i in range(n_steps):
        noise = torch.zeros_like(audio).normal_(means[i], stds[i]).to(device)
        degraded = degraded + noise
    return audio, degraded, noise


def reverse_process(model, indices, norms):
    degraded = torch.zeros(indices.shape[0], 1, n_samples).normal_(0, 1.6).to(device)
    for i in range(n_steps - 1, -1, -1):
        pred_noise = model.forward(
            degraded, indices, norms, pos_embeddings[:indices.shape[0], i, :])
        degraded = degraded - pred_noise
    return degraded


init_weights = make_initializer(0.05)


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
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated1 = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)

    def forward(self, x):
        orig = x
        x = F.leaky_relu(self.dilated1(x), 0.2)
        x = self.conv(x)
        x = x + orig
        return x


class Generator(nn.Module):
    def __init__(self, channels=latent_dim):
        super().__init__()
        self.embedding = nn.Embedding(n_clusters, channels)
        self.channels = channels

        c = self.channels


        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, self.channels, 7, 1, 3),
        )

        self.embed_audio = nn.Conv1d(1, c, 25, 1, 12)

        self.reduce = nn.Conv1d(c * 2 + 33, c, 1, 1, 0)
        self.reduce_again = nn.Conv1d(c * 2, c, 1, 1, 0)


        self.blocks = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 27),
            DilatedBlock(c, 81),
            DilatedBlock(c, 1),
        )

        self.final = nn.Sequential(
            nn.Conv1d(c, c, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, 1, 1, 1, 0)
        )

        self.apply(init_weights)

    def forward(self, audio, indices, norms, pos_embedding):
        # initial shape assertions
        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)
        pos_embedding = pos_embedding.view(-1, 33, 1).repeat(1, 1, n_frames)

        audio = self.embed_audio(audio)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        x = torch.cat([embedded, amp, pos_embedding], dim=1)
        x = self.reduce(x)
        x = F.interpolate(x, size=n_samples, mode='linear')

        x = torch.cat([x, audio], dim=1)
        x = self.reduce_again(x)

        x = self.blocks(x)
        x = self.final(x)

        return x


gen = Generator(channels=latent_dim).to(device)
gen_optim = optimizer(gen, lr=1e-3)


def train_gen(batch, indices, norms):
    gen_optim.zero_grad()
    step = np.random.randint(1, n_steps)
    pos = pos_embeddings[:, step, :]
    orig, degraded, noise = forward_process(batch, step)
    pred_noise = gen.forward(degraded, indices, norms, pos)
    loss = torch.abs(pred_noise - noise).sum()
    loss.backward()
    gen_optim.step()
    return loss


@readme
class DiffusionWithBetterStepEmbedding(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None

        self.real = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.gen = gen

    def view_profiles(self):
        return self.gen.harmonic.profiles.data.cpu().numpy()

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
        return np.abs(zounds.spectral.stft(self.denoise()))

    def check_degraded(self):
        with torch.no_grad():
            audio, degraded, noise = forward_process(self.real, n_steps)
            return playable(degraded, samplerate)

    def denoise(self):
        with torch.no_grad():
            result = reverse_process(gen, self.indices[:1], self.norms[:1])
            return playable(result, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            indices = torch.from_numpy(indices).long().to(device)
            norms = norms.to(device)

            real = item.view(-1, 1, n_samples)

            self.real = real[:batch_size]
            self.indices = indices[:batch_size]
            self.norms = norms[:batch_size]

            gen_loss = train_gen(real[:batch_size], indices[:batch_size], norms[:batch_size])

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
