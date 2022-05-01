import numpy as np
from train.gan import get_latent
from train.optim import optimizer
from upsample import ConvUpsample, FFTUpsampleBlock
from util import device, playable
from util.readmedocs import readme
from sklearn.cluster import MiniBatchKMeans
from modules.phase import MelScale, AudioCodec
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from train.gan import least_squares_disc_loss, least_squares_generator_loss

from util.weight_init import make_initializer

n_clusters = 512
n_samples = 2 ** 14
samplerate = zounds.SR22050()
small_batch = 2

latent_dim = 128
band = zounds.FrequencyBand(20, samplerate.nyquist)
mel_scale = zounds.MelScale(band, latent_dim)
fb = zounds.learn.FilterBank(
    samplerate, 512, mel_scale, 0.1, normalize_filters=True, a_weighting=False)

scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256


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


class Discriminator(nn.Module):
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

        def block(): return nn.Sequential(
            nn.Conv1d(c, c, 13, 4, 5),
            nn.LeakyReLU(0.2),
        )

        self.down = nn.Sequential(
            block(),
            block(),
            block(),
            block(),
        )

        self.final = nn.Conv1d(c, 1, 1, 1, 0)

        self.apply(init_weights)

    def forward(self, x, indices, norms):
        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        c = embedded + amp
        c = c.repeat(1, 1, n_samples // n_frames)
        x = fb.forward(x, normalize=False)

        x = x + c

        features = []
        for layer in self.down:
            x = layer(x)
            features.append(x.view(x.shape[0], -1))

        return x, torch.cat(features, dim=-1)


class Generator(nn.Module):
    def __init__(self, channels=latent_dim, fft=False):
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

        if fft:
            def block(size): return nn.Sequential(
                FFTUpsampleBlock(c, c, size, factor=4, infer=True),
                nn.LeakyReLU(0.2)
            )
        else:
            def block(size): return nn.Sequential(
                nn.ConvTranspose1d(c, c, 8, 4, 2),
                nn.LeakyReLU(0.2),
            )

        self.up = nn.Sequential(
            block(64),
            block(256),
            block(1024),
            block(4096),
            nn.Conv1d(c, c, 7, 1, 3)
        )

        self.apply(init_weights)

    def forward(self, z, indices, norms):
        # initial shape assertions
        z = z.view(-1, latent_dim)
        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        z = z.view(-1, latent_dim, 1).repeat(1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        x = z + embedded + amp

        x = self.up(x)

        x = F.pad(x, (0, 1))
        x = fb.transposed_convolve(x)
        return x


gen = Generator(latent_dim, fft=False).to(device)
gen_optim = optimizer(gen, lr=1e-3)

disc = Discriminator(latent_dim).to(device)
disc_optim = optimizer(disc)


def train_gen(batch, indices, norms):
    gen_optim.zero_grad()
    z = get_latent(batch.shape[0], latent_dim)
    fake = gen.forward(z, indices, norms)

    fj, f_feat = disc.forward(fake, indices, norms)
    rj, r_feat = disc.forward(batch, indices, norms)

    feat_loss = torch.abs(f_feat - r_feat).sum()

    judge_loss = least_squares_generator_loss(fj) + feat_loss

    loss = feat_loss + judge_loss
    loss.backward()
    gen_optim.step()
    print('G', loss.item())
    return fake


def train_disc(batch, indices, norms):
    disc_optim.zero_grad()
    z = get_latent(batch.shape[0], latent_dim)
    fake = gen.forward(z, indices, norms)
    fj, _ = disc.forward(fake, indices, norms)
    rj, _ = disc.forward(batch, indices, norms)
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('D', loss.item())


@readme
class TokenExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)

    def view_spec(self):
        return self.spec.data.cpu().numpy()[0]

    def view_recon(self):
        return self.recon.data.cpu().numpy()[0]

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
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices
            decoded = decode_batch(self.kmeans, indices, norms)
            self.recon = decoded

            indices = torch.from_numpy(indices).long()[:small_batch]
            norms = norms[:small_batch]
            item = item[:small_batch].view(-1, 1, n_samples)

            if i % 2 == 0:
                self.fake = train_gen(item, indices, norms)
            else:
                train_disc(item, indices, norms)
