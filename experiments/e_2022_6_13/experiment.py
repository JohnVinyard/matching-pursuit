from sklearn.cluster import MiniBatchKMeans
import torch
from torch import nn
from fm import HarmonicModel
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.atoms import unit_norm
from modules.ddsp import NoiseModel, OscillatorBank, UnconstrainedOscillatorBank
from modules.latent_loss import latent_loss
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util.readmedocs import readme
import zounds
import numpy as np
from torch.nn import functional as F
from util import device, playable
from modules.decompose import fft_frequency_recompose
from util.weight_init import make_initializer

n_clusters = 512
model_dim = 128
samplerate = zounds.SR22050()
n_samples = 2**14
n_frames = n_samples // 256


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
sequence_length = n_frames

init_weights = make_initializer(0.1)

band = zounds.FrequencyBand(20, samplerate.nyquist)
mel_scale = zounds.MelScale(band, model_dim)
fb = zounds.learn.FilterBank(
    samplerate, 512, mel_scale, 0.1, normalize_filters=True, a_weighting=False).to(device)
aim = AuditoryImage(512, 64, do_windowing=True, check_cola=True).to(device)
batch_size = 8


def perceptual_features(x):
    spec = fb.forward(x, normalize=False)
    spec = aim(spec)
    return spec


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
        x = x.view(-1, 1, n_frames)
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
        indices = self.embedding(indices).view(-1, n_frames, model_dim)
        norms = self.amp(norms).view(-1, n_frames, model_dim)
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
        )

        n_voices = 16
        n_profiles = 16
        self.to_f0 = nn.Conv1d(model_dim, n_voices * 2, 1, 1, 0)
        self.to_harm = nn.Conv1d(model_dim, n_profiles * n_voices, 1, 1, 0)


        self.osc = HarmonicModel(
            n_voices=n_voices, n_profiles=n_profiles, n_harmonics=64, reduce=torch.mean)
        # self.osc = OscillatorBank(
        #     model_dim,
        #     128,
        #     n_samples,
        #     constrain=True,
        #     lowest_freq=40 / samplerate.nyquist,
        #     complex_valued=False,
        #     amp_activation=lambda x: x ** 2)


        # self.osc = UnconstrainedOscillatorBank(
        #     model_dim, 128, n_samples, baselines=True)

        self.noise = NoiseModel(
            model_dim, n_frames, 512, n_samples, model_dim, mask_after=1, squared=True)

        
        self.to_rooms = LinearOutputStack(model_dim, 3, out_channels=8)
        self.to_mix = LinearOutputStack(model_dim, 3, out_channels=1)
        self.verb = NeuralReverb(n_samples, 8)
        self.apply(init_weights)

    def forward(self, indices, norms):
        x = self.cond.forward(indices, norms)
        x = self.net(x)

        j = x.mean(dim=-1)
        r = self.to_rooms(j)
        m = torch.sigmoid(self.to_mix(j)).view(-1, 1, 1)

        f0 = self.to_f0(x)
        harm = self.to_harm(x)
        osc = self.osc.forward(f0, harm)

        # osc = self.osc(x)
        # osc = self.
        noise = self.noise(x)
        signal = osc + noise

        wet = self.verb.forward(signal, r)
        signal = (signal * m) + (wet * (1 - m))
        return signal


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(model_dim * 2, model_dim, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(model_dim, model_dim, 3, 2, 1),
            nn.LeakyReLU(0.2),
        )

        self.dilated = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
        )

        self.classification_head = nn.Conv1d(model_dim, n_clusters, 1, 1, 0)
        self.amp_head = nn.Conv1d(model_dim, model_dim, 1, 1, 0)
        self.adv_head = nn.Conv1d(model_dim, model_dim, 1, 1, 0)

        self.judge = LinearOutputStack(model_dim, 3, out_channels=1)

        self.cond = ConditioningContext(n_clusters, model_dim)
        self.apply(init_weights)

    def forward(self, x, indices, norms):
        x = fb.forward(x, normalize=False)
        c = self.cond.forward(indices, norms)
        c = F.interpolate(c, size=n_samples, mode='nearest')

        x = torch.cat([x, c], dim=1)
        x = self.net(x)

        features = []
        for layer in self.dilated:
            x = layer(x)
            features.append(x.view(x.shape[0], -1))

        # j, a, f, x
        j = self.adv_head(x)
        feat = unit_norm(j, axis=1)
        j = self.judge(feat.permute(0, 2, 1))

        a = self.amp_head(x)
        f = self.classification_head(x)

        x = torch.cat(features, dim=-1)

        return j, a, f, feat.permute(0, 2, 1).reshape(-1, model_dim)


model = Generator().to(device)
optim = optimizer(model, lr=1e-4)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-4)


def train_gen(batch, indices, norms):
    optim.zero_grad()

    recon = model.forward(indices, norms)

    # j, a, f, fx = disc.forward(recon, indices, norms)
    # rj, ra, rf, rx = disc.forward(batch, indices, norms)

    # adv_loss = least_squares_generator_loss(j)

    ff = perceptual_features(recon)
    rf = perceptual_features(batch)
    recon_loss = F.mse_loss(ff, rf)

    loss = recon_loss

    loss.backward()
    optim.step()
    return loss, recon


def train_disc(batch, indices, norms):
    disc_optim.zero_grad()

    recon = model.forward(indices, norms)

    fj, fa, ff, fx = disc.forward(recon, indices, norms)
    rj, ra, rf, rx = disc.forward(batch, indices, norms)

    adv_loss = least_squares_disc_loss(rj, fj)
    
    # fll = latent_loss(fx)
    # rll = latent_loss(rx)

    loss = adv_loss #+ fll + rll

    loss.backward()
    disc_optim.step()
    return loss, rx


@readme
class ClassificationDiscriminator(object):
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
    
    def view_feat(self):
        return self.real_feat.data.cpu().numpy().reshape((-1, n_frames, model_dim))[0]

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)

            small_batch = 2

            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            norms = norms[:small_batch].to(device)
            real = item[:small_batch].view(-1, 1, n_samples)

            self.indices = indices
            self.norms = norms

            self.real = real

            if i % 2 == 0:
                gen_loss, self.fake = train_gen(real, indices, norms)
            else:
                # disc_loss, self.real_feat = train_disc(real, indices, norms)
                pass

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
                # print('DISC', i, disc_loss.item())
