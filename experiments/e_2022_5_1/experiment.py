import numpy as np
from modules.atoms import AudioEvent
from modules.linear import LinearOutputStack
from modules.metaformer import AttnMixer, MetaFormer
from modules.pif import AuditoryImage
from modules.pos_encode import pos_encoded
from modules.reverb import NeuralReverb
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
    samplerate, 512, mel_scale, 0.1, normalize_filters=True, a_weighting=False).to(device)

aim = AuditoryImage(512, 64, do_windowing=False, check_cola=True).to(device)

def perceptual_features(x):
    x = fb.forward(x, normalize=False)
    x = aim(x)
    return x

scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256

n_events = 8
sequence_length = n_frames
n_harmonics = 64
n_rooms = 8


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


class SynthGenerator(nn.Module):
    def __init__(self, n_atoms=n_events, channels=latent_dim):
        super().__init__()

        self.channels = channels

        self.embedding = nn.Embedding(n_clusters, channels)

        self.n_atoms = n_atoms

        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, self.channels, 7, 1, 3),
        )

        c = channels

        # self.net = MetaFormer(
        #     128,
        #     5,
        #     lambda channels: AttnMixer(channels),
        #     lambda channels: nn.LayerNorm((n_frames, channels)),
        #     return_features=False)

        self.transform_latent = LinearOutputStack(c, 3)
        self.embed_pos = LinearOutputStack(c, 2, in_channels=33)

        self.to_synth = LinearOutputStack(c, 3, out_channels=n_atoms * 70)
        self.to_rooms = LinearOutputStack(c, 3, out_channels=n_rooms)
        self.to_verb_mix = LinearOutputStack(c, 3, out_channels=1)

        self.transform = nn.Sequential(
            nn.Conv1d(c, c, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 7, 1, 3),
            nn.LeakyReLU(0.2),
        )

        self.atoms = AudioEvent(
            sequence_length=sequence_length,
            n_samples=n_samples,
            n_events=n_events,
            min_f0=50,
            max_f0=8000,
            n_harmonics=n_harmonics,
            sr=samplerate,
            noise_ws=512,
            noise_step=256)

        # self.atom_gen = AtomGenerator()
        # self.n_atoms = n_atoms
        # self.ln = LinearOutputStack(128, 3, out_channels=128 * n_atoms)
        # # self.baseline = LinearOutputStack(128, 3, out_channels=n_events)
        # n_rooms = 8
        # self.to_room = LinearOutputStack(128, 3, out_channels=n_rooms)
        # self.to_mix = LinearOutputStack(128, 2, out_channels=1)

        self.verb = NeuralReverb(n_samples, n_rooms)

    def forward(self, z, indices, norms):

        # initial shape assertions
        z = z.view(-1, latent_dim)
        z = self.transform_latent(z)

        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        z = z.view(-1, latent_dim, 1).repeat(1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        x = z + embedded + amp
        pos = pos_encoded(z.shape[0], n_frames, 16, device=device)
        pos = self.embed_pos(pos)
        x = x.permute(0, 2, 1)
        x = x + pos

        x = x.permute(0, 2, 1)
        x = self.transform(x)
        x = x.permute(0, 2, 1)

        synth = self.to_synth(x)
        r = self.to_rooms(x.mean(dim=1))
        m = self.to_verb_mix(x.mean(dim=1))

        rooms = torch.softmax(r, dim=-1)
        mix = torch.sigmoid(m)

        x = synth.permute(0, 2, 1).reshape(-1, self.n_atoms, 70, 64)

        f0 = x[:, :, 0, :] ** 2
        osc_env = x[:, :, 1, :] ** 2
        noise_env = x[:, :, 2, :] ** 2
        overall_env = x[:, :, 3, :]
        noise_std = x[:, :, 4, :]
        harm_env = x[:, :, 5:-1, :]

        signal = self.atoms.forward(
            f0,
            overall_env,
            osc_env,
            noise_env,
            harm_env,
            noise_std,
        )
        dry = signal.mean(dim=1, keepdim=True)

        wet = self.verb.forward(dry, rooms)

        signal = ((1 - mix)[:, None, :] * dry) + (wet * mix[:, None, :])
        return signal


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


gen = Generator(channels=latent_dim, fft=True).to(device)
gen_optim = optimizer(gen, lr=1e-4)

disc = Discriminator(latent_dim).to(device)
disc_optim = optimizer(disc)


def train_gen(batch, indices, norms):
    gen_optim.zero_grad(set_to_none=True)
    z = get_latent(batch.shape[0], latent_dim)
    fake = gen.forward(z, indices, norms)

    # fj, f_feat = disc.forward(fake, indices, norms)
    # rj, r_feat = disc.forward(batch, indices, norms)

    # feat_loss = F.mse_loss(f_feat, r_feat)

    real_feat = perceptual_features(batch)
    fake_feat = perceptual_features(fake)
    feat_loss = F.mse_loss(real_feat, fake_feat)
    

    # judge_loss = least_squares_generator_loss(fj)

    loss = feat_loss
    loss.backward()
    gen_optim.step()
    # print('G', loss.item())
    return fake, loss


def train_disc(batch, indices, norms):
    disc_optim.zero_grad(set_to_none=True)
    z = get_latent(batch.shape[0], latent_dim)
    fake = gen.forward(z, indices, norms)
    fj, _ = disc.forward(fake, indices, norms)
    rj, _ = disc.forward(batch, indices, norms)
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    # print('D', loss.item())
    return loss


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

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices
            # decoded = decode_batch(self.kmeans, indices, norms)
            # self.recon = decoded

            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            norms = norms[:small_batch].to(device)
            item = item[:small_batch].view(-1, 1, n_samples)

            self.fake, loss = train_gen(item, indices, norms)
            if i % 10 == 0:
                print('G', i, loss.item())

            # if i % 2 == 0:
            #     self.fake, loss = train_gen(item, indices, norms)
            #     if i % 10 == 0:
            #         print('G', i, loss.item())
            # else:
            #     loss = train_disc(item, indices, norms)
            #     if (i + 1) % 10 == 0:
            #         print('D', i, loss.item())
