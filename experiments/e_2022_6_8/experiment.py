from sklearn.cluster import MiniBatchKMeans
import torch
from torch import nn
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from modules.atoms import unit_norm
from modules.ddsp import NoiseModel, OscillatorBank
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.psychoacoustic import PsychoacousticFeature
from train.optim import optimizer
from util.readmedocs import readme
import zounds
import numpy as np
from torch.nn import functional as F
from util import device, playable
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from util.weight_init import make_initializer

n_clusters = 512
model_dim = 128
samplerate = zounds.SR22050()
n_samples = 2**14
n_frames = n_samples // 256
band_sizes = [2**i for i in range(14, 8, -1)][::-1]


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
sequence_length = n_frames

init_weights = make_initializer(0.05)

feature = PsychoacousticFeature(
    kernel_sizes=[128] * 6
).to(device)

batch_size = 8


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


class StepAndConditioning(nn.Module):
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


class DenoisingLayer(nn.Module):
    def __init__(self, dim, band_size, dilations, to_audio=False, is_disc=False):
        super().__init__()

        self.is_disc = is_disc

        self.net = nn.Sequential(
            *[DilatedBlock(dim, d) for d in dilations]
        )
        self.to_audio = to_audio

        if band_size == 512:
            lowest_freq = 0.001
        else:
            lowest_freq = 0.5

        n_noise_frames = max(n_frames, band_size // 32)
        n_coeffs = (band_size // n_noise_frames) + 1

        if band_size == 512:
            mask_after = 1
        else:
            mask_after = (n_coeffs // 2)
        
        if self.is_disc:
            self.pred_feat = nn.Conv1d(dim, dim, 1, 1, 0)
            # self.amp_pred = nn.Conv1d(dim, 1, 1, 1, 0)
            # self.frame_pred = nn.Conv1d(dim, n_clusters, 1, 1, 0)

        if self.to_audio:

            self.osc = OscillatorBank(
                dim,
                32,
                band_size,
                constrain=True,
                lowest_freq=lowest_freq,
                complex_valued=True)
            self.noise = NoiseModel(
                input_channels=dim,
                channels=dim,
                input_size=n_frames,
                n_noise_frames=n_noise_frames,
                n_audio_samples=band_size,
                mask_after=mask_after)

    def forward(self, x):

        x = self.net(x)

        if self.is_disc:
            # j = self.judge(x)
            # a = self.amp_pred(x)
            # f = self.frame_pred(x)
            # return j, a, f
            x = self.pred_feat(x)
        elif self.to_audio:
            osc = self.osc(x)
            noise = self.noise(x)
            x = (osc + noise)

        return x


class DenoisingStack(nn.Module):
    def __init__(self, dim, band_size, is_disc=False):
        super().__init__()
        self.is_disc = is_disc
        self.band_size = band_size

        if self.is_disc:
            self.audio = nn.Conv1d(64, dim, 1, 1, 0)
            self.reduce = nn.Conv1d(dim * 2, dim, 1, 1, 0)
            self.step_size = band_size // n_frames
            self.window_size = self.step_size * 2

        self.net = nn.Sequential(
            DenoisingLayer(dim, band_size,
                           [1, 3, 9], to_audio=False),
            DenoisingLayer(dim, band_size,
                           [1, 3, 9], to_audio=False),
            DenoisingLayer(dim, band_size,
                           [1, 3, 9], to_audio=not is_disc, is_disc=is_disc),
        )

    def forward(self, x, audio=None):
        if self.is_disc and audio is not None:
            a = feature.banks[self.band_size][0].forward(audio, normalize=False)
            a = feature.banks[self.band_size][0].temporal_pooling(a, self.window_size, self.step_size)[..., :n_frames]
            a = self.audio(a)
            x = torch.cat([a, x], dim=1)
            x = self.reduce(x)

        x = self.net(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self, dim, is_disc=False):
        super().__init__()
        self.cond = StepAndConditioning(n_clusters, dim)
        self.is_disc = is_disc

        self.stacks = nn.ModuleDict(
            {str(bs): DenoisingStack(dim, bs, is_disc=is_disc) for bs in band_sizes})
        
        if self.is_disc:
            self.net = nn.Sequential(
                nn.Conv1d(dim * len(self.stacks), dim, 1, 1, 0),
                nn.LeakyReLU(0.2),
                nn.Conv1d(dim, dim, 3, 1, 1)
            )
            self.judge = nn.Conv1d(dim, 2, 1, 1, 0)
            self.amp = nn.Conv1d(dim, 1, 1, 1, 0)
            self.frame = nn.Conv1d(dim, n_clusters, 1, 1, 0)
        
        self.apply(init_weights)

    def forward(self, indices, norms, audio=None):

        x = self.cond(indices, norms)

        results = {str(bs): self.stacks[str(bs)].forward(x, audio=audio[bs] if audio else None)
                   for bs in band_sizes}
        
        if self.is_disc:
            x = torch.cat(list(results.values()), dim=1)
            x = self.net(x)
            j = self.judge(x)
            a = self.amp(x)
            f = self.frame(x)
            return j, a, f, x
        return results


model = AutoEncoder(model_dim).to(device)
optim = optimizer(model, lr=1e-3)

disc = AutoEncoder(model_dim, is_disc=True).to(device)
disc_optim = optimizer(disc, lr=1e-3)

def train_gen(batch, indices, norms):
    optim.zero_grad()

    recon = model.forward(indices, norms)
    recon = {int(k): v for k, v in recon.items()}

    batch = fft_frequency_decompose(batch, band_sizes[0])
    j, a, f, fx = disc.forward(indices, norms, recon)

    adv_loss = F.cross_entropy(
        j.permute(0, 2, 1).reshape(-1, 2), 
        torch.zeros(indices.shape[0] * n_frames).fill_(1).to(device).long())

    amp_loss = F.mse_loss(a.view(-1, n_frames), norms.view(-1, n_frames))

    frame_loss = F.cross_entropy(f.permute(0, 2, 1).reshape(-1, n_clusters), indices.view(-1))

    loss = (adv_loss * 0) + (amp_loss * 0.1) + frame_loss
    
    loss.backward()
    optim.step()
    return loss, recon

def train_disc(batch, indices, norms):
    disc_optim.zero_grad()

    recon = model.forward(indices, norms)
    recon = {int(k): v for k, v in recon.items()}

    fj, fa, ff, fx = disc.forward(indices, norms, recon)
    rj, ra, rf, rx = disc.forward(indices, norms, fft_frequency_decompose(batch, band_sizes[0]))

    adv_loss = F.cross_entropy(
        fj.permute(0, 2, 1).reshape(-1, 2), 
        torch.zeros(indices.shape[0] * n_frames).fill_(0).to(device).long()) + F.cross_entropy(
        rj.permute(0, 2, 1).reshape(-1, 2), 
        torch.zeros(indices.shape[0] * n_frames).fill_(1).to(device).long())

    # amp loss
    amp_loss = F.mse_loss(ra.view(-1, n_frames), norms.view(-1, n_frames))

    # frame loss
    frame_loss = F.cross_entropy(rf.permute(0, 2, 1).reshape(-1, n_clusters), indices.view(-1))

    loss = (adv_loss * 0) + (amp_loss * 0.1) + frame_loss
    
    loss.backward()
    disc_optim.step()
    return loss


# def train_model(batch, indices, norms):
#     optim.zero_grad()

#     recon = model.forward(indices, norms)
#     recon = {int(k): v for k, v in recon.items()}

#     real = feature.compute_feature_dict(batch)
#     fake = feature.compute_feature_dict(recon)

#     loss = 0
#     for k, v in fake.items():
#         loss = loss + F.mse_loss(v, real[int(k)])

#     loss.backward()
#     optim.step()

#     return loss, recon


@readme
class MultibandAutoEncoder(object):
    def __init__(self, stream):
        super().__init__()
        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None
        self.stream = stream

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
        x = fft_frequency_recompose(self.fake, n_samples)
        return playable(x, samplerate)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)

            small_batch = 4

            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            norms = norms[:small_batch].to(device)
            real = item[:small_batch].view(-1, 1, n_samples)

            self.indices = indices
            self.norms = norms

            self.real = real

            # gen_loss, self.fake = train_model(real, indices, norms)

            if i % 2 == 0:
                gen_loss, self.fake = train_gen(real, indices, norms)
            else:
                disc_loss = train_disc(real, indices, norms)

            if i > 0 and i % 10 == 0:
                # print('LOSS', i, gen_loss.item())
                print('GEN', i, gen_loss.item())
                print('DISC', i, disc_loss.item())
