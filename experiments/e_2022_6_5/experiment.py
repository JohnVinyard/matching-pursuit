from sklearn.cluster import MiniBatchKMeans
from sympy import DenseNDimArray
import torch
from torch import nn
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from train.optim import optimizer
from util.readmedocs import readme
import zounds
import numpy as np
from modules import pos_encoded
from torch.nn import functional as F
from util import device, playable
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from util.weight_init import make_initializer

n_clusters = 512
model_dim = 64
samplerate = zounds.SR22050()
n_samples = 2**14
n_frames = n_samples // 256
band_sizes = [2**i for i in range(14, 8, -1)][::-1]
n_steps = 25


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
sequence_length = n_frames

init_weights = make_initializer(0.1)

n_steps = 25
means = [0] * n_steps
stds = [0.5] * n_steps

batch_size = 8


pos_embeddings = pos_encoded(batch_size, n_steps, 16, device=device)


def forward_process(audio, n_steps):
    degraded = audio
    for i in range(n_steps):
        noise = torch.zeros_like(audio).normal_(means[i], stds[i]).to(device)
        degraded = degraded + noise
    return audio, degraded, noise


def reverse_process(model, indices, norms):
    degraded = torch.zeros(indices.shape[0], 1, n_samples).normal_(0, 1.6).to(device)
    degraded = fft_frequency_decompose(degraded, band_sizes[0])

    for i in range(n_steps - 1, -1, -1):
        pred_noise = model.forward(
            degraded, indices, norms, i)
        
        for k, v in pred_noise.items():
            degraded[k] = degraded[k] - v
    
    degraded = fft_frequency_recompose(degraded, n_samples)
    return degraded

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
    def __init__(self, n_clusters, dim, upsample_size):
        super().__init__()
        self.upsample_size = upsample_size
        self.embedding = EmbedClusterCenters(n_clusters, dim)
        self.amp = EmbedAmp(dim)
        self.reduce = LinearOutputStack(
            dim, 2, out_channels=dim, in_channels=dim*2)

    def forward(self, indices, norms):
        indices = self.embedding(indices).view(-1, n_frames, model_dim)
        norms = self.amp(norms).view(-1, n_frames, model_dim)
        x = torch.cat([indices, norms], dim=-1)
        x = self.reduce(x).permute(0, 2, 1)
        x = F.interpolate(x, size=self.upsample_size, mode='linear')
        x = x.permute(0, 2, 1)
        return x


class StepEmbedding(nn.Module):
    def __init__(self, dim, upsample_size):
        super().__init__()

        self.register_buffer('pos', pos_embeddings[:1, ...])
        self.embed = LinearOutputStack(
            dim, 2, out_channels=dim, in_channels=33)
        self.upsample_size = upsample_size

    def forward(self, x, step):
        pos = self.pos[:, step: step + 1, :]
        pos = self.embed(pos).repeat(x.shape[0], self.upsample_size, 1)
        return pos


class StepAndConditioning(nn.Module):
    def __init__(self, n_clusters, dim, upsample_size):
        super().__init__()
        self.cond = EmbedConditioning(n_clusters, dim, upsample_size)
        self.step = StepEmbedding(dim, upsample_size)
        self.reduce = LinearOutputStack(dim, 2, in_channels=dim * 2)

    def forward(self, degraded, indices, norms, step):
        pos = self.step.forward(degraded, step)
        cond = self.cond.forward(indices, norms)
        x = torch.cat([pos, cond], dim=-1)
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


class DenoisingLayer(nn.Module):
    def __init__(self, n_clusters, dim, band_size, dilations, input_dim=1, to_audio=False):
        super().__init__()
        self.cond = StepAndConditioning(n_clusters, dim, band_size)
        self.embed = nn.Conv1d(input_dim, dim, 25, 1, 12)
        self.net = nn.Sequential(
            nn.Conv1d(dim * 2, dim, 1, 1, 0),
            *[DilatedBlock(dim, d) for d in dilations]
        )
        self.to_audio = to_audio

        if self.to_audio:
            self.audio = nn.Conv1d(dim, 1, 1, 1, 0)

    def forward(self, degraded, indices, norms, step):
        cond = self.cond.forward(degraded, indices, norms, step)
        cond = cond.permute(0, 2, 1)
        x = self.embed(degraded)
        x = torch.cat([cond, x], dim=1)
        x = self.net(x)

        if self.to_audio:
            x = self.audio.forward(x)

        return x


class DenoisingStack(nn.Module):
    def __init__(self, n_clusters, dim, band_size):
        super().__init__()
        self.net = nn.ModuleList([
            DenoisingLayer(n_clusters, dim, band_size, [1, 3, 9], input_dim=1),
            DenoisingLayer(n_clusters, dim, band_size,
                           [1, 3, 9], input_dim=dim),
            DenoisingLayer(n_clusters, dim, band_size,
                           [1, 3, 9], input_dim=dim, to_audio=True),
        ])

    def forward(self, degraded, indices, norms, step):
        x = degraded
        for layer in self.net:
            x = layer.forward(x, indices, norms, step)
        return x


class DiffusionModel(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.stacks = nn.ModuleDict(
            {str(bs): DenoisingStack(n_clusters, dim, bs) for bs in band_sizes})

    def forward(self, degraded, indices, norms, step):

        if isinstance(degraded, dict):
            bands = degraded
        else:
            bands = fft_frequency_decompose(degraded, band_sizes[0])

        results = {str(bs): self.stacks[str(bs)].forward(
            bands[bs], indices, norms, step) for bs in bands.keys()}
        return results


model = DiffusionModel(model_dim)
optim = optimizer(model)


def train_model(batch, indices, norms):
    optim.zero_grad()
    step = np.random.randint(1, n_steps)
    orig, degraded, noise = forward_process(batch, step)

    pred_noise = model.forward(degraded, indices, norms, step)
    noise = fft_frequency_decompose(noise, band_sizes[0])

    loss = 0
    for k, v in pred_noise.items():
        loss = loss + torch.abs(v - noise[int(k)]).sum()
    
    loss.backward()
    optim.step()
    return loss


@readme
class MultibandDiffusionModel(object):
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
        return np.abs(zounds.spectral.stft(self.listen()))

    
    def check_degraded(self):
        with torch.no_grad():
            audio, degraded, noise = forward_process(self.real, n_steps)
            return playable(degraded, samplerate)

    def denoise(self):
        with torch.no_grad():
            result = reverse_process(model, self.indices[:1], self.norms[:1])
            return playable(result, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices

            small_batch = 4

            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            norms = norms[:small_batch].to(device)
            real = item[:small_batch].view(-1, 1, n_samples)

            self.real = real

            gen_loss = train_model(real, indices, norms)

            if i > 0 and i % 10 == 0:
                print('LOSS', i, gen_loss.item())
