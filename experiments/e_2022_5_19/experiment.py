
from re import M
from modules.diffusion import DiffusionProcess
from util.readmedocs import readme
import numpy as np
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


def perceptual_features(x):
    x = codec.to_frequency_domain(x.squeeze())
    x = x[..., 0]
    return x


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


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
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

        self.embed_step = nn.Linear(1, 128)

        self.embed_audio = nn.Conv1d(1, 128, 25, 1, 12)

        self.embed_context = nn.Conv1d(c * 3, c, 1, 1, 0)

        self.downsample_audio = nn.Sequential(
            nn.Conv1d(c, c, 9, 4, 4), # 4096
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 9, 4, 4), # 1024
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 9, 4, 4), # 256
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 9, 4, 4), # 64
        )



        self.upsample_audio = nn.Sequential(
            nn.Conv1d(c*2, c, 3, 1, 1),

            # context
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 1),

            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(c, c, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(c, c, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(c, c, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(c, c, 8, 4, 2),
        )
        

        self.final = nn.Sequential(
            nn.Conv1d(c*2, c, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 25, 1, 12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, c, 25, 1, 12),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, 1, 25, 1, 12),
        )

        self.apply(init_weights)

    def forward(self, noisy, step, indices, norms):
        # initial shape assertions
        indices = indices.view(-1, n_frames)
        norms = norms.view(-1, 1, n_frames)

        embedded = self\
            .embedding(indices).view(-1, n_frames, self.channels)\
            .permute(0, 2, 1)\
            .view(-1, self.channels, n_frames)

        amp = self.amp(norms)

        audio = self.embed_audio(noisy)
        ds_audio = self.downsample_audio(audio)

        step = self.embed_step(step).view(-1, 128, 1).repeat(1, 1, 64)

        x = torch.cat([embedded, amp, step], dim=1)
        x = self.embed_context(x)

        x = torch.cat([x, ds_audio], dim=1)
        x = self.upsample_audio(x)


        x = torch.cat([x, audio], dim=1)

        x = self.final(x)

        return x


gen = Generator(channels=latent_dim).to(device)
gen_optim = optimizer(gen, lr=1e-4)


diff = DiffusionProcess(total_steps=25, variance_per_step=0.5)


def train_gen(batch, indices, norms):
    gen_optim.zero_grad()
    steps = diff.get_steps(batch.shape[0], batch.device)
    signal_and_noise, orig_step, noise = diff.forward_process(batch, steps)
    noise_pred = gen.forward(signal_and_noise, orig_step, indices, norms)
    loss = torch.abs(noise_pred - noise).sum()
    loss.backward()
    gen_optim.step()
    return loss


@readme
class DiscreteDiffusionGenerator(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None
        self.fake = None

        self.batch = None
        self.torch_indices = None
        self.torch_norms = None

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

    # def listen(self):
    #     return playable(self.fake, samplerate)

    # def fake_spec(self):
    #     return np.abs(zounds.spectral.stft(self.listen()))

    def listen(self):
        x = diff.generate(
            (1, 1, n_samples), 
            gen, 
            device, 
            self.torch_indices[:1, ...], 
            self.torch_norms[:1, ...])
        return playable(x, samplerate)

    def noisy(self):
        with torch.no_grad():
            steps = diff.get_steps(1, device, steps=1)
            x, orig_steps, noise = diff.forward_process(
                self.batch[:1, ...], steps)
            return playable(x, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices
            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            self.torch_indices = indices
            norms = norms[:small_batch].to(device)
            self.torch_norms = norms
            real = item[:small_batch].view(-1, 1, n_samples)

            self.batch = real

            gen_loss = train_gen(real, indices, norms)

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
