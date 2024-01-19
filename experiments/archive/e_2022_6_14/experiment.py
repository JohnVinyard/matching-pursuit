
from sklearn.cluster import MiniBatchKMeans
import torch
from torch import nn
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from upsample import FFTUpsampleBlock
from util.readmedocs import readme
import zounds
import numpy as np
from torch.nn import functional as F
from util import device, playable
from util.weight_init import make_initializer


model_dim = 128
samplerate = zounds.SR22050()
n_samples = 2**14

n_clusters = 512
n_frames = n_samples // 256


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
sequence_length = n_frames

init_weights = make_initializer(0.1)

n_steps = 25
means = [0] * n_steps
stds = [0.5] * n_steps

batch_size = 16

pos_embeddings = pos_encoded(batch_size, n_steps, 16, device=device)

def forward_process(audio, n_steps):
    degraded = audio
    for i in range(n_steps):
        noise = torch.zeros_like(audio).normal_(means[i], stds[i]).to(device)
        degraded = degraded + noise
    return audio, degraded, noise


def reverse_process(model, indices, norms):
    degraded = torch.zeros(
        indices.shape[0], 1, n_samples).normal_(0, 1.6).to(device)

    for i in range(n_steps - 1, -1, -1):
        pred_noise = model.forward(
            degraded, indices, norms, pos_embeddings[:indices.shape[0], i, :])

        degraded = degraded - pred_noise

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
        

        self.down = nn.Sequential(
            nn.Conv1d(1, 16, 25, 4, 12), # 4096
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 25, 4, 12), # 1024
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 25, 4, 12), # 256
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 25, 4, 12), # 64
        )
        self.reduce = nn.Conv1d(model_dim * 3, model_dim, 1, 1, 0)
        self.net = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
            DilatedBlock(model_dim, 1),
        )

        self.up = nn.Sequential(
            
            # nn.ConvTranspose1d(model_dim, 64, 8, 4, 2), # 256
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(64, 32, 8, 4, 2), # 1024
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(32, 16, 8, 4, 2), # 4096
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(16, 8, 8, 4, 2), # 16384


            nn.Conv1d(model_dim, 64, 25, 1, 12),
            FFTUpsampleBlock(64, 64, 64, factor=4, infer=True),
            nn.LeakyReLU(0.2),

            nn.Conv1d(64, 32, 25, 1, 12),
            FFTUpsampleBlock(32, 32, 256, factor=4, infer=True),
            nn.LeakyReLU(0.2),

            nn.Conv1d(32, 16, 25, 1, 12),
            FFTUpsampleBlock(16, 16, 1024, factor=4, infer=True),
            nn.LeakyReLU(0.2),

            nn.Conv1d(16, 8, 25, 1, 12),
            FFTUpsampleBlock(8, 8, 4096, factor=4, infer=True),
            nn.LeakyReLU(0.2),

            nn.Conv1d(8, 1, 25, 1, 12),
        )
        
        
        self.embed_step = LinearOutputStack(model_dim, 3, in_channels=33)
        self.apply(init_weights)

    def forward(self, degraded, indices, norms, step):
        orig = degraded

        degraded = self.down(degraded)
        step = self.embed_step(step).view(indices.shape[0], model_dim, 1).repeat(1, 1, n_frames)
        x = self.cond.forward(indices, norms)
        z = torch.cat([degraded, step, x], dim=1)
        z = self.reduce(z)

        z = self.net(z)

        y = self.up(z)

        # y = z
        # for layer in self.up:
        #     y = layer(y)
        #     print(y.shape)

        final = orig + y
        return final


model = Generator().to(device)
optim = optimizer(model, lr=1e-3)


def train_gen(batch, indices, norms):
    optim.zero_grad()

    step = np.random.randint(1, n_steps)
    audio, degraded, noise = forward_process(batch, step)

    pred_noise = model.forward(degraded, indices, norms, pos_embeddings[:, step, :])

    loss = F.mse_loss(pred_noise, noise)

    loss.backward()
    optim.step()
    return loss


@readme
class ConditionalDiffusionModel(object):
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
        return playable(self.denoise(), samplerate)

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
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)

            small_batch = 16

            indices = torch.from_numpy(indices).long()[:small_batch].to(device)
            norms = norms[:small_batch].to(device)
            real = item[:small_batch].view(-1, 1, n_samples)

            self.indices = indices
            self.norms = norms

            self.real = real

            gen_loss = train_gen(real, indices, norms)

            if i > 0 and i % 10 == 0:
                print('GEN', i, gen_loss.item())
            
