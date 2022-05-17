from sklearn.cluster import MiniBatchKMeans
from sympy import Line
import zounds
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.pif import AuditoryImage
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from util.weight_init import make_initializer
import numpy as np
from torch import nn
import torch

from torch.nn import functional as F

n_samples = 2**17
samplerate = zounds.SR22050()

sequence_length = n_samples // 256
n_clusters = 512
small_batch = 4


scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256

n_events = 8


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


class MusicGenerator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.embedding = nn.Embedding(n_clusters, self.channels)

        self.amp = nn.Sequential(
            nn.Conv1d(1, 16, 7, 1, 3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, self.channels, 3, 1, 1)
        )

        self.pos = LinearOutputStack(self.channels, 2, in_channels=33)
        self.encode = LinearOutputStack(
            self.channels, 3, in_channels=channels * 3)

        layer = nn.TransformerEncoderLayer(
            self.channels, nhead=4,
            dim_feedforward=self.channels,
            batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)

        self._mask = None

        self.to_frame = LinearOutputStack(
            self.channels, 3, out_channels=n_clusters)
        self.to_amp = LinearOutputStack(self.channels, 3, out_channels=1)

        self.apply(init_weights)

    def forward(self, indices, norms):
        indices = indices.view(indices.shape[0], -1)
        indices = self.embedding(indices).permute(0, 2, 1)

        norms = norms.permute(0, 2, 1)
        norms = self.amp(norms)

        pos = pos_encoded(indices.shape[0], sequence_length, 16, device)
        pos = self.pos(pos).permute(0, 2, 1)

        x = torch.cat([indices, norms, pos], dim=1).permute(0, 2, 1)
        x = self.encode(x)

        if self._mask is None:
            self._mask = torch.triu(torch.full(
                (sequence_length, sequence_length), float('-inf')), diagonal=1)

        x = self.transformer.forward(x, self._mask)

        indices = self.to_frame(x)
        amp = torch.relu(self.to_amp(x))

        return indices, amp


gen = MusicGenerator(128)
optim = optimizer(gen, lr=1e-3)


def train(indices, norms):
    optim.zero_grad()

    batch = indices.shape[0]

    pred_indices, pred_norms = gen.forward(indices, norms)

    cf = 7

    real_norms = norms[:, -cf:, :]
    real_indices = indices[:, -cf:, :].permute(0, 2, 1)

    pred_norms = pred_norms[:, -cf:, :]
    pred_indices = pred_indices[:, -cf:, :].permute(0, 2, 1)

    amp_loss = F.mse_loss(pred_norms, real_norms)
    frame_loss = F.cross_entropy(pred_indices, real_indices.view(batch, -1))

    total_loss = amp_loss + frame_loss
    total_loss.backward()
    optim.step()

    return total_loss, pred_indices, pred_norms


@readme
class VQTransformerExperiment(object):
    def __init__(self, stream):
        super().__init__()

        self.stream = stream
        self.indices = None
        self.norms = None

        self.pred_indices = None
        self.pred_norms = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.gen = gen
    
    def view_indices(self):
        with torch.no_grad():
            indices = torch.softmax(self.pred_indices, dim=1)
            return indices.data.cpu().numpy()[0]
    
    def view_norms(self):
        with torch.no_grad():
            return self.pred_norms.data.cpu().numpy().squeeze()

    def run(self):
        for i, item in enumerate(self.stream):
            spec, norms = to_frames(item)
            self.norms = norms
            self.spec = spec

            update_kmeans(i, self.kmeans, spec)

            indices = encode_batch(self.kmeans, spec)
            self.indices = indices

            indices = torch.from_numpy(indices).long().to(device)
            norms = norms.to(device)

            loss, self.pred_indices, self.pred_norms = train(indices, norms)

            if i > 0 and i % 10 == 0:
                print(loss.item())
