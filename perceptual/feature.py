from torch import nn
from scipy.signal import gammatone
import zounds
import numpy as np
import torch
from torch.nn import functional as F



class CochleaModel(nn.Module):
    def __init__(
            self,
            samplerate: zounds.SampleRate,
            scale: zounds.FrequencyScale,
            kernel_size: int,
            phase_locking_cutoff_hz: int = 5000):

        super().__init__()
        self.samplerate = samplerate
        self.scale = scale
        self.kernel_size = kernel_size
        self.phase_locking_cutoff_hz = phase_locking_cutoff_hz
        self.phase_locking_kernel_size = int(
            samplerate.nyquist / phase_locking_cutoff_hz)

        filters = np.stack([
            gammatone(
                freq=x,
                ftype='fir',
                order=4,
                numtaps=kernel_size,
                fs=int(samplerate))[0] for x in scale.center_frequencies
        ], axis=0)

        self.register_buffer(
            'filters',
            torch.from_numpy(filters).float().view(len(scale), 1, kernel_size))

    def forward(self, x):
        x = x.view(x.shape[0], 1, -1)
        n_samples = x.shape[-1]

        x = F.conv1d(
            x, self.filters, padding=self.kernel_size // 2)[..., :n_samples]

        # half-wave rectification
        x = F.relu(x)

        # compression
        x = torch.sqrt(x)

        # loss of phase-locking above 5khz
        x = F.avg_pool1d(
            x,
            self.phase_locking_kernel_size,
            stride=1,
            padding=self.phase_locking_kernel_size // 2)[..., :n_samples]
        return x


class NormalizedSpectrogram(nn.Module):
    def __init__(self, pool_window, n_bins, loudness_gradations, embedding_dim, out_channels):
        super().__init__()
        self.pool_window = pool_window
        self.n_bins = n_bins
        self.embedding_dim = embedding_dim
        self.loudness_gradations = loudness_gradations
        self.out_channels = out_channels

        self.loudness_embedding = nn.Embedding(loudness_gradations, embedding_dim)
        self.reduce = nn.Conv1d(n_bins + embedding_dim, out_channels, 1, 1, 0)
    
    def forward(self, x):
        x = x.view(x.shape[0], self.n_bins, -1)
        n_frames = x.shape[-1] // (self.pool_window // 2)

        
        x = F.avg_pool1d(
            x, 
            self.pool_window, 
            stride=self.pool_window // 2, 
            padding=self.pool_window // 2)[..., :n_frames]
        
        # we'd like the max norm in each window to be 1
        norms = torch.norm(x, dim=1)
        norms = norms / (norms.max(dim=1, keepdim=True)[0] + 1e-8)
        x = x / (norms[:, None, :] + 1e-8)

        embeddings = (norms * self.loudness_gradations * 0.9999).long()
        embeddings = self.loudness_embedding(embeddings)

        x = torch.cat([x, embeddings.permute(0, 2, 1)], dim=1)
        x = self.reduce(x)
        return x
