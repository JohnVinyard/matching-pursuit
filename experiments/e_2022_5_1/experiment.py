from util import device
from util.readmedocs import readme
from sklearn.cluster import MiniBatchKMeans
from modules.phase import MelScale, AudioCodec
import zounds
import torch

n_clusters = 512
n_samples = 2 ** 14
samplerate = zounds.SR22050()

scale = MelScale()
codec = AudioCodec(scale)
n_freq_bins = 256
n_frames = n_samples // 256


def to_frames(batch):
    spec = codec.to_frequency_domain(batch)[..., 0]
    norms = torch.norm(spec, dim=-1, keepdim=True)
    spec = spec / (norms + 1e-8)
    return spec, norms

def update_kmeans(i, kmeans, frames):
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

@readme
class TokenExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.spec = None
        self.recon = None
        self.indices = None
        self.norms = None

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    
    def view_spec(self):
        return self.spec.data.cpu().numpy()[0]
    
    def view_recon(self):
        return self.recon.data.cpu().numpy()[0]
    
    def view_indices(self):
        return self.indices[0].squeeze()
    
    def view_norms(self):
        return self.norms.data.cpu().numpy()[0].squeeze()

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