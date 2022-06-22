import pickle
import torch
from experiments.e_2022_6_15.experiment import Generator
import zounds
from modules.linear import LinearOutputStack
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device, playable
from torch import nn
from util.readmedocs import readme

from util.weight_init import make_initializer
from torch.nn import functional as F

samplerate = zounds.SR22050()
n_clusters = 512
n_samples = 2 ** 15
step_size = 256
n_freq_bins = 256

scale = MelScale()
codec = AudioCodec(scale)

sequence_length = n_samples // step_size

init_weights = make_initializer(0.1)

with open('kmeans.dat', 'rb') as f:
    kmeans = pickle.load(f)

gen = Generator(kmeans).to(device).eval()
gen.load_state_dict(torch.load('gen.dat'))


def to_frames(batch):
    spec = codec.to_frequency_domain(batch.view(-1, n_samples))[..., 0]
    norms = torch.norm(spec, dim=-1, keepdim=True)
    spec = spec / (norms + 1e-8)
    return spec, norms


def encode_batch(kmeans, frames):
    frames = frames.data.cpu().numpy().reshape(-1, n_freq_bins)
    indices = kmeans.predict(frames)
    return indices.reshape(-1, sequence_length, 1)


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
        seq_len = sequence_length - 1

        indices = indices.view(indices.shape[0], -1)
        indices = self.embedding(indices).permute(0, 2, 1)

        norms = norms.permute(0, 2, 1)
        norms = self.amp(norms)

        pos = pos_encoded(indices.shape[0], seq_len, 16, device)
        pos = self.pos(pos).permute(0, 2, 1)

        x = torch.cat([indices, norms, pos], dim=1).permute(0, 2, 1)
        x = self.encode(x)

        if self._mask is None:
            self._mask = torch.triu(torch.full(
                (seq_len, seq_len), float('-inf')), diagonal=1).to(x.device)

        x = self.transformer.forward(x, self._mask)

        indices = self.to_frame(x)
        amp = torch.relu(self.to_amp(x))

        return indices, amp

music_gen = MusicGenerator(128).to(device)
optim = optimizer(music_gen, lr=1e-4)


def train(indices, norms):
    optim.zero_grad()

    batch = indices.shape[0]

    input_indices = indices[:, :-1]
    input_norms = norms[:, :-1]

    target_indices = indices[:, 1:]
    target_norms = norms[:, 1:]

    pred_indices, pred_norms = music_gen.forward(input_indices, input_norms)

    amp_loss = F.mse_loss(pred_norms, target_norms)
    frame_loss = F.cross_entropy(
        pred_indices.reshape(-1, n_clusters), target_indices.reshape(-1))

    total_loss = amp_loss + frame_loss
    total_loss.backward()
    optim.step()

    return total_loss, pred_indices, pred_norms

@readme
class TokenTransformerExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.indices = None
        self.norms = None
        self.real = None
        self.kmeans = kmeans
    
    def view_indices(self):
        with torch.no_grad():
            indices = torch.softmax(self.pred_indices, dim=1)
            return indices.data.cpu().numpy()[0]
    
    def view_norms(self):
        with torch.no_grad():
            return self.pred_norms.data.cpu().numpy().squeeze()
        
    
    def generate(self, steps=512):
        indices = self.indices[:1]
        norms = self.norms[:1]

        with torch.no_grad():
            for j in range(steps):
                i, n = music_gen.forward(indices[:, j:-1], norms[:, j:-1])

                i = torch.argmax(i, dim=-1, keepdim=True)

                indices = torch.cat([indices, i[:, -1:]], dim=1)
                norms = torch.cat([norms, n[:, -1:]], dim=1)
            
            audio = []
            for j in range(4):
                start = j * 64
                stop = (j + 1) * 64
                a = gen.forward(indices[:, start:stop], norms[:, start:stop])
                audio.append(a)
            
            audio = torch.cat(audio, dim=-1)
            return playable(audio, samplerate)
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            spec, norms = to_frames(item)
            self.spec = spec

            indices = encode_batch(self.kmeans, spec)
            indices = torch.from_numpy(indices).long().to(device)
            norms = norms.to(device)
            real = item

            self.indices = indices
            self.norms = norms
            self.real = real

            loss, self.pred_indices, self.pred_norms = train(indices, norms)

            if i > 0 and i % 10 == 0:
                print(loss.item())