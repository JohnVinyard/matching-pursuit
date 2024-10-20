import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, loggers
from data import AudioIterator
from modules import stft, sparsify, sparsify_vectors, iterative_loss
from modules.anticausal import AntiCausalAnalysis
from util import device, encode_audio

# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 18

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256
n_events = 16
context_dim = 32

n_frames = n_samples // transform_step_size


def transform(x: torch.Tensor):
    batch_size = x.shape[0]

    x = stft(x, 2048, 256, pad=True).view(
        batch_size, n_frames, 1025)[..., :1024].permute(0, 2, 1)
    return x


class Model(nn.Module):
    def __init__(
            self,
            in_channels: int = 1024,
            hidden_channels: int = 256):

        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.encoder = AntiCausalAnalysis(
            in_channels=in_channels,
            channels=hidden_channels,
            kernel_size=2,
            dilations=[1, 2, 4, 8, 16, 32, 64, 1],
            pos_encodings=True)
        self.to_event_vectors = nn.Conv1d(hidden_channels, context_dim, 1, 1, 0)
        self.to_event_switch = nn.Conv1d(hidden_channels, 1, 1, 1, 0)

    def encode(self, transformed: torch.Tensor):
        batch_size = transformed.shape[0]

        if transformed.shape[1] == 1:
            transformed = transform(transformed)

        x = transformed

        encoded = self.encoder.forward(x)

        event_vecs = self.to_event_vectors(encoded).permute(0, 2, 1)  # batch, time, channels

        event_switch = self.to_event_switch(encoded)
        attn = torch.relu(event_switch).permute(0, 2, 1).view(batch_size, 1, -1)

        attn, attn_indices, values = sparsify(attn, n_to_keep=n_events, return_indices=True)

        vecs, indices = sparsify_vectors(event_vecs.permute(0, 2, 1), attn, n_to_keep=n_events)

        scheduling = torch.zeros(batch_size, n_events, encoded.shape[-1], device=encoded.device)
        for b in range(batch_size):
            for j in range(n_events):
                index = indices[b, j]
                scheduling[b, j, index] = attn[b, 0][index]

        return vecs, scheduling

    def generate(self, vecs: torch.Tensor, scheduling: torch.Tensor):
        pass

    def iterative(self, audio: torch.Tensor):
        channels = []
        schedules = []
        vecs = []

        spec = transform(audio)

        for i in range(n_events):
            v, sched = self.encode(spec)
            vecs.append(v)
            schedules.append(sched)
            ch, _, _, _ = self.generate(v, sched)
            current = transform(ch)
            spec = (spec - current).clone().detach()
            channels.append(ch)

        channels = torch.cat(channels, dim=1)
        vecs = torch.cat(vecs, dim=1)
        schedules = torch.cat(schedules, dim=1)

        return channels, vecs, schedules

    def forward(self, audio: torch.Tensor):
        raise NotImplementedError()




def train_and_monitor(batch_size: int = 8):
    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True)


    collection = LmdbCollection(path='iterativedecomposition')

    recon_audio, orig_audio, random_audio = loggers(
        ['recon', 'orig', 'random'],
        'audio/wav',
        encode_audio,
        collection)

    serve_conjure([
        orig_audio,
        recon_audio,
        random_audio,
    ], port=9999, n_workers=1)

    def train():
        model = Model(in_channels=1024, hidden_channels=256).to(device)
        optim = Adam(model.parameters(), lr=1e-3)


        for i, item in enumerate(iter(stream)):
            optim.zero_grad()
            target = item.view(batch_size, 1, n_samples).to(device)
            recon, encoded, scheduling = model.iterative(target)
            loss = iterative_loss(target, recon, transform)
            loss.backward()
            optim.step()
            print(i, loss.item())

    train()


if __name__ == '__main__':
    train_and_monitor()

