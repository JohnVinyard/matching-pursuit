import torch
from torch import nn
from torch.optim import Adam

from conjure import LmdbCollection, serve_conjure, loggers, SupportedContentType, NumpySerializer, NumpyDeserializer
from data import AudioIterator
from modules import stft, flattened_multiband_spectrogram
from modules.anticausal import AntiCausalAnalysis
from modules.infoloss import CorrelationLoss
from modules.transfer import fft_convolve, freq_domain_transfer_function_to_resonance
from modules.upsample import upsample_with_holes
from util import device, encode_audio, make_initializer

# the size, in samples of the audio segment we'll overfit
n_samples = 2 ** 15

# the samplerate, in hz, of the audio signal
samplerate = 22050

# derived, the total number of seconds of audio
n_seconds = n_samples / samplerate

transform_window_size = 2048
transform_step_size = 256

n_frames = n_samples // transform_step_size

initializer = make_initializer(0.05)




def transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, n_frames, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


def loss_transform(x: torch.Tensor) -> torch.Tensor:
    return flattened_multiband_spectrogram(
        x,
        stft_spec={
            'long': (128, 64),
            'short': (64, 32),
            'xs': (16, 8),
        },
        smallest_band_size=512)



def reconstruction_loss(target: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
    t = loss_transform(target)
    r = loss_transform(recon)
    return torch.abs(t - r).sum()


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        channels = 32
        sparse_channels = 256

        self.analyze = AntiCausalAnalysis(1024, channels, 2, [1, 2, 4, 8, 16, 32, 64, 1], do_norm=True)
        self.proj_sparse = nn.Conv1d(channels, sparse_channels, 1, 1, 0)
        self.proj_dense = nn.Conv1d(sparse_channels, channels, 1, 1, 0)
        # self.synthesize = AntiCausalAnalysis(channels, channels, 2, [1, 2, 4, 8, 16, 32, 64, 1], do_norm=True, reverse_causality=True)

        window_size = 2048
        self.window_size = window_size
        n_coeffs = window_size // 2 + 1
        self.n_coeffs = n_coeffs

        self.resonances = nn.Parameter(torch.zeros(1, channels, n_coeffs).uniform_(0, 1))

        self.apply(initializer)


    def forward(self, audio: torch.Tensor):
        batch, channels, time = audio.shape

        spec = transform(audio)

        x = self.analyze(spec)
        x = self.proj_sparse(x)
        sparse = x = torch.relu(x)

        x = self.proj_dense(x)
        # x = self.synthesize(x)
        x = upsample_with_holes(x, desired_size=n_samples)

        res = freq_domain_transfer_function_to_resonance(self.window_size, torch.clamp(self.resonances, 0, 0.9999), n_frames=n_frames, apply_decay=True)[..., :n_samples].view(batch, -1, time)
        x = fft_convolve(x, res)
        x = torch.sum(x, dim=1, keepdim=True)
        return x, sparse


def to_numpy(x: torch.Tensor):
    return x.data.cpu().numpy()


def train_and_monitor(batch_size: int = 8):
    stream = AudioIterator(
        batch_size=batch_size,
        n_samples=n_samples,
        samplerate=samplerate,
        normalize=True)

    collection_name = 'sparse-streaming'
    collection = LmdbCollection(path=collection_name)

    collection.destroy()

    collection = LmdbCollection(path=collection_name)

    recon_audio, orig_audio = loggers(
        ['recon', 'orig'],
        'audio/wav',
        encode_audio,
        collection)

    envelopes, = loggers(
        ['envelopes'],
        SupportedContentType.Spectrogram.value,
        to_numpy,
        collection,
        serializer=NumpySerializer(),
        deserializer=NumpyDeserializer())

    serve_conjure([
        orig_audio,
        recon_audio,
        envelopes,
    ], port=9999, n_workers=1)



    def train():
        model = Model().to(device)
        loss_model = CorrelationLoss(n_elements=512).to(device)
        optim = Adam(model.parameters(), lr=1e-3)


        for i, item in enumerate(iter(stream)):
            optim.zero_grad()

            target = item.view(batch_size, 1, n_samples).to(device)
            orig_audio(target)
            recon, sparse = model.forward(target)
            recon_audio(recon)

            envelopes(sparse[0])

            non_zero = (sparse > 0).sum()
            sparsity = (non_zero / sparse.numel()).item()

            sparsity_loss = torch.abs(sparse).sum() * 0

            loss = reconstruction_loss(target, recon) + sparsity_loss
            # loss = loss_model.noise_loss(target, recon) + sparsity_loss

            loss.backward()
            optim.step()
            print(i, loss.item(), non_zero, sparsity)


    train()


if __name__ == '__main__':
    train_and_monitor(batch_size=1)
