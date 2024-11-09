import torch
from torch import nn
from torch.optim import Adam
from conjure import loggers, serve_conjure, LmdbCollection
from conjure.logger import encode_audio
from data import get_one_audio_segment
from modules import stft, sparsify, sparsify2, gammatone_filter_bank
from modules.atoms import unit_norm
from modules.overfitraw import OverfitRawAudio
from modules.transfer import fft_convolve
from util import device
from torch.nn import functional as F

n_samples = 2 ** 16
transform_window_size = 2048
transform_step_size = 256


def stft_transform(x: torch.Tensor):
    batch_size = x.shape[0]
    x = stft(x, transform_window_size, transform_step_size, pad=True)
    n_coeffs = transform_window_size // 2 + 1
    x = x.view(batch_size, -1, n_coeffs)[..., :n_coeffs - 1].permute(0, 2, 1)
    return x


class SparseLossFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter_size = 16384
        self.n_filters = 512

        time_proj = torch.zeros(n_samples, 16).uniform_(-1, 1)
        self.register_buffer('time_proj', time_proj)
        channel_proj = torch.zeros(self.n_filters, 16).uniform_(-1, 1)
        self.register_buffer('channel_proj', channel_proj)

        f = gammatone_filter_bank(self.n_filters, self.filter_size, device=device, band_spacing='geometric')

        # f = unit_norm(f, axis=-1)

        # self.filters = nn.Parameter(torch.zeros(self.n_filters, self.filter_size).uniform_(-1, 1))
        self.filters = nn.Parameter(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, time = x.shape
        filters = F.pad(self.filters[None, :, :], (0, n_samples - self.filter_size))
        result = fft_convolve(x, filters)
        result = torch.relu(result)
        orig_shape = result.shape
        result = result.reshape(batch, -1)
        result = F.dropout(result, 0.05)
        result = result / torch.sum(result, dim=-1, keepdim=True)
        result = result.reshape(*orig_shape)
        # result = sparsify(result, n_to_keep=512)
        sparse, times, atoms = sparsify2(result, n_to_keep=512)



        # b = torch.sum(times, dim=1, keepdim=True)
        # c = torch.sum(atoms, dim=1, keepdim=True)

        b = times @ self.time_proj
        c = atoms @ self.channel_proj
        z = torch.cat([b, c], dim=-1)
        print(z.shape)
        return z


class LossFeature(nn.Module):
    def __init__(self):
        super().__init__()

        filter_width = 15
        filter_padding = filter_width // 2

        self.net = nn.Sequential(

            nn.Conv2d(1, 8, (filter_width, filter_width), stride=(1, 1), padding=(filter_padding, filter_padding)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),

            nn.Conv2d(8, 16, (filter_width, filter_width), stride=(1, 1), padding=(filter_padding, filter_padding)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),

            nn.Conv2d(16, 32, (filter_width, filter_width), stride=(1, 1), padding=(filter_padding, filter_padding)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),

            nn.Conv2d(32, 64, (filter_width, filter_width), stride=(1, 1), padding=(filter_padding, filter_padding)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),

            nn.Conv2d(64, 128, (filter_width, filter_width), stride=(1, 1), padding=(filter_padding, filter_padding)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),

            nn.Conv2d(128, 256, (filter_width, filter_width), stride=(1, 1), padding=(filter_padding, filter_padding)),
            nn.ReLU(),
            nn.AvgPool2d((3, 3), stride=(2, 2), padding=(1, 1)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        spec = torch.abs(torch.fft.rfft(x, dim=-1))
        return spec

        # spec = stft_transform(x)
        # spec = spec[:, None, :, :]
        # x = self.net(spec)
        # return x


def train(n_samples: int = 2 ** 16):
    target = get_one_audio_segment(n_samples=n_samples, device=device)
    loss_model = SparseLossFeature().to(device)
    # loss_model = LossFeature().to(device)

    model = OverfitRawAudio(shape=(1, 1, n_samples), std=0.1, normalize=False).to(device)
    optim = Adam(model.parameters(), lr=1e-2)

    collection = LmdbCollection(path='noise')

    recon_audio, orig_audio = loggers(
        ['recon', 'orig', ],
        'audio/wav',
        encode_audio,
        collection)

    orig_audio(target)

    serve_conjure([
        orig_audio,
        recon_audio,
    ], port=9999, n_workers=1)

    orig_audio(target)

    with torch.no_grad():
        original_features = loss_model.forward(target)

    while True:
        optim.zero_grad()
        recon = model.forward(None)
        recon_audio(recon)
        recon_feature = loss_model(recon)
        loss = torch.abs(original_features - recon_feature).sum()
        loss.backward()
        optim.step()
        print(loss.item())


if __name__ == '__main__':
    train(n_samples=n_samples)
