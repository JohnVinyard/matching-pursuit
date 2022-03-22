import zounds
from config.dotenv import Config
import numpy as np
from modules.linear import LinearOutputStack, ResidualBlock
from modules.metaformer import MetaFormer, PoolMixer, AttnMixer
from modules.mixer import Mixer
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from modules.transformer import Transformer
from util import device
import torch
from torch import nn
from torch.optim import Adam
from torch.nn import functional as F

from data.datastore import batch_stream
from util.weight_init import make_initializer

'''
Questions to answer:
- are negative phases meaningful?  Are there other alternatives?
- can I backprop through angle() satisfactorily (understand cosine and sine)
- Can I create an adaptible NERF model (i.e., NERF autoencoder)

Answered:
- Can I learn 1d (YES) or 2d NERF representations of the new spectrogram representation
'''

init_weights = make_initializer(0.14)


def activation(x):
    return F.leaky_relu(x, 0.2)


def binary_pos_encoded(batch_size, time_dim, n_freqs, device):
    n_freqs = (n_freqs * 2) + 1
    pos = torch.zeros(batch_size, time_dim, n_freqs).to(device)
    for i in range(n_freqs, 1, -1):
        pos[:, ::i, i - 1] = 1
    return pos
# class ModulatedLayer(nn.Module):
#     def __init__(self, channels, forward_layers, conditioning_layers):
#         super().__init__()
#         self.f = LinearOutputStack(channels, forward_layers, activation=activation)
#         self.weight = LinearOutputStack(channels, conditioning_layers, activation=activation)
#         self.bias = LinearOutputStack(channels, conditioning_layers, activation=activation)

#     def forward(self, x, conditioning):
#         x = self.f(x)
#         w = self.weight(conditioning)
#         b = self.bias(conditioning)

#         return (x * torch.sigmoid(w)) + b


# class ModulatedStack(nn.Module):
#     def __init__(self, channels, layers, freq_bins, pos_encoder):
#         super().__init__()

#         self.pos_encoder = pos_encoder
#         self.initial = LinearOutputStack(channels, 1, in_channels=33, activation=activation)
#         self.net = nn.Sequential(
#             *[ModulatedLayer(channels, 2, 2) for _ in range(layers)])
#         self.mag = LinearOutputStack(channels, 3, out_channels=freq_bins, activation=activation)
#         self.phase = LinearOutputStack(channels, 3, out_channels=freq_bins, activation=activation)

#     def forward(self, latent):
#         pos = self.pos_encoder(latent.shape[0], 128, 16, device)
#         x = self.initial(pos)

#         for layer in self.net:
#             x = layer(x, latent[:, None, :])

#         mag = self.mag(x)
#         phase = self.phase(x)
#         x = torch.cat([mag[..., None], phase[..., None]], dim=-1)
#         return x


# def binary_pos_encoded(batch_size, time_dim, n_freqs, device):
#     n_freqs = (n_freqs * 2) + 1
#     pos = torch.zeros(batch_size, time_dim, n_freqs).to(device)
#     for i in range(n_freqs, 1, -1):
#         pos[:, ::i, i - 1] = 1
#     return pos


def reduce_mean(x):
    return x.mean(dim=1)


def reduce_select(x, index=-1):
    return x[:, index, :]
def reduce_last(x):
    return x[:, -1, :]


def reduce_max(x):
    _, i = torch.max(torch.abs(x), dim=1, keepdim=True)
    return torch.take(x, i).squeeze()


class Encoder(nn.Module):
    def __init__(self, pos_encoder, reduction):
        super().__init__()

        self.pos_encoder = pos_encoder
        self.reduction = reduction

        self.embed_pos = LinearOutputStack(
            128, 2, in_channels=33, out_channels=64, activation=activation)

        self.embed_mag = LinearOutputStack(
            128, 2, in_channels=256, out_channels=64, activation=activation)
        self.embed_phase = LinearOutputStack(
            128, 2, in_channels=256, out_channels=64, activation=activation)
        self.down = nn.Linear(64 * 3, 128)
        # self.t = Mixer(128, 128, 5)
        self.t = MetaFormer(
            128,
            5,
            lambda channels: AttnMixer(channels),
            lambda channels: None)
        self.final = LinearOutputStack(128, 2, activation=activation)

        self.apply(init_weights)

    def forward(self, x):
        batch, time, channels, _ = x.shape

        # pos = self.pos_encoder(batch, 128, 16, x.device)
        pos = random_fourier_features(batch, 128, 64, 100)
        pos = self.embed_pos(pos)

        mag = x[..., 0]
        phase = x[..., 1]
        mag = self.embed_mag(mag)
        phase = self.embed_phase(phase)
        x = torch.cat([mag, phase, pos], dim=-1)
        x = self.down(x)

        x = self.t(x)

        x = self.reduction(x)
        x = self.final(x)
        return x

B = torch.FloatTensor(32, 32).normal_(0, 1).to(device)

def random_fourier_features(batch_size, seq_len, n_features, scale):
    pos = torch.linspace(0, 1, seq_len).to(device)
    freqs = torch.arange(1, (n_features // 2) + 1, 1).to(device)
    z = torch.outer(pos, freqs)
    z = z @ (B.T * scale)

    s = torch.sin(z * 2 * np.pi)
    c = torch.cos(z * 2 * np.pi)

    final = torch.cat([s, c], dim=1)

    final = final[None, ...].repeat(batch_size, 1, 1)
    return final

class OneDNerf(nn.Module):
    def __init__(self, pos_encoder):
        super().__init__()

        self.embed = nn.Linear(33, 128)

        self.net = nn.Sequential(
            LinearOutputStack(128, 5, activation=activation,
                              in_channels=128 * 2),
            # MetaFormer(
            #     128,
            #     5,
            #     lambda channels: AttnMixer(channels),
            #     lambda channels: None)
        )

        self.mag = LinearOutputStack(
            128, 3, out_channels=256, activation=activation)
        self.phase = LinearOutputStack(
            128, 3, out_channels=256, activation=activation)
        self.pos_encoder = pos_encoder

        self.apply(init_weights)

    def forward(self, conditioning):
        """
        Note: typically positional encoding would be passed
        in, but why not just generate it here, since it's static
        """
        pos = self.pos_encoder(1, 128, 16, device).repeat(
            conditioning.shape[0], 1, 1)
        pos = self.embed(pos)

        if conditioning is not None:
            conditioning = conditioning[:, None, :].repeat(1, pos.shape[1], 1)
            pos = torch.cat([conditioning, pos], dim=-1)

        x = self.net(pos)
        mag = torch.abs(self.mag(x))
        phase = torch.sin(self.phase(x)) * (np.pi * 2)
        x = torch.cat([mag[..., None], phase[..., None]], dim=-1)
        return x


def preprocess(batch, n_samples):
    batch /= (np.abs(batch).max() + 1e-12)
    batch = torch.from_numpy(batch).to(device).reshape(-1, n_samples)
    spec = codec.to_frequency_domain(batch).float().to(device)
    return spec


def stream(batch_size, n_samples, overfit=False):
    bs = batch_stream(
        Config.audio_path(),
        '*.wav',
        batch_size,
        n_samples)

    batch = preprocess(next(bs), n_samples)
    while True:
        yield batch
        if not overfit:
            batch = preprocess(next(bs), n_samples)


pe = pos_encoded

codec = AudioCodec(MelScale())
model = OneDNerf(pos_encoder=pe).to(device)
# model = ModulatedStack(128, 5, 256, pos_encoder=pe).to(device)
encoder = Encoder(pos_encoder=pe, reduction=reduce_last).to(device)
optim = Adam(model.parameters(), lr=1e-3, betas=(0, 0.9))


def train_ae(batch):
    optim.zero_grad()
    encoded = encoder(batch)
    decoded = model(encoded)

    loss = F.mse_loss(decoded[..., 0], batch[..., 0]) + F.mse_loss(decoded[..., 1], batch[..., 1])
    loss.backward()
    optim.step()
    print(loss.item())
    return encoded, decoded


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    batch_size = 4
    n_samples = 2 ** 15
    overfit = False

    data_stream = stream(1 if overfit else batch_size, n_samples, overfit)

    def pos_encoding():
        pos = pe(batch_size, 128, 16, device)
        return pos[0].data.cpu().numpy()

    def real_mag():
        return item[0, :, :, 0].data.cpu().numpy()

    def real_phase():
        return item[0, :, :, 1].data.cpu().numpy()

    def fake_mag():
        return r[0, :, :, 0].data.cpu().numpy()

    def fake_phase():
        return r[0, :, :, 1].data.cpu().numpy()

    def real():
        return codec.listen(item)

    def fake():
        return codec.listen(r)

    def latent():
        return e.data.cpu().numpy()

    for item in data_stream:
        e, r = train_ae(item)

    input('Waiting..')
