import zounds
from config.dotenv import Config
import numpy as np
from modules.linear import LinearOutputStack, ResidualBlock
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
- Can I learn 1d (YES) or 2d NERF representations of the new spectrogram representation
- Can I create an adaptible NERF model (i.e., NERF autoencoder)
'''

init_weights = make_initializer(0.1)

def activation(x):
    # return F.leaky_relu(x, 0.2)
    return torch.sin(x)


class ModulatedLayer(nn.Module):
    def __init__(self, channels, forward_layers, conditioning_layers):
        super().__init__()
        self.f = LinearOutputStack(channels, forward_layers, activation=activation)
        self.weight = LinearOutputStack(channels, conditioning_layers, activation=activation)
        self.bias = LinearOutputStack(channels, conditioning_layers, activation=activation)

    def forward(self, x, conditioning):
        x = self.f(x)
        w = self.weight(conditioning)
        b = self.bias(conditioning)

        return (x * torch.sigmoid(w)) + b


class ModulatedStack(nn.Module):
    def __init__(self, channels, layers, freq_bins):
        super().__init__()
        self.initial = LinearOutputStack(channels, 1, in_channels=33, activation=activation)
        self.net = nn.Sequential(
            *[ModulatedLayer(channels, 2, 2) for _ in range(layers)])
        self.mag = LinearOutputStack(channels, 3, out_channels=freq_bins, activation=activation)
        self.phase = LinearOutputStack(channels, 3, out_channels=freq_bins, activation=activation)

    def forward(self, latent):
        pos = pos_encoded(latent.shape[0], 128, 16, device)
        x = self.initial(pos)

        for layer in self.net:
            x = layer(x, latent[:, None, :])

        mag = self.mag(x)
        phase = self.phase(x)
        x = torch.cat([mag[..., None], phase[..., None]], dim=-1)
        return x


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_pos = LinearOutputStack(128, 2, in_channels=33, out_channels=64, activation=activation)

        self.embed_mag = LinearOutputStack(
            128, 2, in_channels=256, out_channels=64, activation=activation)
        self.embed_phase = LinearOutputStack(
            128, 2, in_channels=256, out_channels=64, activation=activation)
        self.down = nn.Linear(64 * 3, 128)
        # self.t = Transformer(128, 5)
        self.t = Mixer(128, 128, 5)
        self.final = LinearOutputStack(128, 2, activation=activation)
        self.apply(init_weights)

    def forward(self, x):
        batch, time, channels, _ = x.shape

        pos = pos_encoded(batch, 128, 16, x.device)
        pos = self.embed_pos(pos)

        mag = x[..., 0]
        phase = x[..., 1]
        mag = self.embed_mag(mag)
        phase = self.embed_phase(phase)
        x = torch.cat([mag, phase, pos], dim=-1)
        x = self.down(x)

        x = self.t(x)
        x = x[:, -1, :]
        x = self.final(x)
        return x


class OneDNerf(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(33, 128)
        self.net = LinearOutputStack(128, 5, activation=activation)
        self.mag = LinearOutputStack(128, 3, out_channels=256, activation=activation)
        self.phase = LinearOutputStack(128, 3, out_channels=256, activation=activation)
        self.apply(init_weights)

    def forward(self, x, conditioning=None):
        pos = pos_encoded(1, 128, 16, device)
        pos = self.embed(pos)

        if conditioning is not None:
            pos = conditioning + pos
        
        x = self.net(pos)
        mag = self.mag(x) ** 2
        phase = torch.sin(self.phase(x)) * (np.pi * 2)
        x = torch.cat([mag[..., None], phase[..., None]], dim=-1)
        return x


def preprocess(batch, n_samples):
    batch /= (np.abs(batch).max() + 1e-12)
    batch = torch.from_numpy(batch).to(device).reshape(-1, n_samples)
    spec = codec.to_frequency_domain(batch).float()
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


codec = AudioCodec(MelScale())
model = OneDNerf().to(device)
# model = ModulatedStack(128, 5, 256).to(device)
encoder = Encoder().to(device)
optim = Adam(model.parameters(), lr=1e-3, betas=(0, 0.9))

def train_ae(batch):
    optim.zero_grad()
    encoded = encoder(batch)
    decoded = model(encoded)

    loss = F.mse_loss(decoded, batch)
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
