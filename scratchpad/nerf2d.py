import zounds
from config.dotenv import Config
import numpy as np
from modules.linear import LinearOutputStack, ResidualBlock
from modules.metaformer import AttnMixer, MetaFormer
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

batch_size = 1

init_weights = make_initializer(0.02)

time_encoding = pos_encoded(batch_size, 128, 16, device)
freq_encoding = pos_encoded(batch_size, 256, 16, device)

pos = torch.zeros(1, 128, 256, 66)
for i in range(128):
    for j in range(256):
        pos[0, i, j, :] = torch.cat([time_encoding[0, i], freq_encoding[0, j]])


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


class Nerf2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = LinearOutputStack(64, 4, in_channels=66)
        self.mag = LinearOutputStack(64, 4, out_channels=1)
        self.phase = LinearOutputStack(64, 4, out_channels=1)
        self.apply(init_weights)

    def forward(self, pos, conditioning):
        pos = pos.repeat(conditioning.shape[0], 1, 1, 1)
        output = self.net(pos)

        mag = self.mag(output) ** 2
        phase = torch.sin(self.phase(output)) * np.pi * 2


        output = torch.cat([mag, phase], dim=-1)
        return output


model = Nerf2d().to(device)
optim = Adam(model.parameters(), lr=1e-4, betas=(0, 0.9))


def train_ae(batch):
    optim.zero_grad()

    conditioning = torch.FloatTensor(batch_size, 32).normal_(0, 1).to(device)
    decoded = model(pos, conditioning)

    loss = F.mse_loss(decoded, batch)
    loss.backward()
    optim.step()
    print(loss.item())
    return conditioning, decoded

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    n_samples = 2 ** 15
    overfit = True

    data_stream = stream(1 if overfit else batch_size, n_samples, overfit)

    def pos_encoding():
        pos = pos_encoded(batch_size, 128, 16, device)
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
        _, r = train_ae(item)

    input('Waiting..')
