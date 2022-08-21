import numpy as np
from modules.ddsp import overlap_add
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device, playable, readme
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from modules.psychoacoustic import PsychoacousticFeature
from util import make_initializer

n_samples = 2 ** 15
samplerate = zounds.SR22050()

window_size = 512
step_size = window_size // 2
n_coeffs = (window_size // 2) + 1

n_frames = n_samples // step_size

model_dim = 128
latent_dim = 128
n_atoms = 16

n_bands = 128
kernel_size = 512

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


pif = PsychoacousticFeature([128] * 6).to(device)

init_weights = make_initializer(0.1)

def perceptual_feature(x):
    bands = pif.compute_feature_dict(x)
    return torch.cat(bands, dim=-2)

def perceptual_loss(a, b):
    return F.mse_loss(a, b)



class ToSamples(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Conv1d(n_frames, n_frames, 7, 1, 3),
            # nn.Upsample(scale_factor=2),
            # nn.LeakyReLU(0.2),

            # nn.Conv1d(n_frames, n_frames, 7, 1, 3),
            # nn.Upsample(scale_factor=2),
            # nn.LeakyReLU(0.2),

            # # nn.Conv1d(n_frames, n_frames, 7, 1, 3),
            # # nn.Upsample(scale_factor=2),
            # # nn.LeakyReLU(0.2),

            # nn.Conv1d(n_frames, n_frames, 7, 1, 3),
            nn.Conv1d(model_dim, window_size, 1, 1, 0)
        )

    def forward(self, x):
        x = self.net(x)
        x = x.permute(0, 2, 1)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        encoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(encoder, 6, norm=None)
        self.to_env = nn.Conv1d(model_dim, 1, 1, 1, 0)
        self.embed = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)


        decoder = nn.TransformerEncoderLayer(model_dim, 4, model_dim, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder, 6, norm=None)

        self.to_samples = ToSamples()

        self.embed_again = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)
        self.apply(init_weights)

    def forward(self, x):
        batch = x.shape[0]

        orig = x

        n = fb.forward(orig, normalize=False)
        n = fb.temporal_pooling(n, 512, 256)[..., :n_frames]
        pos = pos_encoded(batch, n_frames, n_freqs=16, device=n.device).permute(0, 2, 1)
        n = torch.cat([pos, n], dim=1)

        n = self.embed(n)
        n = n.permute(0, 2, 1)
        x = self.context(n)
        x = x.permute(0, 2, 1)

        norms = self.to_env(x).view(batch, -1)
        norms = torch.softmax(norms, dim=-1)
        values, indices = torch.topk(norms, k=n_atoms, dim=-1)
        n = torch.zeros_like(x)

        for b in range(batch):
            for i in range(n_atoms):
                # latents.append(x[b, :, indices[b, i]][None, :])
                n[b, :, indices[b, i]] = x[b, :, indices[b, i]] * values[b, i]
            

        n = torch.cat([pos, n], dim=1)
        n = self.embed_again(n)

        n = n.permute(0, 2, 1)
        n = self.decoder(n) # (batch, n_frames, model_dim)
        n = n.permute(0, 2, 1)

        n = self.to_samples(n) # (batch, n_frames, window_size)
        n = n * torch.hamming_window(window_size, device=n.device)[None, None, :]
        final = overlap_add(n[:, None, :, :])
        return final[..., :n_samples]


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return recon, loss


@readme
class SparseAutoencoderExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None

    def listen(self):
        return playable(self.fake, samplerate)

    def orig(self):
        return playable(self.real, samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            self.fake, loss = train(item)

            print(i, loss.item())
