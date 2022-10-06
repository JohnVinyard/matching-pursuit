from pathlib import Path
import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.fft import fft_convolve, fft_shift
from train.optim import optimizer
from upsample import ConvUpsample
from modules.normalization import ExampleNorm
from torch.nn import functional as F
from modules.phase import MelScale
from modules.latent_loss import latent_loss

from util import device, playable
from modules import pos_encoded, stft

from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 16
event_latent_dim = exp.model_dim

mel_scale = MelScale()



band = zounds.FrequencyBand(30, exp.samplerate.nyquist)
scale = zounds.MelScale(band, 128)



class SegmentGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.env = ConvUpsample(
            event_latent_dim,
            exp.model_dim,
            4,
            exp.n_frames * 2,
            out_channels=1,
            mode='nearest')

        self.n_coeffs = 257

        self.model_dim = exp.model_dim
        self.n_samples = exp.n_samples
        self.n_frames = exp.n_frames
        self.window_size = 512

        self.n_inflections = 1

        
        self.resolution = 32

        self.transfer = ConvUpsample(
            event_latent_dim, 
            exp.model_dim, 
            8, 
            exp.n_samples, 
            out_channels=1, 
            mode='learned')

        
        self.transfer = nn.Sequential(
            nn.Linear(128, 1024),
            Reshape((128, 8)),
            nn.ConvTranspose1d(128, 64, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 32, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, 8, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(8, 4, 8, 4, 2),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(4, 1, 8, 4, 2),
        )



    def forward(self, time, transfer):
        transfer = transfer.view(-1, event_latent_dim)
        # time = time.view(-1, event_latent_dim)

        # x = x.view(-1, self.model_dim)
        batch = transfer.shape[0]

        # create envelope
        env = self.env(transfer) ** 2
        orig_env = env
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        # Theory: having trouble not relying on energy/noise because the
        # gradients here are just too small
        # tf = self.transfer(transfer)
        loss = 0
        tf = self.transfer.forward(transfer)
        orig_tf = tf

        
        final = fft_convolve(env, tf)
        # final = tf

        final = torch.mean(final, dim=1, keepdim=True)

        return final, orig_env, loss, orig_tf


class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self):
        super().__init__()

        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])

        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

        self.decode = SegmentGenerator()

        self.to_time = LinearOutputStack(
            exp.model_dim, 1, out_channels=1)

        self.to_transfer = LinearOutputStack(
            exp.model_dim, 1, out_channels=event_latent_dim)

        self.norm = ExampleNorm()

    def generate(self, time, transfer):
        x, env, loss, tf = self.decode.forward(time, transfer)
        x = x.view(-1, n_events, exp.n_samples)
        output = torch.sum(x, dim=1, keepdim=True)
        return output

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, exp.n_samples)
        x = exp.fb.forward(x, normalize=False)
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        x = self.norm(x)
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        x = self.context(x)
        x = self.norm(x)

        x, indices = self.sparse(x)
        encoded = x

        # split into independent time and transfer function representations
        time = torch.sigmoid(self.to_time(x).view(-1, n_events))

        transfer = self.to_transfer(x).view(-1, event_latent_dim)


        x, env, loss, tf = self.decode.forward(time, transfer)
        x = x.view(batch, n_events, exp.n_samples)

        output = fft_shift(x, time[..., None])

        output = torch.sum(x, dim=1, keepdim=True)


        loss = 0
        return output, indices, encoded, env.view(batch, n_events, -1), loss, tf, time, transfer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))

    def generate(self, time, transfer):
        return self.summary.generate(time, transfer)

    def forward(self, x):
        x, indices, encoded, env, loss, tf, time, transfer = self.summary(x)
        return x, indices, encoded, env, loss, tf, time, transfer


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded, env, vq_loss, tf, time, transfer = model.forward(
        batch)
    
    # ll = (latent_loss(time) + latent_loss(transfer)) * 0.25
    ll = latent_loss(transfer)

    time_loss = torch.abs(time.mean() - 0.5) + torch.abs(time.std() - 0.25)

    tf_spec = stft(tf, 512, 256, pad=True)
    norms = torch.norm(tf_spec, dim=-1, keepdim=True)
    normed = tf_spec / (norms + 1e-8)
    diff = torch.diff(normed, dim=-2)
    diff_norms = torch.norm(diff, dim=-1)
    tf_loss = diff_norms.mean()

    loss = exp.perceptual_loss(recon, batch) + ll + time_loss + tf_loss


    loss.backward()
    optim.step()
    return loss, recon, indices, encoded, env, time, transfer


@readme
class LastTry(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.indices = None
        self.encoded = None
        self.env = None

        self.time_latent = None
        self.transfer_latent = None

        self.model = model

    def orig(self):
        return playable(self.real, exp.samplerate, normalize=True)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, exp.samplerate, normalize=True)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def positions(self):
        indices = self.indices.data.cpu().numpy()[0]
        canvas = np.zeros((exp.n_frames, 16))
        canvas[indices] = 1
        return canvas


    def e(self):
        return self.env.data.cpu().numpy()[0].T

    def t_latent(self):
        return self.time_latent.data.cpu().numpy().squeeze()

    def tf_latent(self):
        return self.transfer_latent.data.cpu().numpy().squeeze().reshape((-1, event_latent_dim))

    def random(self, n_events=16, t_std=0.05, tf_std=0.05):
        with torch.no_grad():
            t = torch.zeros(1, n_events, event_latent_dim,
                            device=device).normal_(0, t_std)
            tf = torch.zeros(1, n_events, event_latent_dim,
                             device=device).normal_(0, tf_std)
            audio = model.generate(t, tf)
            return playable(audio, exp.samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.real = item
            loss, self.fake, self.indices, self.encoded, self.env, self.time_latent, self.transfer_latent = train_model(
                item)
            print('GEN', i, loss.item())
