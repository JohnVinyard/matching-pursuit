import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.ddsp import overlap_add
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normal_pdf import pdf
from modules.sparse import VectorwiseSparsity
from train.optim import optimizer
from upsample import ConvUpsample
from modules.normalization import ExampleNorm, limit_norm
from torch.nn import functional as F

from util import device, playable
from modules import pos_encoded

from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

n_events = 8


class SegmentGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.env = ConvUpsample(
            exp.model_dim,
            exp.model_dim,
            4,
            exp.n_frames,
            out_channels=1,
            mode='learned',
            norm=ExampleNorm())

        self.n_coeffs = 257

        self.model_dim = exp.model_dim
        self.n_samples = exp.n_samples
        self.n_frames = exp.n_frames
        self.window_size = 512

        self.n_inflections = 4

        self.transfer = LinearOutputStack(
            exp.model_dim, 3, out_channels=self.n_coeffs * 2 * self.n_inflections)
        self.means = LinearOutputStack(
            exp.model_dim, 3, out_channels=self.n_inflections)
        self.stds = LinearOutputStack(
            exp.model_dim, 3, out_channels=self.n_inflections)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, self.model_dim)

        means = torch.sigmoid(self.means(x))
        stds = torch.sigmoid(self.stds(x)) * 0.25

        rng = torch.linspace(0, 1, self.n_samples, device=means.device)
        selections = pdf(rng[None, None, :],
                         means[:, :, None], stds[:, :, None])
        selections = selections / (selections.max() + 1e-8)

        env = self.env(x) ** 2
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        env_selections = env * selections

        tf = self.transfer(x)
        tf = tf.view(-1, self.n_inflections, self.n_coeffs *
                     2, 1).repeat(1, 1, 1, self.n_frames)
        tf = tf.view(-1, self.n_inflections, self.n_coeffs, 2, self.n_frames)

        # ensure that the norm of the coefficients does not exceed one
        # to avoid feedback
        tf = limit_norm(tf, dim=3)

        tf = tf.view(-1, self.n_inflections, self.n_coeffs * 2, self.n_frames)

        real = tf[:, :, :self.n_coeffs, :]
        imag = tf[:, :, self.n_coeffs:, :]

        tf = torch.complex(real, imag)

        tf = torch.cumprod(tf, dim=-1)

        tf = tf.view(-1, self.n_coeffs, self.n_frames)
        tf = torch.fft.irfft(tf, dim=1, norm='ortho')
        tf = \
            tf.permute(0, 2, 1).view(-1, 1, self.n_frames, self.window_size) \
            * torch.hamming_window(self.window_size, device=tf.device)[None, None, None, :]

        tf = overlap_add(tf, trim=self.n_samples)

        tf = tf.view(batch, self.n_inflections, self.n_samples)

        env_spec = torch.fft.rfft(env_selections, dim=-1, norm='ortho')
        tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
        spec = env_spec * tf_spec
        final = torch.fft.irfft(spec, dim=-1, norm='ortho')

        final = torch.mean(final, dim=1, keepdim=True)

        return final


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
            exp.model_dim, keep=n_events, channels_last=False, dense=False)

        self.decode = SegmentGenerator()

        self.norm = ExampleNorm()

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

        x, indices = self.sparse(x)
        x = self.norm(x)
        encoded = x

        x = x.view(-1, exp.model_dim)

        x = self.decode(x).view(batch, n_events, exp.n_samples)

        output = torch.zeros(batch, 1, exp.n_samples * 2, device=x.device)
        for b in range(batch):
            for i in range(n_events):
                start = indices[b, i] * 256
                end = start + exp.n_samples
                output[b, :, start: end] += x[b, i]

        output = output[..., :exp.n_samples]

        return output, indices, encoded


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x):
        x, indices, encoded = self.summary(x)
        return x, indices, encoded


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon, indices, encoded


@readme
class WaveguideSynthesisExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.indices = None
        self.encoded = None

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

    def encoding(self):
        return self.encoded.data.cpu().numpy().reshape((-1, exp.model_dim))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.real = item
            loss, self.fake, self.indices, self.encoded = train_model(item)
            print('GEN', i, loss.item())
