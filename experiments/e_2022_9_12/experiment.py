import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.ddsp import overlap_add
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normal_pdf import pdf
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from train.optim import optimizer
from upsample import ConvUpsample
from modules.normalization import ExampleNorm, limit_norm
from torch.nn import functional as F
from modules.phase import MelScale
from vector_quantize_pytorch import VectorQuantize
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from modules.latent_loss import latent_loss

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

n_events = 16
event_latent_dim = 8

mel_scale = MelScale()

class SegmentGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.env = ConvUpsample(
            event_latent_dim,
            exp.model_dim,
            4,
            exp.n_frames,
            out_channels=1,
            mode='nearest',
            norm=ExampleNorm())


        self.n_coeffs = 256

        self.model_dim = exp.model_dim
        self.n_samples = exp.n_samples
        self.n_frames = exp.n_frames
        self.window_size = 512

        self.n_inflections = 1

        
        self.transfer = LinearOutputStack(
            exp.model_dim, 
            3, 
            out_channels=self.n_coeffs * 2 * self.n_inflections, 
            in_channels=event_latent_dim)

    def forward(self, time, transfer):
        time = time.view(-1, self.model_dim)
        transfer = transfer.view(-1, event_latent_dim)
        time = time.view(-1, event_latent_dim)

        # x = x.view(-1, self.model_dim)
        batch = time.shape[0]


        # create envelope
        env = self.env(time) ** 2
        orig_env = env
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        
        tf = self.transfer(transfer)
        loss = 0

        tf = tf.view(-1, self.n_inflections, self.n_coeffs *
                     2, 1).repeat(1, 1, 1, self.n_frames)
        tf = tf.view(-1, self.n_inflections, self.n_coeffs, 2, self.n_frames)


        tf = tf.view(-1, self.n_inflections, self.n_coeffs * 2, self.n_frames)
        orig_tf = tf.view(-1, self.n_coeffs * 2, self.n_frames)

        real = torch.clamp(tf[:, :, :self.n_coeffs, :], 0, 1) * 0.9999
        imag = torch.clamp(tf[:, :, self.n_coeffs:, :], -1, 1) * np.pi

        real = real * torch.cos(imag)
        imag = real * torch.sin(imag)
        tf = torch.complex(real, imag)
        tf = torch.cumprod(tf, dim=-1)
        tf = tf.view(-1, self.n_coeffs, self.n_frames)
        tf = mel_scale.to_time_domain(tf.permute(0, 2, 1))[..., :self.n_samples]
        tf = tf.view(batch, self.n_inflections, self.n_samples)

        # convolve impulse with transfer function
        env_spec = torch.fft.rfft(env, dim=-1, norm='ortho')
        tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
        spec = env_spec * tf_spec
        final = torch.fft.irfft(spec, dim=-1, norm='ortho')

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
            exp.model_dim, keep=n_events, channels_last=False, dense=False)

        self.decode = SegmentGenerator()

        self.to_time = LinearOutputStack(exp.model_dim, 3, out_channels=event_latent_dim)
        self.to_transfer = LinearOutputStack(exp.model_dim, 3, out_channels=event_latent_dim)

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
        x = self.norm(x)

        x, indices = self.sparse(x)
        encoded = x

        # split into independent time and transfer function representations
        time = self.to_time(x).view(-1, event_latent_dim)
        transfer = self.to_transfer(x).view(-1, event_latent_dim)

        x, env, loss, tf = self.decode.forward(time, transfer)
        x = x.view(batch, n_events, exp.n_samples)

        output = torch.sum(x, dim=1, keepdim=True)

        # output = torch.zeros(batch, 1, exp.n_samples * 2, device=x.device)
        # for b in range(batch):
        #     for i in range(n_events):
        #         start = indices[b, i] * 256
        #         end = start + exp.n_samples
        #         output[b, :, start: end] += x[b, i]

        # output = output[..., :exp.n_samples]

        return output, indices, encoded, env.view(batch, n_events, -1), loss, tf, time, transfer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x):
        x, indices, encoded, env, loss, tf, time, transfer = self.summary(x)
        return x, indices, encoded, env, loss, tf, time, transfer


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded, env, vq_loss, tf, time, transfer = model.forward(batch)

    # diff = torch.diff(tf, dim=-1)
    # diff_norms = torch.norm(diff, dim=1).mean() * 0.1

    time_loss = latent_loss(time) * 0.1
    tf_loss = latent_loss(transfer) * 0.1

    loss = exp.perceptual_loss(recon, batch) + time_loss + tf_loss 
    loss.backward()
    clip_grad_norm_(model.parameters(), 1)
    optim.step()
    return loss, recon, indices, encoded, env, time, transfer


@readme
class WaveguideSynthesisExperiment2(object):
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

    # def encoding(self):
        # return self.encoded.data.cpu().numpy().reshape((-1, exp.model_dim))
    
    def e(self):
        return self.env.data.cpu().numpy()[0].T
    
    def t_latent(self):
        return self.time_latent.data.cpu().numpy().squeeze().reshape((-1, event_latent_dim))
    
    def tf_latent(self):
        return self.transfer_latent.data.cpu().numpy().squeeze().reshape((-1, event_latent_dim))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.real = item
            loss, self.fake, self.indices, self.encoded, self.env, self.time_latent, self.transfer_latent = train_model(item)
            print('GEN', i, loss.item())
