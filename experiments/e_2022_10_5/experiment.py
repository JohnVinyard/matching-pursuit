from pathlib import Path
from re import M
import torch
from torch import log_, nn
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.transfer import ImpulseGenerator, PosEncodedImpulseGenerator, TransferFunction, position, schedule_atoms
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



class GenerateSpectrogram(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_time_latent = LinearOutputStack(exp.model_dim, 2, in_channels=33)

        self.to_tf_latent = nn.Sequential(
            nn.Conv1d(32, 64, 7, 4, 3), # 32
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 7, 4, 3), # 8
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 3, 2, 1), # 4
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 4, 4, 0)
        )

        self.env_latent = nn.Sequential(
            nn.Conv1d(1, 16, 7, 4, 3), # 64
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 7, 4, 3), # 16
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 7, 4, 3), # 4
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, 4, 0)
        )

        self.reduce = nn.Conv1d(exp.model_dim * 3, exp.model_dim, 1, 1, 0)

        self.to_spec = ConvUpsample(
            exp.model_dim, 
            exp.model_dim, 
            4, 
            exp.n_frames, 
            mode='learned', 
            out_channels=257)
        

        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, times, envs, transfer_functions):
        envs = envs.view(-1, 1, 256)

        tf = self.to_tf_latent(transfer_functions).view(-1, 128, 1)

        env = self.env_latent(envs)
        env = env.view(-1, 128, 1)

        time = self.to_time_latent(times).view(-1, 128, 1)

        x = torch.cat([env, time, tf], dim=1)
        x = self.reduce(x)

        x = self.to_spec(x).view(-1, n_events, 257, 128)
        x = torch.sum(x, dim=1).permute(0, 2, 1)
        return x


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
            exp.model_dim, 
            exp.model_dim, 
            8, 
            scale.n_bands, 
            mode='nearest', 
            out_channels=self.resolution)

        self.tf = TransferFunction(
            exp.samplerate, 
            scale, 
            exp.n_frames, 
            self.resolution, 
            exp.n_samples, 
            softmax_func=lambda x: F.gumbel_softmax(x, dim=-1, hard=True))
        


    def forward(self, transfer):
        transfer = transfer.view(-1, event_latent_dim)

        # batch = time.shape[0]
        batch = transfer.shape[0]

        # create envelope, normalized to one
        # attack, irrespective of amplitude
        env = self.env(transfer)
        env = env ** 2
        orig_env = env

        env = env.view(batch, 1, -1)
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        loss = 0
        tf = self.transfer.forward(transfer)
        orig_tf = tf
        tf = tf.permute(0, 2, 1).view(batch, scale.n_bands, self.resolution)
        tf = self.tf.forward(tf)

        final = fft_convolve(env, tf)

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
            exp.model_dim, 1, out_channels=33)
        self.impulse = PosEncodedImpulseGenerator(
            exp.n_frames, exp.n_samples, softmax=lambda x: F.gumbel_softmax(x, dim=-1, hard=True))

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
        orig = x

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
        time = self.to_time(x).view(-1, 33)
        impulse, _ = self.impulse(time)
        impulse = impulse.view(-1, n_events, exp.n_samples)

        transfer = self.to_transfer(x).view(-1, event_latent_dim)

        x, env, loss, tf = self.decode.forward(transfer)

        x = x.view(batch, n_events, exp.n_samples)

        x = fft_convolve(x, impulse)

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
optim = optimizer(model, lr=1e-3)

synth_grad = GenerateSpectrogram().to(device)
grad_optim = optimizer(synth_grad, lr=1e-3)


def train_synthetic_grad(batch):
    grad_optim.zero_grad()

    with torch.no_grad():
        # get the spec of the reconstruction
        recon, indices, encoded, env, vq_loss, tf, time, transfer = model.forward(batch)
        spec = stft(recon, 512, 256, pad=256, log_amplitude=True).view(-1, 128, 257)

    # get the predicted spec based on params
    specs = synth_grad.forward(time, env, tf)

    # bring the prediction closer to actual spec
    loss = F.mse_loss(specs, spec)
    loss.backward()
    grad_optim.step()
    return loss, specs


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded, env, vq_loss, tf, time, transfer = model.forward(batch)
    
    # get predicted spec
    gen_spec = synth_grad.forward(time, env, tf)

    # compute real spec
    spec = stft(batch, 512, 256, pad=256, log_amplitude=True).view(-1, 128, 257)

    # bring the predicted spec closer to the real spec
    loss = F.mse_loss(gen_spec, spec)

    
    # ll = latent_loss(transfer) * 0.5


    # e = env.view(-1, exp.n_frames * 2)

    # e = torch.softmax(e, dim=-1)

    # env_centroids = \
    #     torch.sum(torch.linspace(0, 1, e.shape[-1], device=e.device)[None, :] * e, dim=-1) \
    #     / torch.sum(e, dim=-1, keepdim=True)

    # the assumption is that events should be uniformly distributed
    # time_loss = torch.abs(time.mean() - 0.5) + torch.abs(time.std() - 0.25) * 0.5

    # env_loss = env_centroids.mean() * 0.01

    # loss = exp.perceptual_loss(recon, batch) + ll #+ env_loss


    loss.backward()
    optim.step()
    return loss, recon, indices, encoded, env, time, transfer


@readme
class SyntheticGradients(object):
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

        self.generated_spec = None

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
        x = self.time_latent.data.cpu().numpy().squeeze().reshape((-1, n_events))
        x = np.argmax(x, axis=0)
        return x

    def tf_latent(self):
        return self.transfer_latent.data.cpu().numpy().squeeze().reshape((-1, event_latent_dim))
    
    def gen_spec(self):
        return self.generated_spec.data.cpu().numpy().squeeze()

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

            if i % 2 == 0:
                self.real = item
                loss, self.fake, self.indices, self.encoded, self.env, self.time_latent, self.transfer_latent = train_model(
                    item)
                print('GEN', i, loss.item())
            else:
                loss, self.generated_spec = train_synthetic_grad(item)
                print('SYTN GRAD', loss.item())
