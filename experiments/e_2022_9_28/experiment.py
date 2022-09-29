from pathlib import Path
import torch
from torch import nn
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.ddsp import AudioModel, overlap_add
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normal_pdf import pdf
from modules.psychoacoustic import PsychoacousticFeature
from modules.scattering import MoreCorrectScattering
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.transfer import ImpulseGenerator, PosEncodedImpulseGenerator, TransferFunction
from modules.fft import fft_convolve
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from modules.normalization import ExampleNorm, limit_norm, unit_norm
from torch.nn import functional as F
from modules.phase import MelScale
from modules.stft import stft
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
    kernel_size=512,
    residual_loss=False)

n_events = 16
event_latent_dim = exp.model_dim

mel_scale = MelScale()


band = zounds.FrequencyBand(30, exp.samplerate.nyquist)
scale = zounds.MelScale(band, 128)



class SegmentGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.to_audio_params = ConvUpsample(
            event_latent_dim,
            exp.model_dim,
            4,
            exp.n_frames,
            mode='nearest',
            out_channels=exp.model_dim)
        self.audio_model = AudioModel(
            exp.n_samples, 
            exp.model_dim, 
            exp.samplerate, 
            exp.n_frames, 
            exp.n_frames * 8)
        
        '''
        Things to try here:
            - PDF
            - FFT shift
            - softmax
            - gumbel softmax
            - pos encoded
        '''
        # self.to_impulse = LinearOutputStack(exp.model_dim, 2, out_channels=exp.model_dim)
        self.to_impulse = ConvUpsample(event_latent_dim, exp.model_dim, 4, exp.n_frames, mode='nearest', out_channels=1)
        self.impulse = ImpulseGenerator(exp.n_samples, softmax=lambda x: F.gumbel_softmax(x, dim=-1, hard=True))

        self.model_dim = exp.model_dim
        self.n_samples = exp.n_samples
        self.n_frames = exp.n_frames
        self.window_size = 512


    def forward(self, time, transfer):
        transfer = transfer.view(-1, event_latent_dim)
        time = time.view(-1, event_latent_dim)

        batch = time.shape[0]

        orig_env = time
        orig_tf = transfer
        loss = 0

        # generate audio event
        event = self.to_audio_params(transfer)
        event = self.audio_model(event)

        # generate impulse/placement
        imp = self.to_impulse(time).view(-1, exp.n_frames)
        imp = self.impulse(imp)

        # positions audio events in time
        final = fft_convolve(event, imp)

        final = torch.mean(final, dim=1, keepdim=True)

        return final, orig_env, loss, orig_tf, event


class Model(nn.Module):
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
            exp.model_dim, 1, out_channels=event_latent_dim)

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
        time = self.to_time(x).view(-1, event_latent_dim)

        transfer = self.to_transfer(x).view(-1, event_latent_dim)

        x, env, loss, tf, events = self.decode.forward(time, transfer)
        x = x.view(batch, n_events, exp.n_samples)

        output = torch.sum(x, dim=1, keepdim=True)

        loss = 0
        return output, indices, encoded, env.view(batch, n_events, -1), loss, tf, time, transfer, events




model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded, env, vq_loss, tf, time, transfer, events = model.forward(
        batch)
    
    event_spec = stft(events.view(-1, exp.n_samples), 512, 256, pad=True)
    spec_diff = torch.diff(event_spec, dim=1)
    diff_norm = torch.norm(spec_diff, dim=-1).mean() * 0.1

    pl = exp.perceptual_loss(recon, batch)


    loss = pl + diff_norm
    loss.backward()
    optim.step()
    return loss, recon, indices, encoded, env, time, transfer


@readme
class PlacementExperiment(object):
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
