from pathlib import Path
import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.sparse import AtomPlacement, VectorwiseSparsity
from modules.transfer import ImpulseGenerator, PosEncodedImpulseGenerator, TransferFunction, schedule_atoms
from modules.shape import Reshape
from modules.fft import fft_convolve, fft_shift
from train.optim import optimizer
from upsample import ConvUpsample, FFTUpsampleBlock, PosEncodedUpsample
from modules.normalization import ExampleNorm, max_norm
from torch.nn import functional as F
from modules.latent_loss import latent_loss


from util import device, playable
from modules import pos_encoded
from util.music import MusicalScale

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


scale = MusicalScale()

def fft_convolve(env, tf):
    env = F.pad(env, (0, env.shape[-1]))
    tf = F.pad(tf, (0, tf.shape[-1]))

    env_spec = torch.fft.rfft(env, dim=-1, norm='ortho')
    tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
    spec = env_spec * tf_spec
    final = torch.fft.irfft(spec, dim=-1, norm='ortho')
    return final[..., :exp.n_samples]



class SegmentGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.env = ConvUpsample(
        #     event_latent_dim,
        #     exp.model_dim,
        #     4,
        #     exp.n_frames * 2,
        #     out_channels=1,
        #     mode='nearest')

        self.env = nn.Sequential(
            nn.Linear(event_latent_dim, exp.model_dim * 4),
            Reshape((exp.model_dim, 4)),

            nn.ConvTranspose1d(exp.model_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(32, 16, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(16, 8, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(8, 4, 4, 2, 1),
            nn.LeakyReLU(0.2),

            nn.Conv1d(4, 1, 3, 1, 1)
        )

        

        self.n_coeffs = 257

        self.model_dim = exp.model_dim
        self.n_samples = exp.n_samples
        self.n_frames = exp.n_frames
        self.window_size = 512

        self.n_inflections = 1

        
        self.resolution = 64

        
        self.is_continuous = False

        self.transfer = ConvUpsample(
            exp.model_dim, 
            exp.model_dim, 
            8, 
            scale.n_bands, 
            mode='nearest', 
            out_channels=1 if self.is_continuous else self.resolution)
        

        self.tf = TransferFunction(
            exp.samplerate, 
            scale, 
            exp.n_frames, 
            self.resolution, 
            exp.n_samples, 
            softmax_func=lambda x: F.gumbel_softmax(x, dim=-1, hard=True),
            is_continuous=self.is_continuous,
            resonance_exp=1)
        



    def forward(self, time, transfer):
        transfer = transfer.view(-1, event_latent_dim)
        time = time.view(-1, event_latent_dim)

        # x = x.view(-1, self.model_dim)
        batch = transfer.shape[0]

        # create envelope
        env = self.env(time).view(batch, 1, -1)
        env = torch.relu(env)
        orig_env = env
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1)
        env = env * noise

        loss = 0
        tf = self.transfer.forward(transfer)

        tf = tf.permute(0, 2, 1)
        if not self.is_continuous:
            tf = tf.view(batch, scale.n_bands, self.resolution)
        tf = self.tf.forward(tf)
        orig_tf = None

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
            exp.model_dim, 1, out_channels=event_latent_dim)

        self.to_transfer = LinearOutputStack(
            exp.model_dim, 1, out_channels=event_latent_dim)

        self.norm = ExampleNorm()



    def forward(self, x):
        batch = x.shape[0]

        target = x = x.view(-1, 1, exp.n_samples)

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
        orig_time = time = self.to_time(x).view(batch, n_events, exp.model_dim)
        t_loss = 0

        orig_transfer = transfer = self.to_transfer(x).view(-1, event_latent_dim)
        # transfer, tf_indices, tf_loss = self.transfer_vq(transfer)
        tf_indices = None
        tf_loss = 0

        x, env, loss, tf = self.decode.forward(time, transfer)
        x = x.view(batch, n_events, exp.n_samples)

        output = torch.sum(x, dim=1, keepdim=True)

        loss = t_loss + tf_loss
        return output, indices, encoded, env.view(batch, n_events, -1), loss, tf, time, transfer, orig_time, orig_transfer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))
    
    def encode(self, x):
        return self.summary.encode(x)
    
    def decode(self, time, transfer):
        return self.generate(time, transfer)

    def generate(self, time, transfer):
        return self.summary.generate(time, transfer)

    def forward(self, x):
        x, indices, encoded, env, loss, tf, time, transfer, orig_time, orig_transfer = self.summary(x)
        return x, indices, encoded, env, loss, tf, time, transfer, orig_time, orig_transfer


model = Model().to(device)
# try:
#     model.load_state_dict(torch.load(Path(__file__).parent.joinpath('model.dat'), map_location=device))
#     print('loaded model')
# except IOError:
#     print('Could not load weights')
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded, env, vq_loss, tf, time, transfer, orig_time, orig_transfer = model.forward(batch)

    # transfer latent should be from a standard normal distribution
    # ll = (latent_loss(orig_transfer)) * 0.5


    # peripheral_loss = vq_loss + ll
    pl = exp.perceptual_loss(recon, batch)

    loss = pl #+ peripheral_loss
    loss.backward()
    optim.step()
    return loss, recon, indices, encoded, env, time, transfer


@readme
class WaveguideSynthesisExperiment4(object):
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
