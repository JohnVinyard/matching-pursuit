from pathlib import Path
import torch
from torch import nn
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.ddsp import overlap_add
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normal_pdf import pdf
from modules.psychoacoustic import PsychoacousticFeature
from modules.scattering import MoreCorrectScattering
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.transfer import ImpulseGenerator, PosEncodedImpulseGenerator, TransferFunction, schedule_atoms
from modules.fft import fft_convolve, fft_shift
from train.optim import optimizer
from upsample import ConvUpsample, FFTUpsampleBlock, PosEncodedUpsample
from modules.normalization import ExampleNorm, limit_norm, max_norm, unit_norm
from torch.nn import functional as F
from modules.phase import MelScale
from modules.stft import stft
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_
from modules.latent_loss import latent_loss
from collections import Counter


from util import device, playable
from modules import pos_encoded
from util.music import MusicalScale

from util.readmedocs import readme
import numpy as np

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)

n_events = 16
event_latent_dim = exp.model_dim

def hard_softmax(x, soft=False, backward_trick=True):
    x = torch.softmax(x, dim=-1)

    if soft:
        return x
    
    x_backward = x
    
    values, indices = torch.max(x, dim=-1, keepdim=True)
    values = values + (1 - values)
    z = torch.zeros_like(x)
    z = torch.scatter(z, dim=-1, index=indices, src=values)

    if not backward_trick:
        return z
    
    x_forward = z
    return x_backward + (x_forward - x_backward).detach()

def exp_softmax(x):
    return hard_softmax(x, soft=False, backward_trick=True)

    x = max_norm(x, dim=-1)
    return F.gumbel_softmax(torch.exp(x), tau=1, dim=-1, hard=True)
    

scale = MusicalScale()

# TODO: What about discretization is causing issues?
class VQ(nn.Module):
    def __init__(self, code_dim, n_codes, commitment_weight=1, one_hot=False, passthrough=False):
        super().__init__()
        self.code_dim = code_dim
        self.n_codes = n_codes
        self.codebook = nn.Parameter(torch.zeros(
            n_codes, code_dim).normal_(0, 1))

        self.commitment_cost = commitment_weight

        self.one_hot = one_hot

        self.passthrough = passthrough

        if self.one_hot:
            self.up = nn.Linear(code_dim, n_codes)
            self.down = nn.Linear(n_codes, code_dim)
        

    def forward(self, x):

        if self.passthrough:
            return x, None, 0

        if self.one_hot:
            x = self.up(x)
            x = exp_softmax(x)
            x = self.down(x)
            return x, None, 0


        dist = torch.cdist(x, self.codebook)
        mn, indices = torch.min(dist, dim=-1)
        codes = self.codebook[indices]

        # bring codes closer to embeddings
        code_loss = ((codes - x.detach()) ** 2).mean()

        # bring embeddings closter to codes
        embedding_loss = (((x - codes.detach()) ** 2).mean() * self.commitment_cost)

        loss = code_loss + embedding_loss

        quantized = x + (codes - x).detach()
        return quantized, indices, loss

def fft_convolve(env, tf):
    env = F.pad(env, (0, env.shape[-1]))
    tf = F.pad(tf, (0, tf.shape[-1]))

    env_spec = torch.fft.rfft(env, dim=-1, norm='ortho')
    tf_spec = torch.fft.rfft(tf, dim=-1, norm='ortho')
    spec = env_spec * tf_spec
    final = torch.fft.irfft(spec, dim=-1, norm='ortho')
    return final[..., :exp.n_samples]


class EnvelopeGenerator(nn.Module):
    def __init__(self, n_envelopes, n_frames, n_samples):
        super().__init__()
        self.n_envelopes = n_envelopes
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.to_env_selection = nn.Linear(exp.model_dim, n_envelopes)
        self.env = nn.Parameter(torch.zeros(n_envelopes, n_frames).uniform_(0, 0.02))
        self.up = FFTUpsampleBlock(1, 1, size=n_frames, factor=n_samples // n_frames)
    
    def forward(self, x):
        x = self.to_env_selection(x)
        x = exp_softmax(x)

        env = x @ self.env[None, ...]

        env = torch.relu(env)
        env = max_norm(env)
        orig_env = env

        env = env.view(-1, 1, self.n_frames)
        env = self.up(env)
        return env, orig_env

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

        # self.env = nn.Sequential(
        #     nn.Linear(event_latent_dim, exp.model_dim * 4),
        #     Reshape((exp.model_dim, 4)),

        #     nn.ConvTranspose1d(exp.model_dim, 64, 4, 2, 1),
        #     nn.LeakyReLU(0.2),

        #     nn.ConvTranspose1d(64, 32, 4, 2, 1),
        #     nn.LeakyReLU(0.2),

        #     nn.ConvTranspose1d(32, 16, 4, 2, 1),
        #     nn.LeakyReLU(0.2),

        #     nn.ConvTranspose1d(16, 8, 4, 2, 1),
        #     nn.LeakyReLU(0.2),

        #     nn.ConvTranspose1d(8, 4, 4, 2, 1),
        #     nn.LeakyReLU(0.2),

        #     nn.Conv1d(4, 1, 3, 1, 1)
        # )
        
        # self.env = EnvelopeGenerator(16, exp.n_frames * 2, exp.n_samples)

        self.n_coeffs = 257

        self.model_dim = exp.model_dim
        self.n_samples = exp.n_samples
        self.n_frames = exp.n_frames
        self.window_size = 512

        self.n_inflections = 1

        
        self.resolution = 64

        
        self.is_continuous = False

        # TODO: Use some kind of normalization
        # here to keep this from exploding when using gumbel
        # softmax
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
            softmax_func=exp_softmax,
            # softmax_func=lambda x: F.gumbel_softmax(torch.exp(x), dim=-1, hard=True),
            # softmax_func=lambda x: torch.softmax(x, dim=-1),
            is_continuous=self.is_continuous,
            resonance_exp=2)
        

        self.noise_factor = nn.Parameter(torch.zeros(1).fill_(1e5))

        self.amp = LinearOutputStack(exp.model_dim, 2, out_channels=1)


    def forward(self, time, transfer):
        transfer = transfer.view(-1, event_latent_dim)
        # time = time.view(-1, event_latent_dim)

        # x = x.view(-1, self.model_dim)
        batch = transfer.shape[0]

        # create envelope
        env = torch.sigmoid(self.env(transfer))
        mx, _ = torch.max(env, dim=-1, keepdim=True)
        env = env / (mx + 1e-8)
        # env = env ** 2

        orig_env = env
        env = F.interpolate(env, size=self.n_samples, mode='linear')
        
        noise = torch.zeros(1, 1, self.n_samples, device=env.device).uniform_(-1, 1) * self.noise_factor
        env = env * noise

        loss = 0
        tf = self.transfer.forward(transfer)

        # KLUDGE: Keep exp() in transfer function from exploding
        tf = tf.permute(0, 2, 1)
        if not self.is_continuous:
            tf = tf.view(batch, scale.n_bands, self.resolution)
        tf = self.tf.forward(tf)
        orig_tf = None


        final = fft_convolve(env, tf)

        amp = torch.sigmoid(self.amp(transfer)) #** 2
        final = final * amp


        final = torch.mean(final, dim=1, keepdim=True)

        return final, orig_env, loss, orig_tf


class BestFitScheduler(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, time_latent, stems, targets):
        fm = fft_convolve(stems, targets)
        values, best = torch.max(fm, dim=-1, keepdim=True)
        sparse = torch.zeros_like(fm)
        sparse = torch.scatter(sparse, dim=-1, index=best, src=values)

        scheduled = fft_convolve(sparse, stems)
        return scheduled



class PosEncodedScheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_pos = nn.Linear(exp.model_dim, 33)
        self.pos = PosEncodedImpulseGenerator(
            exp.n_frames, 
            exp.n_samples, 
            softmax=exp_softmax)
    
    def forward(self, time_latent, stems, targets):
        time_latent = self.to_pos(time_latent)
        sparse, _ = self.pos.forward(time_latent)
        sparse = sparse.view(-1, n_events, exp.n_samples)
        scheduled = fft_convolve(sparse, stems)
        return scheduled


class ImpulseScheduler(nn.Module):
    def __init__(self):
        super().__init__()
        # self.to_pos = ConvUpsample(
        #     exp.model_dim, 
        #     exp.model_dim, 
        #     4, 
        #     exp.n_frames, 
        #     mode='learned', 
        #     out_channels=1)
        
        self.to_pos = nn.Linear(exp.model_dim, exp.n_frames)
        
        self.imp = ImpulseGenerator(
            exp.n_samples, 
            softmax=exp_softmax
        )
        
    def forward(self, time_latent, stems, targets):
        time_latent = self.to_pos(time_latent).view(-1, exp.n_frames)
        sparse  = self.imp.forward(time_latent)
        sparse = sparse.view(-1, n_events, exp.n_samples)
        scheduled = fft_convolve(sparse, stems)
        return scheduled


class FFTShiftScheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_pos = nn.Linear(exp.model_dim, 1)
        
    
    def forward(self, time_latent, stems, targets):
        time_latent = self.to_pos(time_latent)
        time_latent = torch.sigmoid(time_latent)
        scheduled = fft_shift(stems, time_latent)
        return scheduled


class AtomPlacementScheduler(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_pos = nn.Linear(exp.model_dim, 1)
        
    
    def forward(self, time_latent, stems, targets):
        time_latent = self.to_pos(time_latent)
        time_latent = torch.sigmoid(time_latent)
        scheduled = schedule_atoms(stems, time_latent.view(-1, n_events), targets)
        return scheduled


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
            exp.model_dim, 1, out_channels=128)

        self.scheduler = ImpulseScheduler()

        self.to_transfer = LinearOutputStack(
            exp.model_dim, 1, out_channels=event_latent_dim)
        self.transfer_vq = VQ(exp.model_dim, 2048, commitment_weight=0.1, passthrough=True)

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
        orig_time = time = self.to_time(x).view(-1, 128)
        t_loss = 0

        orig_transfer = transfer = self.to_transfer(x).view(-1, event_latent_dim)
        transfer, tf_indices, tf_loss = self.transfer_vq(transfer)


        x, env, loss, tf = self.decode.forward(time, transfer)
        x = x.view(batch, n_events, exp.n_samples)


        # x = schedule_atoms(x, time.view(-1, n_events), target)

        # x = fft_convolve(x, impulse)

        x = self.scheduler.forward(time, x, target)

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
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon, indices, encoded, env, vq_loss, tf, time, transfer, orig_time, orig_transfer = model.forward(batch)

    # transfer latent should be from a standard normal distribution
    ll = (latent_loss(orig_transfer)) * 0.5

    # envelope centroid loss
    rng = torch.linspace(0, 1, env.shape[-1], device=env.device)[None, None, :]
    env_centroid = torch.sum(rng * env, dim=-1) / torch.sum(rng, dim=-1)
    env_loss = env_centroid.mean()

    peripheral_loss = vq_loss + ll + env_loss
    pl = exp.perceptual_loss(recon, batch)

    loss = pl + peripheral_loss
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
        with torch.no_grad():
            t = exp_softmax(self.time_latent)
            return t.data.cpu().numpy().squeeze()

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
