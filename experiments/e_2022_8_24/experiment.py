import torch
from torch import nn
import zounds
from torch.nn import functional as F
from modules.phase import AudioCodec, MelScale, overlap_add
from modules.psychoacoustic import PsychoacousticFeature
from modules.scattering import MoreCorrectScattering
from modules.stft import stft
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify_vectors
from upsample import ConvUpsample
from util import device, playable
from util.readmedocs import readme
from util.weight_init import make_initializer
from train.optim import optimizer
import numpy as np

n_samples = 2 ** 15
samplerate = zounds.SR22050()
window_size = 512
step_size = window_size // 2
n_coeffs = window_size // 2 + 1
n_frames = n_samples // step_size

model_dim = 128

n_events = 16
resolution = 8

env_size = resolution
transfer_size = n_coeffs * 2 * resolution

event_dim = n_frames + (n_coeffs * 2)

n_bands = 128
kernel_size = 512

# TODO: Package this up into a class
band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)


init_weights = make_initializer(0.1)

pif = PsychoacousticFeature().to(device)

mel_scale = MelScale()
codec = AudioCodec(mel_scale)

long_window = 4096
long_step = long_window // 2
long_coeffs = long_window // 2 + 1
long_frames = n_samples // long_step

scatter_scale = zounds.MelScale(band, 32)
scatter = MoreCorrectScattering(samplerate, scatter_scale, 512, 0.1).to(device)

def perceptual_feature(x):
    # bands = pif.compute_feature_dict(x)
    # return torch.cat(list(bands.values()), dim=-1)
    return stft(
        x, window_size, step_size, pad=True, log_amplitude=True)
    # return scatter.forward(x)

    # spec = codec.to_frequency_domain(x.view(-1, n_samples))
    # return spec[..., 0]


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    return F.mse_loss(a, b)


class EventRenderer(object):
    """
    Take a batch of vectors representing energy/bang/actication and 
    a spectral transfer function and render to audio
    """

    def __init__(self):
        super().__init__()

    def render(self, _, env, transfer):
        batch = env.shape[0]

        # TODO: This could be sparse interpolation for much finer control
        # TODO: Energy gets harder and harder to apply?
        env = env.view(batch, -1, long_frames)

        amp = env
        noise = torch.zeros(batch, long_window, long_frames,
                            device=env.device).uniform_(-1, 1)
        energy = amp * noise

        transfer = transfer.view(batch, long_coeffs * 2, long_frames)
        coeffs = transfer

        # TODO: Figure out magnitude
        real = coeffs[:, :long_coeffs, :]
        imag = coeffs[:, long_coeffs:, :]

        real = torch.norm(transfer.view(batch, long_coeffs, 2, long_frames), dim=2)

        # ensure the transfer function is stable by 
        r = real.view(batch, -1)
        mx, _ = torch.max(r, dim=-1,keepdim=True)
        r = r / (mx + 1e-8)
        # real = 0.8 + (r.view(batch, long_coeffs, long_frames) * 0.1999)
        real = r.view(batch, long_coeffs, long_frames)

        imag = torch.angle(torch.complex(real, imag)) * np.pi
        tf = real * torch.exp(1j * imag)

        output_frames = []
        for i in range(long_frames):

            if len(output_frames):
                local_energy = output_frames[-1] + energy[:, :, i: i + 1]
            else:
                local_energy = energy[:, :, i: i + 1]

            spec = torch.fft.rfft(local_energy, dim=1, norm='ortho')
            spec = spec * tf[:, :, i: i + 1]
            new_frame = torch.fft.irfft(spec, dim=1, norm='ortho')

            output_frames.append(new_frame)

        output_frames = torch.cat(output_frames, dim=-1)
        output_frames = output_frames.view(
            batch, 1, long_window, long_frames).permute(0, 1, 3, 2)
        output = overlap_add(output_frames)[..., :n_samples]
        output = output.view(batch, 1, n_samples)
        return output


class AudioSegmentRenderer(object):
    def __init__(self):
        super().__init__()

    def render(self, x, params, indices):
        x = x.view(-1, n_events, n_samples)
        batch = x.shape[0]

        times = indices * step_size

        output = torch.zeros(batch, 1, n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(n_events):
                time = times[b, i]
                output[b, :, time: time + n_samples] += x[b, i][None, :]

        output = output[..., :n_samples]
        return output


# class DilatedBlock(nn.Module):
#     def __init__(self, channels, dilation):
#         super().__init__()
#         self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
#         self.conv = nn.Conv1d(channels, channels, 1, 1, 0)

#     def forward(self, x):
#         orig = x
#         x = self.dilated(x)
#         x = self.conv(x)
#         x = x + orig
#         return x

class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self, disc=False):
        super().__init__()
        self.disc = disc

        # self.context = nn.Sequential(
        #     DilatedBlock(model_dim, 1),
        #     DilatedBlock(model_dim, 3),
        #     DilatedBlock(model_dim, 9),
        #     DilatedBlock(model_dim, 1),
        # )

        encoder = nn.TransformerEncoderLayer(
            model_dim, 4, model_dim, batch_first=True, activation=F.gelu)
        self.context = nn.TransformerEncoder(
            encoder, 6, norm=None)

        self.reduce = nn.Conv1d(model_dim + 33, model_dim, 1, 1, 0)

        self.attend = nn.Linear(model_dim, 1)
        self.to_events = nn.Linear(model_dim, event_dim)

        self.env_factor = 32
        self.to_env = ConvUpsample(
            event_dim, model_dim, 8, n_frames * self.env_factor, mode='learned', out_channels=1)
        self.to_coeffs = ConvUpsample(
            event_dim, model_dim, 8, long_frames, mode='learned', out_channels=long_coeffs * 2)

        self.judge = nn.Linear(model_dim, 1)

    def forward(self, x, add_noise=False):
        batch = x.shape[0]
        x = x.view(-1, 1, n_samples)
        x = torch.abs(fb.convolve(x))
        x = fb.temporal_pooling(x, window_size, step_size)[..., :n_frames]
        pos = pos_encoded(batch, n_frames, 16,
                          device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        x = x.permute(0, 2, 1)
        x = self.context(x)
        if self.disc:
            x = self.judge(x)
            x = torch.sigmoid(x)
            return x

        attn = torch.softmax(self.attend(x).view(batch, n_frames), dim=-1)
        x = x.permute(0, 2, 1)
        x, indices = sparsify_vectors(x, attn, n_events, normalize=True)
        x = self.to_events(x)

        # print('STD', x.std())
        # norms = torch.norm(x, dim=-1, keepdim=True)
        # x = (x / (norms + 1e-8)) * 10


        x = x.view(-1, event_dim)
        env = self.to_env.forward(x).view(
            batch * n_events, 1, n_frames * self.env_factor) ** 2
        env = F.interpolate(env, size=n_samples, mode='linear')
    
        env = F.pad(env, (0, long_step))
        env = env\
            .unfold(-1, long_window, long_step) \
            * torch.hamming_window(long_window, device=env.device)[None, None, None, :]
        
        env = env\
            .view(batch * n_events, 1, long_frames, long_window)\
            .permute(0,3, 1, 2)\
            .view(batch * n_events, long_window, long_frames)

        tf = self.to_coeffs(x).view(batch * n_events, long_coeffs * 2, long_frames)

        return x, env, tf, indices


class Model(nn.Module):
    def __init__(self, disc=False):
        super().__init__()
        self.summary = Summarizer(disc=disc)
        self.atom_renderer = EventRenderer()
        self.audio_renderer = AudioSegmentRenderer()
        self.disc = disc
        self.apply(init_weights)

    def render(self, params, env, tf, indices):
        atoms = self.atom_renderer.render(params, env, tf)
        audio = self.audio_renderer.render(atoms, params, indices)
        return audio

    def forward(self, x, add_noise=False):
        if self.disc:
            return self.summary(x)
        else:
            x, env, tf, indices = self.summary(x, add_noise=add_noise)
            return x, env, tf, indices


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

# disc = Model(disc=True).to(device)
# disc_optim = optimizer(disc, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    params, env, tf, indices = model.forward(batch)
    recon = model.render(params, env, tf, indices)


    # env_loss = torch.abs(env).sum() * 0.01
    recon_loss = perceptual_loss(recon, batch) 

    loss = recon_loss
    loss.backward()
    optim.step()
    return env, loss, recon


# def train_disc(batch):
#     disc_optim.zero_grad()

#     params, env, tf, indices = model.forward(batch)
#     recon = model.render(params, env, tf, indices)

#     rj = disc.forward(batch)
#     fj = disc.forward(recon)

#     loss = torch.abs(1 - rj).mean() + torch.abs(0 - fj).mean()
#     loss.backward()
#     disc_optim.step()
#     return loss


@readme
class TransferFunctionReinforcementLearning(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.env = None

    def orig(self):
        return playable(self.real, samplerate, normalize=True)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, samplerate, normalize=True)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def e(self):
        return self.env.data.cpu().numpy().squeeze().sum(axis=1)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            std = item.std(axis=-1, keepdim=True)
            item = item / (std + 1e-4)

            self.real = item

            # if i % 2 == 0:
            self.env, loss, self.fake = train_model(item)
            print('GEN', loss.item())
            # else:
            # disc_loss = train_disc(item)
            # print('DISC', disc_loss.item())
