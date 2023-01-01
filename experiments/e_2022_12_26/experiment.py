import zounds
from config.dotenv import Config
from config.experiment import Experiment
import torch
from torch import nn
from torch.nn import functional as F
from fft_shift import fft_shift
from loss.least_squares import least_squares_disc_loss, least_squares_generator_loss
from loss.serial import matching_pursuit, serial_loss
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.physical import Window
from modules.reverb import NeuralReverb
from modules.sparse import SparseEncoderModel
from modules.stft import stft
from perceptual.feature import CochleaModel, NormalizedSpectrogram
from scalar_scheduling import pos_encoded
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample

from util.readmedocs import readme
from util import device, playable
import numpy as np

n_events = 8
window_size = 512
step_size = window_size // 2


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

min_freq = 30 / exp.samplerate.nyquist
max_freq = 3000 / exp.samplerate.nyquist
freq_span = max_freq - min_freq

discrete_freqs = torch.linspace(min_freq, max_freq, 128, device=device)

fade = torch.ones(1, 1, exp.n_samples, device=device)
fade[:, :, :10] = torch.linspace(0, 1, 10, device=device)
fade[:, :, -10:] = torch.linspace(1, 0, 10, device=device)

time_params = 2
f0_params = 1
f0_variance_params = exp.n_frames
n_harmonics = 64
n_noise_bands = n_harmonics

harmonics = torch.arange(1, n_harmonics + 1, device=device)
n_amp_params = (n_noise_bands + n_harmonics) * exp.n_frames
discrete_f0 = 128
discrete_freqs = torch.linspace(min_freq, max_freq, discrete_f0, device=device)
total_params = time_params + f0_params + f0_variance_params + n_amp_params + discrete_f0


def softmax(x):
    # x = max_norm(x, dim=-1)
    return F.gumbel_softmax(torch.exp(x), dim=-1, hard=True)
    # return torch.softmax(x, dim=-1)


def unit_activation(x):
    return torch.sigmoid(x)
    # return torch.clamp(x, 0, 1)
    # return (torch.sin(x) + 1) * 0.5

def unpack(x):
    means = x[..., 0:1]
    stds = x[..., 1:2]
    f0 = x[..., 2:3] ** 2
    f0_var = (x[..., 3:3 + 128] * 2) - 1
    amp_params = x[..., 3 + 128: 3 + 128 + n_amp_params] ** 2
    discrete_f0 = x[..., 3 + 128 + n_amp_params:]
    return means, stds, f0, f0_var, amp_params, discrete_f0


class Atoms(nn.Module):
    def __init__(self, scalar_position=False):
        super().__init__()
        self.window = Window(exp.n_samples, 0, 1)
        self.scalar_position = scalar_position

        encoder = nn.TransformerEncoderLayer(exp.model_dim, 4, exp.model_dim, batch_first=True)
        self.context = nn.TransformerEncoder(encoder, 4)


        self.switch = LinearOutputStack(exp.model_dim, 3, out_channels=2)

        self.f0 = LinearOutputStack(exp.model_dim, 3, out_channels=1)
        self.f0_var = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=8, end_size=exp.n_frames, mode='learned', out_channels=1)
        self.amp_params = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=8, end_size=exp.n_frames, mode='learned', out_channels=n_harmonics * 2
        )

        self.loc = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=8, end_size=exp.n_frames, mode='learned', out_channels=1
        )

        self.scalar_pos = LinearOutputStack(exp.model_dim, 3, out_channels=1)
    
    def forward(self, x):
        x = x.view(-1, n_events, exp.model_dim)
        x = self.context(x)
        x = x.reshape(-1, exp.model_dim)


        # x = x.view(-1, n_events, exp.model_dim) + self.base_events
        # x = x.view(-1, exp.model_dim)
        # means, stds, f0, f0_var, amp_params, discrete_f0 = unpack(x)

        # scalar = unit_activation(self.scalar_pos(x).view(-1, n_events, 1))

        sw = self.switch.forward(x)
        sw = softmax(sw)


        discrete_f0 = unit_activation(self.loc(x))
        f0 = unit_activation(self.f0(x)) ** 2
        f0_var = (unit_activation(self.f0_var(x)) * 2) - 1
        amp_params = self.amp_params(x) ** 2

        loc = softmax(discrete_f0).view(-1, n_events, 128)
        loc_full = torch.zeros(x.shape[0] // n_events, n_events, exp.n_samples, device=loc.device)
        step = exp.n_samples // 128
        loc_full[:, :, ::step] = loc

        # discrete_f0 = discrete_f0.reshape(-1, n_events, exp.n_frames)
        # discrete_f0 = torch.softmax(discrete_f0, dim=-1)
        # discrete_f0 = discrete_f0.view(-1, len(discrete_freqs))
        # f0 = discrete_f0 @ discrete_freqs
        
        f0 = min_freq + (f0 * freq_span)

        # TODO: What's a reasonable variance here?
        f0_span = f0 * 0.01
        f0 = f0.view(-1, 1, 1).repeat(1, 1, exp.n_frames)
        f0_change = f0_var.view(-1, 1, exp.n_frames) * f0_span.view(-1, 1, 1)
        f0 = f0 + f0_change

        harm = f0.view(-1, 1, exp.n_frames) * harmonics[None, :, None]

        # ensure we don't have any aliasing due to greater-than-nyquist frequencies
        indices = torch.where(harm > 1)
        harm[indices] = 0


        harm = harm * np.pi
        harm = F.interpolate(harm, size=exp.n_samples, mode='linear')
        harm = torch.sin(torch.cumsum(harm, dim=-1))

        noise = torch.zeros(x.shape[0], 1, exp.n_samples, device=device).uniform_(-1, 1)

        # TODO: Is this the right way to build FIR filters for these frequencies?
        noise_filters = harm[..., :128] * torch.hamming_window(128, device=device)[None, None, :]
        noise_filters = F.pad(noise_filters, (0, exp.n_samples - 128))
        noise_filter_spec = torch.fft.rfft(noise_filters, dim=-1, norm='ortho')
        noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')
        noise_bands = noise_spec * noise_filter_spec
        noise_bands = torch.fft.irfft(noise_bands, dim=-1, norm='ortho')
        noise_bands = max_norm(noise_bands, dim=-1)

        full = torch.cat([harm, noise_bands], dim=1)
        amp_params = F.interpolate(
            amp_params.view(-1, n_noise_bands + n_harmonics, exp.n_frames), size=exp.n_samples, mode='linear')
        
        x = full * amp_params

        x = x.view(-1, n_events, n_harmonics + n_noise_bands, exp.n_samples)
        x = torch.sum(x, dim=2)

        x = x * fade #* sw[..., 0:1]

        x = fft_convolve(x, loc_full)

        # x = fft_shift(x, scalar)[..., :exp.n_samples]

        return x


# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.net = PosEncodedUpsample(
#             exp.model_dim, 
#             exp.model_dim, 
#             size=n_events, 
#             out_channels=exp.model_dim, 
#             layers=6)
        
#         self.atoms = Atoms()

#         self.verb = NeuralReverb.from_directory(
#             Config.impulse_response_path(), exp.samplerate, exp.n_samples)

#         self.n_rooms = self.verb.n_rooms

#         self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
#         self.to_room = LinearOutputStack(
#             exp.model_dim, 2, out_channels=self.n_rooms)
        
#         self.apply(lambda p: exp.init_weights(p))
        
    
#     def forward(self, x):
#         orig_x = x = x.view(-1, exp.model_dim)
#         x = self.net(x)
#         x = self.atoms(x)
#         x = torch.sum(x, dim=1, keepdim=True)

#         # expand to params
#         mx = torch.sigmoid(self.to_mix(orig_x)).view(-1, 1, 1)
#         rm = torch.softmax(self.to_room(orig_x), dim=-1)

#         wet = self.verb.forward(x, torch.softmax(rm, dim=-1))
#         final = (mx * wet) + ((1 - mx) * x)
#         final = max_norm(final)
#         return final



# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hearing_model = CochleaModel(
#             exp.samplerate, 
#             zounds.MelScale(zounds.FrequencyBand(20, exp.samplerate.nyquist - 10), 128),
#             kernel_size=512)
        
#         self.norm = ExampleNorm()
        
#         self.n_frames = exp.n_frames
#         self.audio_feature = NormalizedSpectrogram(
#             pool_window=512, 
#             n_bins=128, 
#             loudness_gradations=256, 
#             embedding_dim=64, 
#             out_channels=128)
        
#         encoder = nn.TransformerEncoderLayer(
#             exp.model_dim, 4, exp.model_dim, batch_first=True)
#         self.context = nn.TransformerEncoder(encoder, 6)

#         self.reduce = nn.Conv1d(128 + 33, exp.model_dim, 1, 1, 0)

#         self.judge = nn.Conv1d(exp.model_dim, 1, 1, 1, 0)

#         self.apply(lambda p: exp.init_weights(p))
    
#     def forward(self, x):
#         batch = x.shape[0]

#         x = self.hearing_model.forward(x)
#         x = self.audio_feature.forward(x)

#         x = self.norm(x)

#         pos = pos_encoded(
#             batch, self.n_frames, 16, device=x.device).permute(0, 2, 1)

#         x = torch.cat([x, pos], dim=1)
#         x = self.reduce(x)

#         x = x.permute(0, 2, 1)
#         x = self.context(x)
#         x = x.permute(0, 2, 1)

#         x = self.judge(x)[:, :, -1]
#         x = torch.sigmoid(x)
#         return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SparseEncoderModel(
            Atoms(), 
            exp.samplerate, 
            exp.n_samples, 
            exp.model_dim, 
            exp.scale.n_bands, 
            n_events, 
            exp.model_dim, 
            exp.fb, 
            exp.scale,
            window_size,
            step_size,
            exp.n_frames,
            lambda x: x,
            collapse=False,
            transformer_context=True)
        
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x):
        x = self.encoder(x)
        # x = torch.sum(x, dim=1, keepdim=True)
        x = max_norm(x, dim=-1)
        return x

model = Model().to(device)
optim = optimizer(model, lr=1e-4)

# gen = Generator().to(device)
# gen_optim = optimizer(gen, lr=1e-4)

# disc = Discriminator().to(device)
# disc_optim = optimizer(disc, lr=1e-4)


# def train_gen(batch):
#     gen_optim.zero_grad()
#     latent = torch.zeros(batch.shape[0], exp.model_dim, device=device).normal_(0, 1)
#     x = gen.forward(latent)
#     fj = disc.forward(x)

#     # l = least_squares_generator_loss(fj)
#     l = torch.abs(0.9 - fj).sum()

#     l.backward()
#     gen_optim.step()
#     print('G', l.item())
#     return x

# def train_disc(batch):
#     disc_optim.zero_grad()
#     with torch.no_grad():
#         latent = torch.zeros(batch.shape[0], exp.model_dim, device=device).normal_(0, 1)
#         x = gen.forward(latent)

#     fj = disc.forward(x)
#     rj = disc.forward(batch)

#     # l = least_squares_disc_loss(rj, fj)

#     l = torch.abs(0.1 - fj).sum() + torch.abs(0.9 - rj).sum()

#     l.backward()
#     disc_optim.step()
#     print('D', l.item())

def contrast_normalized_stft(x):
    # x = stft(x, 512, 256, pad=True)
    x = exp.pooled_filter_bank(x)[:, None, :, :]
    
    unfold = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    uf = unfold.forward(x)
    norms = torch.norm(uf, dim=1, keepdim=True)
    normed = uf / (norms + 1e-8)
    # norms = norms.repeat(1, 9, 1)
    x = torch.cat([normed, norms], dim=1)
    return x

def experiment_loss(recon, batch):
    # recon = recon.sum(dim=1, keepdim=True)
    # loss = F.mse_loss(contrast_normalized_stft(recon), contrast_normalized_stft(batch))
    # return loss


    transform = lambda p: stft(p, 512, 256, pad=True)
    loss = serial_loss(recon, batch, transform)
    return loss



def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = experiment_loss(recon, batch)
    loss.backward()
    optim.step()
    with torch.no_grad():
        return loss, recon.sum(dim=1, keepdim=True)


@readme
class CompromiseExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None
        self.fake = None

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def orig(self):
        return playable(self.real, exp.samplerate)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            # if i % 2 == 0:
            #     self.fake = train_gen(item)
            # else:
            #     train_disc(item)

            l, recon = train(item)
            self.fake = recon
            print('R', i, l.item())
