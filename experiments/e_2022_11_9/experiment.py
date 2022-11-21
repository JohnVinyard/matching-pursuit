import torch
from torch import nn
from torch.distributions import Normal
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded, two_d_pos_encode
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.stft import stft
from modules.transformer import Transformer
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import device, playable
from util.readmedocs import readme
import numpy as np
from torch.nn import functional as F
from modules.reverb import NeuralReverb
from config import Config

from util.weight_init import make_initializer

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

init_weights = make_initializer(0.1)


# Experiment Params ########################################################
n_events = 32
n_harmonics = 16
samples_per_frame = 256
min_center_freq = 20
max_center_freq = 4000

# it'd be nice to summarize the harmonics/resonance spectrogram...somehow
# harmonics, f0, impulse_loc, impulse_std, bandwidth_loc, bandwidth_std, amplitude
params_per_event = n_harmonics + 6

resonance_baseline = 0.8
noise_coeff = 1
render_type = 'nerf'
two_d_nerf = False

train_generator = False
should_train_renderer = False
train_true = True
train_encoder = False
patch_size = (4, 4)

transformer_encoder = False

collapse_latent = True

def soft_clamp(x):
    x_backward = x
    x_forward = torch.clamp(x_backward, 0, 1)
    y = x_backward + (x_forward - x_backward).detach()
    return y

def activation(x):
    # return torch.sigmoid(x)
    return (torch.sin(x) + 1) * 0.5
    # return torch.clamp(x, 0, 1)
    # return soft_clamp(x)

# #########################################################################



mel_scale = MelScale()
codec = AudioCodec(mel_scale)


class Window(nn.Module):
    def __init__(self, n_samples, mn, mx, epsilon=1e-8):
        super().__init__()
        self.n_samples = n_samples
        self.mn = mn
        self.mx = mx
        self.scale = self.mx - self.mn
        self.epsilon = epsilon

    def forward(self, means, stds):
        dist = Normal(self.mn + (means * self.scale), self.epsilon + stds)
        rng = torch.linspace(0, 1, self.n_samples, device=means.device)[
            None, None, :]
        windows = torch.exp(dist.log_prob(rng))
        windows = max_norm(windows, dim=-1)
        return windows


class Resonance(nn.Module):
    def __init__(self, n_samples, samples_per_frame, factors, samplerate, min_freq_hz, max_freq_hz):
        super().__init__()
        self.n_samples = n_samples
        self.register_buffer('factors', factors)
        self.n_freqs = factors.shape[0]
        self.samples_per_frame = samples_per_frame
        self.n_frames = n_samples // samples_per_frame

        self.samplerate = samplerate
        self.nyquist = self.samplerate // 2
        self.min_freq = min_freq_hz / self.nyquist
        self.max_freq = max_freq_hz / self.nyquist
        self.freq_scale = self.max_freq - self.min_freq

    def forward(self, f0, res):
        batch, n_events, _ = f0.shape
        batch, n_events, n_freqs = res.shape

        # first, we need freq values for all harmonics
        f0 = self.min_freq + (self.freq_scale * f0)
        freqs = f0 * self.factors[None, None, :]
        freqs = freqs[..., None].repeat(
            1, 1, 1, self.n_frames).view(-1, 1, self.n_frames)

        # zero out anything above the nyquist frequency
        indices = torch.where(freqs >= 1)
        freqs[indices] = 0

        freqs = F.interpolate(freqs, size=self.n_samples, mode='linear')

        # we also need resonance values for each harmonic
        res = res[..., None].repeat(1, 1, 1, self.n_frames)
        res = torch.cumprod(res, dim=-1).view(-1, 1, self.n_frames)
        res = F.interpolate(res, size=self.n_samples, mode='linear')

        # generate resonances
        final = res * torch.sin(torch.cumsum(freqs * 2 * np.pi, dim=-1))
        final = final.view(batch, n_events, n_freqs, self.n_samples)
        final = torch.sum(final, dim=2)
        return final


class BandLimitedNoise(nn.Module):
    def __init__(self, n_samples, samplerate, min_center_freq_hz=20, max_center_freq_hz=4000):
        super().__init__()
        self.n_samples = n_samples
        self.samplerate = samplerate
        self.nyquist = self.samplerate // 2
        self.min_center_freq = min_center_freq_hz / self.nyquist
        self.max_center_freq = max_center_freq_hz / self.nyquist
        self.window = Window(n_samples // 2 + 1,
                             self.min_center_freq, self.max_center_freq)

    def forward(self, center_frequencies, bandwidths):
        windows = self.window.forward(center_frequencies, bandwidths)
        n = torch.zeros(1, 1, self.n_samples,
                        device=center_frequencies.device).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(n, dim=-1, norm='ortho')
        # filter noise in the frequency domain
        filtered = noise_spec * windows
        # invert
        band_limited_noise = torch.fft.irfft(filtered, dim=-1, norm='ortho')
        return band_limited_noise

class NerfEventRenderer(nn.Module):
    def __init__(self, latent_dim, n_frames, freq_bins, n_events, n_rooms, patch_size):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_frames = n_frames
        self.freq_bins = freq_bins
        self.n_events = n_events
        self.n_rooms = n_rooms
        self.patch_size = patch_size

        width, height = self.patch_size

        self.patch_frames = n_frames // width
        self.patch_bins = freq_bins // height

        self.two_d = two_d_nerf

        if self.two_d:
            self.register_buffer(
                'grid', two_d_pos_encode(self.patch_frames, self.patch_bins, device))
            self.expand_pos_encoding = nn.Linear(34, 128)
            self.net = LinearOutputStack(
                exp.model_dim, layers=3, out_channels=np.prod(self.patch_size), in_channels=exp.model_dim)
        else:
            self.pos = PosEncodedUpsample(
                exp.model_dim, 
                exp.model_dim, 
                size=n_frames, 
                out_channels=self.freq_bins, 
                layers=4, 
                concat=False)


            
    def forward(self, x):
        if self.two_d:
            grid = self.grid[None, ...].permute(0, 2, 3, 1)
            grid = self.expand_pos_encoding(grid)
            x = x[..., None, None].permute(0, 2, 3, 1)
            x = x + grid
            x = self.net(x)
            x = x\
                .view(-1, self.patch_frames, self.patch_bins, self.patch_size[0], self.patch_size[1])\
                .permute(0, 2, 4, 1, 3)\
                .reshape(-1, self.freq_bins, self.n_frames)
            return x
        else:
            x = self.pos(x)
            return x

class Renderer(nn.Module):
    """
    The renderer is responsible for taking a batch of events and global context vectors
    and producing a spectrogram rendering of each event
    """

    def __init__(
            self,
            latent_dim,
            n_frames,
            n_freq_bins,
            n_events,
            n_rooms,
            render_type='1d'):

        super().__init__()
        self.n_frames = n_frames
        self.n_freq_bins = n_freq_bins
        self.latent_dim = latent_dim
        self.n_events = n_events
        self.reduce = LinearOutputStack(
            exp.model_dim,
            2,
            out_channels=exp.model_dim,
            in_channels=n_rooms + 1 + params_per_event)

        if render_type == '1d':
            self.net = ConvUpsample(
                exp.model_dim,
                exp.model_dim,
                4,
                end_size=n_frames,
                mode='learned',
                out_channels=n_freq_bins)
        elif render_type == '2d':
            self.net = nn.Sequential(
                nn.Linear(exp.model_dim, 4 * exp.model_dim),
                Reshape((exp.model_dim, 2, 2)),
                nn.ConvTranspose2d(exp.model_dim, 64, 4, 2, 1),  # 4
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 8
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(32, 16, 4, 2, 1),  # 16
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(16, 8, 4, 2, 1),  # 32
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(8, 4, 4, 2, 1),  # 64
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(4, 4, 4, 2, 1),  # 128
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(4, 1, (4, 3), (2, 1), (1, 1)),  # 256
            )
        elif render_type == 'nerf':
            self.net = NerfEventRenderer(
                latent_dim, n_frames, n_freq_bins, n_events, n_rooms, patch_size=patch_size)
        else:
            raise ValueError(f'unknown render type {render_type}')

        self.global_context_size = n_rooms + 1

        self.apply(init_weights)

    def forward(self, event, context):

        context = context\
            .view(-1, 1, self.global_context_size)\
            .repeat(1, self.n_events, 1)\
            .view(-1, self.global_context_size)

        x = torch.cat([event, context], dim=-1)
        x = self.reduce(x)
        specs = self.net(x)
        specs = specs.view(-1, n_events, self.n_freq_bins, self.n_frames)
        specs = torch.sum(specs, dim=1)
        specs = specs.permute(0, 2, 1)
        # we're producing magnitudes, so positive values only
        specs = specs ** 2
        return specs

def generate_random_synth_params(batch, n_rooms, device):

    res_baseline = resonance_baseline
    res_span = 1 - res_baseline

    event_params = torch.zeros(batch, n_events, params_per_event, device=device).uniform_(0, 1)

    freq_means = event_params[:, :, 0].view(batch, n_events, 1) ** 2
    freq_stds = event_params[:, :, 1].view(batch, n_events, 1) * 0.1

    time_means = event_params[:, :, 2].view(batch,n_events, 1)
    time_stds = event_params[:, :, 3].view(batch, n_events, 1) * 0.1

    f0 = event_params[:, :, 4].view(batch, n_events, 1) ** 2
    amps = event_params[:, :, 5].view(batch, n_events, 1) ** 2

    res = (event_params[:, :, 6:].view(
        batch, n_events, n_harmonics) * res_span) + res_baseline
    

    mx = torch.zeros(batch, 1, device=device).uniform_(0, 1)
    rooms = torch.softmax(torch.zeros(batch, n_rooms, device=device).normal_(0, 1), dim=-1)
    
    return freq_means, freq_stds, time_means, time_stds, f0, amps, res, mx, rooms


class Model(nn.Module):
    def __init__(self, n_samples, n_events, n_harmonics, samples_per_frame):
        super().__init__()
        self.n_samples = n_samples
        self.n_events = n_events
        self.n_harmonics = n_harmonics
        self.samples_per_frame = samples_per_frame
        self.impulses = Window(n_samples, 0, 1)
        self.noise = BandLimitedNoise(n_samples, int(exp.samplerate))

        self.resonance = Resonance(
            n_samples,
            samples_per_frame=samples_per_frame,
            factors=torch.arange(1, n_harmonics + 1, device=device),
            samplerate=int(exp.samplerate),
            min_freq_hz=min_center_freq,
            max_freq_hz=max_center_freq)

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)
        self.n_rooms = self.verb.n_rooms

        if transformer_encoder:
            encoder = nn.TransformerEncoderLayer(exp.model_dim, 4, exp.model_dim, batch_first=True)
            self.context = nn.TransformerEncoder(encoder, 4)
            # self.context = Transformer(exp.model_dim, 5)
        else:
            self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])
        
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(
            exp.model_dim, 2, out_channels=self.n_rooms)

        if collapse_latent:
            self.to_event_params = PosEncodedUpsample(
                        exp.model_dim, 
                        exp.model_dim, 
                        size=n_events, 
                        out_channels=params_per_event, 
                        layers=4,
                        concat=True, 
                        multiply=False, 
                        learnable_encodings=False)
        else:
            self.to_event_params = nn.Sequential(
                LinearOutputStack(exp.model_dim, 2, out_channels=params_per_event)
            )

        
        self.apply(init_weights)
    

    def forward(self, x, noise_mix=0, return_encodings=False):

        # encode
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

        if train_encoder:
            x = x.permute(0, 2, 1)
            x = self.context(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.context(x)
        
        x = self.norm(x)

        x, indices = self.sparse(x)

        verb_params, _ = torch.max(x, dim=1)

        # expand to params
        mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
        rooms = torch.softmax(self.to_room(verb_params), dim=-1)

        verb_params = torch.cat(
            [mx.view(-1, 1), rooms.view(-1, self.n_rooms)], dim=-1)

        if collapse_latent:
            event_params = self.to_event_params(x).permute(0, 2, 1)
        else:
            event_params = self.to_event_params.forward(x)
        
        event_params = event_params.view(-1, n_events, params_per_event)

        if noise_mix == 0:
            event_params = activation(event_params)
        else:
            noise = torch.zeros_like(
                event_params, device=x.device).uniform_(0, 1)
            actual_amt = 1 - noise_mix
            event_params = (noise * noise_mix) + \
                (actual_amt * activation(event_params))

        res_baseline = resonance_baseline
        res_span = 1 - res_baseline


        freq_means = event_params[:, :, 0].view(batch, self.n_events, 1) ** 2
        freq_stds = event_params[:, :, 1].view(batch, self.n_events, 1) * 0.1

        time_means = event_params[:, :, 2].view(batch, self.n_events, 1)
        time_stds = event_params[:, :, 3].view(batch, self.n_events, 1) * 0.1

        f0 = event_params[:, :, 4].view(batch, self.n_events, 1) ** 2
        amps = event_params[:, :, 5].view(batch, self.n_events, 1) ** 2

        res = (event_params[:, :, 6:].view(
            batch, self.n_events, self.n_harmonics) * res_span) + res_baseline
        
        if return_encodings:
            return freq_means, freq_stds, time_means, time_stds, f0, res, amps, mx, rooms

        located = self.synthesize(
            freq_means, 
            freq_stds, 
            time_means, 
            time_stds, 
            f0, 
            res, 
            amps, 
            mx, 
            rooms)

        return located, verb_params, event_params
    
    def synthesize(self, freq_means, freq_stds, time_means, time_stds, f0, res, amps, mx, rooms):

        mx = mx.view(-1, 1, 1)

        # gradients do not flow through the following operations
        windows = self.noise.forward(freq_means, freq_stds)
        events = self.impulses.forward(time_means, time_stds)

        resonances = self.resonance.forward(f0, res)

        # locate events in time and scale by amplitude
        located = windows * events * amps

        # convolve impulses with resonances
        located = fft_convolve(located, resonances) + (located * noise_coeff)

        located = torch.sum(located, dim=1, keepdim=True)

        dry = located
        wet = self.verb.forward(located, rooms)

        located = (dry * (1 - mx)) + (wet * mx)

        return located


model = Model(
    exp.n_samples,
    n_events,
    n_harmonics,
    samples_per_frame).to(device)
optim = optimizer(model, lr=1e-3)

render = Renderer(
    exp.model_dim,
    n_frames=128,
    n_freq_bins=256,
    n_events=n_events,
    n_rooms=model.n_rooms,
    render_type=render_type).to(device)
render_optim = optimizer(render, lr=1e-3)


def iteration_to_noise_mix(iteration):
    if train_true:
        return 0
    else:
        value = 1 - (iteration * 1e-4)
        return max(value, 0)


def train_renderer(batch, iteration):
    render_optim.zero_grad()

    # noise_mix = iteration_to_noise_mix(iteration)

    with torch.no_grad():
        # recon, latent, params = model.forward(batch, noise_mix=noise_mix)
        # params = params.view(-1, params_per_event)
        freq_means, freq_stds, time_means, time_stds, f0, amps, res, mx, rooms = generate_random_synth_params(
            batch.shape[0], model.n_rooms, device=device)
        recon = model.synthesize(
            freq_means, freq_stds, time_means, time_stds, f0, res, amps, mx, rooms)        
        params = torch.cat([
            freq_means, 
            freq_stds, 
            time_means, 
            time_stds, 
            f0, 
            amps, 
            res, 
        ], dim=-1)
        latent = torch.cat(
            [mx.view(-1, 1), rooms.view(-1, model.n_rooms)], dim=-1)

    rendered = render.forward(params.view(-1, params_per_event), latent)

    with torch.no_grad():
        actual = codec.to_frequency_domain(
            recon.view(-1, exp.n_samples))[..., 0]

    loss = F.mse_loss(rendered, actual)
    loss.backward()
    render_optim.step()
    return loss, rendered, actual, recon


def train(batch, iteration):
    optim.zero_grad()
    recon, latent, params = model.forward(batch)
    real_spec = codec.to_frequency_domain(
        batch.view(-1, exp.n_samples))[..., 0]
    pred_spec = render.forward(params.view(-1, params_per_event), latent)
    loss = F.mse_loss(pred_spec, real_spec)
    loss.backward()
    optim.step()
    return recon, latent, loss

# def train_encodings(batch, iteration):
#     with torch.no_grad():
#         freq_means, freq_stds, time_means, time_stds, f0, amps, res, mx, rooms = generate_random_synth_params(
#             batch.shape[0], model.n_rooms, device=device)
#         random_audio = model.synthesize(
#             freq_means, freq_stds, time_means, time_stds, f0, res, amps, mx, rooms)        
#         real_packed = torch.cat([
#             freq_means.view(-1), 
#             freq_stds.view(-1), 
#             time_means.view(-1), 
#             time_stds.view(-1), 
#             f0.view(-1), 
#             res.view(-1), 
#             amps.view(-1), 
#             mx.view(-1), 
#             rooms.view(-1)])

    
#     freq_means, freq_stds, time_means, time_stds, f0, res, amps, mx, rooms = model.forward(random_audio, return_encodings=True)
#     fake_packed = torch.cat([
#             freq_means.view(-1), 
#             freq_stds.view(-1), 
#             time_means.view(-1), 
#             time_stds.view(-1), 
#             f0.view(-1), 
#             res.view(-1), 
#             amps.view(-1), 
#             mx.view(-1), 
#             rooms.view(-1)])
    
#     loss = F.mse_loss(fake_packed, real_packed)
#     loss.backward()

#     optim.step()

#     return loss

    

def train_direct(batch, iteration):
    optim.zero_grad()
    recon, latent, params = model.forward(batch)

    # real_spec = stft(batch, 512, 256, pad=True)
    # fake_spec = stft(recon, 512, 256, pad=True)
    # loss = F.mse_loss(fake_spec, real_spec)

    loss = exp.perceptual_loss(recon, batch)

    # fake_spec = torch.fft.rfft(recon, dim=-1, norm='ortho')
    # real_spec = torch.fft.rfft(batch, dim=-1, norm='ortho')
    # loss = F.mse_loss(torch.abs(fake_spec), torch.abs(real_spec))

    loss.backward()
    optim.step()
    return recon, latent, loss


@readme
class ResonantAtomsExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.win = None
        self.real = None
        self.latent = None
        self.rendered = None
        self.actual = None
        self.render_recon = None
        self.model = model

    def actual_spec(self):
        return self.actual.data.cpu().numpy()[0]

    def real_spec(self):
        with torch.no_grad():
            sp = codec.to_frequency_domain(self.real.view(-1, exp.n_samples))
            return sp.data.cpu().numpy()[0, ..., 0]

    def orig(self):
        return playable(self.real, exp.samplerate)

    def listen(self):
        return playable(self.win[0, 0], exp.samplerate)

    def fake_spec(self):
        if train_true:
            sp = codec.to_frequency_domain(self.win[0, 0].view(-1, exp.n_samples))
            return sp.data.cpu().numpy()[0, ..., 0]
        else:
            return self.rendered.data.cpu().numpy()[0]

    def z(self):
        return self.latent.data.cpu().numpy().squeeze()

    def rr(self):
        return playable(self.render_recon, exp.samplerate)

    def rr_spec(self):
        return np.abs(zounds.spectral.stft(self.rr()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            if train_true:
                w, latent, loss = train_direct(item, i)
                self.win = w
                self.latent = latent
                print(i, 'D', loss.item())
            
            if train_encoder:
                loss = train_encodings(item, i)
                print(i, 'E', loss.item())

            if train_generator:
                w, latent, loss = train(item, i)
                self.win = w
                self.latent = latent
                print(i, 'G', loss.item())

            if should_train_renderer:
                loss, rendered, actual, recon = train_renderer(item, i)
                self.render_recon = recon
                self.rendered = rendered
                self.actual = actual
                print(i, 'R', loss.item())
