import torch
from torch import nn
from torch.distributions import Normal
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import pos_encoded
from modules.sparse import VectorwiseSparsity
from modules.stft import stft
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

n_events = 32
n_harmonics = 16
samples_per_frame = 256
min_center_freq = 20
max_center_freq = 4000

# it'd be nice to summarize the harmonics/resonance spectrogram...somehow
# harmonics, f0, impulse_loc, impulse_std, bandwidth_loc, bandwidth_std, amplitude
params_per_event = n_harmonics + 6

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
        rng = torch.linspace(0, 1, self.n_samples, device=means.device)[None, None, :]
        windows = torch.exp(dist.log_prob(rng))
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
        freqs = freqs[..., None].repeat(1, 1, 1, self.n_frames).view(-1, 1, self.n_frames)
        freqs = F.interpolate(freqs, size=self.n_samples, mode='linear')

        # we also need resonance values for each harmonic        
        res = res[..., None].repeat(1, 1, 1, self.n_frames)
        res = torch.cumprod(res, dim=-1).view(-1, 1, self.n_frames)
        res = F.interpolate(res, size=self.n_samples, mode='linear')

        # generate resonances
        final = res * torch.sin(torch.cumsum(freqs * 2 * np.pi, dim=-1))
        final = final.view(batch, n_events, n_freqs, self.n_samples)
        final = torch.mean(final, dim=2)
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
        n = torch.zeros(1, 1, self.n_samples, device=center_frequencies.device).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(n, dim=-1, norm='ortho')
        # filter noise in the frequency domain
        filtered = noise_spec * windows
        # invert
        band_limited_noise = torch.fft.irfft(filtered, dim=-1, norm='ortho')
        return band_limited_noise

class Renderer(nn.Module):
    """
    The renderer is responsible for taking a batch of events and global context vectors
    and producing a spectrogram rendering of each event
    """
    def __init__(self, latent_dim, n_frames, n_freq_bins, n_events, n_rooms):
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

        self.net = ConvUpsample(
            exp.model_dim, 
            exp.model_dim, 
            4, 
            end_size=n_frames, 
            mode='learned', 
            out_channels=n_freq_bins)
        
        self.global_context_size = n_rooms + 1
        
        self.apply(init_weights)
    
    def forward(self, event, context):
        context = context.view(-1, 1, self.global_context_size).repeat(1, self.n_events, 1).view(-1, self.global_context_size)
        x = torch.cat([event, context], dim=-1)
        x = self.reduce(x)
        specs = self.net(x)
        specs = specs.view(-1, n_events, self.n_freq_bins, self.n_frames)
        specs = torch.mean(specs, dim=1)
        specs = specs.permute(0, 2, 1)
        # we're producing magnitudes, so positive values only
        specs = specs ** 2
        return specs



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
        
        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(exp.model_dim, 2, out_channels=self.n_rooms)

        self.to_event_params = nn.Sequential(
            # PosEncodedUpsample(
            #     exp.model_dim, 
            #     exp.model_dim, 
            #     size=n_events, 
            #     out_channels=params_per_event, 
            #     layers=4)

            LinearOutputStack(exp.model_dim, 2, out_channels=params_per_event)
        )

        # (n_harmonics + 6) * n_events

        # self.to_f0 = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events)
        # self.to_harmonics = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events * n_harmonics)

        # self.to_amplitudes = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events)

        # self.to_freq_means = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events)
        # self.to_freq_stds = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events)

        # self.to_time_means = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events)
        # self.to_time_stds = LinearOutputStack(
        #     exp.model_dim, 2, out_channels=n_events)

        self.apply(init_weights)

    def forward(self, x, noise_mix=0):

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
        x = self.context(x)
        x = self.norm(x)

        x, indices = self.sparse(x)
        
        # x, _ = torch.max(x, dim=-1)

        verb_params, _ = torch.max(x, dim=1)
        

        '''
        TODO: 
            - Expand from (batch, model_dim) to (batch, n_events, n_event_params)
            - optionally apply noise at this point
            - apply sigmoid and transform to event params
        '''

        # expand to params
        mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
        rooms = torch.softmax(self.to_room(verb_params), dim=-1)

        verb_params = torch.cat([mx.view(-1, 1), rooms.view(-1, self.n_rooms)], dim=-1)

        event_params = self.to_event_params.forward(x).view(-1, n_events, params_per_event)

        if noise_mix == 0:
            event_params = torch.sigmoid(event_params)
        else:
            noise = torch.zeros_like(event_params, device=x.device).uniform_(0, 1)
            actual_amt = 1 - noise_mix
            event_params = (noise * noise_mix) + (actual_amt * torch.sigmoid(event_params))

        
        res_baseline = 0.9
        res_span = 1 - res_baseline

        freq_means = event_params[:, :, 0].view(batch, self.n_events, 1)
        freq_stds = event_params[:, :, 1].view(batch, self.n_events, 1) * 0.1
        time_means = event_params[:, :, 2].view(batch, self.n_events, 1)
        time_stds = event_params[:, :, 3].view(batch, self.n_events, 1) * 0.1
        f0 = event_params[:, :, 4].view(batch, self.n_events, 1) ** 2
        amps = event_params[:, :, 5].view(batch, self.n_events, 1)
        res = (event_params[:, :, 6:].view(batch, self.n_events, self.n_harmonics) * res_span) + res_baseline


        # gradients do not flow through the following operations
        windows = self.noise.forward(freq_means, freq_stds)
        events = self.impulses.forward(time_means, time_stds)

        resonances = self.resonance.forward(f0, res)

        # locate events in time and scale by amplitude
        located = windows * events * amps

        # convolve impulses with resonances
        located = located + fft_convolve(located, resonances)

        located = torch.mean(located, dim=1, keepdim=True)

        dry = located
        wet = self.verb.forward(located, rooms)

        located = (dry * (1 - mx)) + (wet * mx)

        return located, verb_params, event_params

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
    n_rooms=model.n_rooms).to(device)
render_optim = optimizer(render, lr=1e-3)

def iteration_to_noise_mix(iteration):
    value = 1 - (iteration * 1e-6)
    return max(value, 0)
    # return 0

def train_renderer(batch, iteration):
    render_optim.zero_grad()

    noise_mix = iteration_to_noise_mix(iteration)

    with torch.no_grad():
        recon, latent, params = model.forward(batch, noise_mix=noise_mix)
        params = params.view(-1, params_per_event)
    
    rendered = render.forward(params, latent)

    with torch.no_grad():
        actual = codec.to_frequency_domain(recon.view(-1, exp.n_samples))[..., 0]
    
    loss = F.mse_loss(rendered, actual)
    loss.backward()
    render_optim.step()
    return loss, rendered, actual

def train(batch, iteration):
    optim.zero_grad()
    recon, latent, params = model.forward(batch)
    real_spec = codec.to_frequency_domain(batch.view(-1, exp.n_samples))[..., 0]
    pred_spec = render.forward(params.view(-1, params_per_event), latent)
    loss = F.mse_loss(pred_spec, real_spec)
    loss.backward()
    optim.step()
    return recon, latent, loss


def train_direct(batch, iteration):
    optim.zero_grad()
    recon, latent, params = model.forward(batch)

    # real_spec = codec.to_frequency_domain(batch.view(-1, exp.n_samples))[..., 0]
    # pred_spec = codec.to_frequency_domain(recon.view(-1, exp.n_samples))[..., 0]

    real_spec = stft(batch, 512, 256, pad=True)
    fake_spec = stft(recon, 512, 256, pad=True)

    # real_spec = exp.perceptual_feature(batch)
    # fake_spec = exp.perceptual_feature(recon)

    loss = F.mse_loss(fake_spec, real_spec)
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
        self.model = model

    def actual_spec(self):
        return self.actual.data.cpu().numpy()[0]

    def real_spec(self):
        with torch.no_grad():
            sp = codec.to_frequency_domain(self.real.view(-1, exp.n_samples))
            return sp.data.cpu().numpy()[0, ..., 0]

    def orig(self):
        return playable(self.real, exp.samplerate)
    
    # def fake_spec(self):
    #     sp = codec.to_frequency_domain(self.win[0, 0].view(-1, exp.n_samples))
    #     return sp.data.cpu().numpy()[0, ..., 0]

    def listen(self):
        return playable(self.win[0, 0], exp.samplerate)

    def fake_spec(self):
        return self.rendered.data.cpu().numpy()[0]
    
    def z(self):
        return self.latent.data.cpu().numpy().squeeze()

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            # w, latent, loss = train_direct(item, i)
            # self.win = w
            # self.latent = latent
            # print(i, 'D', loss.item())
        
            w, latent, loss = train(item, i)
            self.win = w
            self.latent = latent
            print(i, 'G', loss.item())

            loss, rendered, actual = train_renderer(item, i)
            self.rendered = rendered
            self.actual = actual
            print(i, 'R', loss.item())

