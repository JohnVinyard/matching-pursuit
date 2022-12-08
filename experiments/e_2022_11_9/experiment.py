import torch
from torch import nn
from torch.distributions import Normal
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm, unit_norm
from modules.phase import AudioCodec, MelScale
from modules.pos_encode import ExpandUsingPosEncodings, pos_encoded, two_d_pos_encode
from modules.psychoacoustic import PsychoacousticFeature
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.stft import stft
from modules.transformer import Transformer
from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import device, playable
from util.music import MusicalScale
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
n_events = 16
n_harmonics = 32
samples_per_frame = 256
min_center_freq = 20
max_center_freq = 4000

n_f0_steps = len(exp.scale)

resonance_steps = n_harmonics
precompute_resonance = True
learned_envelopes = False

# it'd be nice to summarize the harmonics/resonance spectrogram...somehow
# harmonics, f0, impulse_loc, impulse_std, bandwidth_loc, bandwidth_std, amplitude
params_per_event = (n_harmonics * resonance_steps) + \
    5 + n_f0_steps if precompute_resonance else n_harmonics + 5 + n_f0_steps

resonance_baseline = 0.75
noise_coeff = 1

transformer_encoder = False
collapse_latent = False


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
    # return (F.hardtanh(x) + 1) * 0.5


learning_rate = 1e-4


def softmax(x):
    # return torch.softmax(x, dim=-1)
    return F.gumbel_softmax(x, dim=-1, hard=True)

# #########################################################################


mel_scale = MelScale()
codec = AudioCodec(mel_scale)


class Window(nn.Module):
    def __init__(self, n_samples, mn, mx, epsilon=1e-8, padding=0):
        super().__init__()
        self.n_samples = n_samples
        self.mn = mn
        self.mx = mx
        self.scale = self.mx - self.mn
        self.epsilon = epsilon
        self.padding = padding

        self.up = ConvUpsample(
            2, exp.model_dim, 8, 128, mode='nearest', out_channels=1)


    def forward(self, means, stds):

        if learned_envelopes:
            x = torch.cat([means, stds], dim=-1).view(-1, 2)
            x = unit_norm(x)
            x = self.up(x)

            up_size = self.n_samples - self.padding
            x = F.interpolate(x, size=up_size, mode='linear')
            if self.padding > 0:
                x = F.pad(x, (0, self.padding))
            x = x.view(-1, n_events, self.n_samples)
            x = x ** 2
            x = max_norm(x)
            return x
        else:
            dist = Normal(self.mn + (means * self.scale), self.epsilon + stds)
            rng = torch.linspace(0, 1, self.n_samples, device=means.device)[
                None, None, :]
            windows = torch.exp(dist.log_prob(rng))
            windows = max_norm(windows)
            return windows

        


class Resonance(nn.Module):
    def __init__(
            self,
            n_samples,
            samples_per_frame,
            factors,
            samplerate,
            min_freq_hz,
            max_freq_hz,
            precomputed=False):

        super().__init__()
        self.precomputed = precomputed

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

        if self.precomputed:
            resonances = torch.linspace(resonance_baseline, 0.999, resonance_steps)\
                .view(resonance_steps, 1).repeat(1, exp.n_frames)
            resonances = torch.cumprod(resonances, dim=-1)
            self.register_buffer('resonance', resonances)

    def forward(self, f0, res):
        batch, n_events, _ = f0.shape
        # batch, n_events, n_freqs = res.shape

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
        if self.precomputed:
            res = softmax(res)
            res = res @ self.resonance
            res = res.view(-1, 1, exp.n_frames)
        else:
            res = res[..., None].repeat(1, 1, 1, self.n_frames)
            res = torch.exp(torch.cumsum(torch.log(res + 1e-12), dim=-1))
            # res = torch.cumprod(res, dim=-1).view(-1, 1, self.n_frames)
            res = res.view(-1, 1, self.n_frames)

        res = F.interpolate(res, size=self.n_samples, mode='linear')

        # generate resonances
        final = res * torch.sin(torch.cumsum(freqs * 2 * np.pi, dim=-1))
        final = final.view(batch, n_events, -1, self.n_samples)
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
                             self.min_center_freq, self.max_center_freq, padding=1)

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
            max_freq_hz=max_center_freq,
            precomputed=precompute_resonance)

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)
        self.n_rooms = self.verb.n_rooms

        if transformer_encoder:
            # encoder = nn.TransformerEncoderLayer(
            #     exp.model_dim, 4, exp.model_dim, batch_first=True)
            # self.context = nn.TransformerEncoder(encoder, 4)
            self.context = Transformer(exp.model_dim, 5)
        else:
            self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])

        self.reduce = nn.Conv1d(exp.scale.n_bands + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(
            exp.model_dim, 2, out_channels=self.n_rooms)

        scale = MusicalScale()
        self.register_buffer('f0s', torch.from_numpy(
            np.array(list(scale.center_frequencies)) / exp.samplerate.nyquist).float())

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
                LinearOutputStack(exp.model_dim, 4,
                                  out_channels=params_per_event)
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

        if transformer_encoder:
            x = x.permute(0, 2, 1)
            x = self.context(x)
            x = x.permute(0, 2, 1)
        else:
            x = self.context(x)

        x = self.norm(x)

        x, indices = self.sparse(x)
        verb_params, _ = torch.max(x, dim=1)
        collapsed_latent = verb_params

        # expand to params
        mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
        rooms = torch.softmax(self.to_room(verb_params), dim=-1)

        verb_params = torch.cat(
            [mx.view(-1, 1), rooms.view(-1, self.n_rooms)], dim=-1)

        if collapse_latent:
            event_params = self.to_event_params(collapsed_latent).permute(0, 2, 1)
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

        freq_means = event_params[:, :, 0].view(batch, self.n_events, 1)
        freq_stds = event_params[:, :, 1].view(
            batch, self.n_events, 1)
        
        if not learned_envelopes:
            freq_means = (freq_means * 2) - 0.5

        time_means = event_params[:, :, 2].view(batch, self.n_events, 1)
        time_stds = event_params[:, :, 3].view(batch, self.n_events, 1)

        if not learned_envelopes:
            time_means = (time_means * 2) - 0.5
            time_stds = time_stds * 0.1

        if n_f0_steps == 1:
            f0 = event_params[:, :, 4:4 + n_f0_steps].view(batch, self.n_events, n_f0_steps) ** 2
        else:
            f0 = event_params[:, :, 4:4 + n_f0_steps].view(batch, self.n_events, n_f0_steps)
            f0 = softmax(f0)
            f0 = (f0 @ self.f0s).view(batch, n_events, 1)

        amps = event_params[:, :, 4 + n_f0_steps].view(batch, self.n_events, 1) ** 2

        if precompute_resonance:
            res = event_params[:, :, (5 + n_f0_steps):].view(
                batch, self.n_events, self.n_harmonics, resonance_steps)
        else:
            res = (event_params[:, :, (5 + n_f0_steps):].view(
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
optim = optimizer(model, lr=learning_rate)


def train_direct(batch, iteration):
    optim.zero_grad()
    recon, latent, params = model.forward(batch)

    # real_spec = stft(batch, 512, 256, pad=True, log_amplitude=False)
    # fake_spec = stft(recon, 512, 256, pad=True, log_amplitude=False)
    # loss = F.mse_loss(fake_spec, real_spec)

    # fake_spec = torch.fft.rfft(recon, dim=-1, norm='ortho')
    # real_spec = torch.fft.rfft(batch, dim=-1, norm='ortho')
    # loss = loss + F.mse_loss(torch.abs(fake_spec), torch.abs(real_spec))

    loss = exp.perceptual_loss(recon, batch, norm='l1')

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
        sp = codec.to_frequency_domain(
            self.win[0, 0].view(-1, exp.n_samples))
        return sp.data.cpu().numpy()[0, ..., 0]

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

            w, latent, loss = train_direct(item, i)
            self.win = w
            self.latent = latent
            print(i, 'D', loss.item())
