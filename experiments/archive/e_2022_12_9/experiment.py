from torch import Tensor, nn
from config.dotenv import Config
from config.experiment import Experiment
import zounds
import torch
import numpy as np
from torch.jit import ScriptModule, script_method
from torch.nn import functional as F
from modules.latent_loss import latent_loss
from modules.atoms import unit_norm
from modules.decompose import fft_frequency_decompose
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.physical import Window, harmonics
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.stft import stft

from train.optim import optimizer
from upsample import ConvUpsample, PosEncodedUpsample
from util import playable
from util.music import MusicalScale
from util.readmedocs import readme
from util import device
from random import choice

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

total_noise_coeffs = exp.n_samples // 2 + 1

min_f0_hz = 40
max_f0_hz = 3000

min_f0 = (min_f0_hz / exp.samplerate.nyquist)
max_f0 = (max_f0_hz / exp.samplerate.nyquist)
f0_span = max_f0 - min_f0


n_frames = 128
n_harmonics = 8
harmonic_factors = torch.arange(1, n_harmonics + 1, step=1, device=device)
freq_domain_filter_size = 16
n_events = 8

logged = {}

min_resonance = 0.5
res_span = 1 - min_resonance
loss_type = 'perceptual'
discrete_freqs = True
learning_rate = 1e-4
conv_mode = 'nearest'


def softmax(x):
    # return F.gumbel_softmax(x, dim=-1, hard=True)
    return torch.softmax(x, dim=-1)


# class MultiscaleLoss(nn.Module):
#     def __init__(self, kernel_size, n_bands):
#         super().__init__()
#         scale = zounds.LinearScale(
#             zounds.FrequencyBand(1, exp.samplerate.nyquist), n_bands, False)
#         self.fb = zounds.learn.FilterBank(
#             exp.samplerate, kernel_size, scale, 0.25, normalize_filters=True).to(device)

#     def _feature(self, x):
#         x = x.view(-1, 1, exp.n_samples)
#         bands = fft_frequency_decompose(x, 512)
#         output = []
#         for size, band in bands.items():
#             spec = self.fb.forward(band, normalize=False)
#             windowed = spec.unfold(-1, 128, 64)
#             pooled = windowed.mean(dim=-1)
#             windowed = torch.abs(torch.fft.rfft(
#                 windowed, dim=-1, norm='ortho'))
#             windowed = unit_norm(windowed, axis=-1)
#             output.append(pooled.view(-1))
#             output.append(windowed.view(-1))
#         return torch.cat(output)

#     def forward(self, a, b):
#         a = self._feature(a)
#         b = self._feature(b)
#         return torch.abs(a - b).sum()


def amp_activation(x):
    # return x
    # return torch.relu(x)
    return F.leaky_relu(x, 0.2)


def activation(x):
    return torch.sigmoid(x)
    # return (torch.sin(x) + 1) * 0.5


scale = MusicalScale()
frequencies = torch.from_numpy(np.array(
    list(scale.center_frequencies)) / exp.samplerate.nyquist).float().to(device)

param_sizes = {
    'f0': len(scale) if discrete_freqs else 1,
    'f0_fine': n_frames,
    'amp': n_frames,
    'harmonic_amps': n_harmonics,
    'harmonic_decay': n_harmonics,
    'freq_domain_envelope': freq_domain_filter_size,
}

total_synth_params = sum(param_sizes.values())


def build_param_slices():
    current = 0
    slices = {}
    for k, v in param_sizes.items():
        slices[k] = slice(current, current + v)
        current = current + v
    return slices


param_slices = build_param_slices()


multiscale = PsychoacousticFeature().to(device)


def multiscale_feature(x):
    d = multiscale.compute_feature_dict(x)
    x = torch.cat([v.view(-1) for v in d.values()])
    return x


def multiscale_loss(a, b):
    a = multiscale_feature(a)
    b = multiscale_feature(b)
    return torch.abs(a - b).sum()


def fb_feature(x):
    x = exp.fb.forward(x, normalize=False)
    x = exp.fb.temporal_pooling(
        x, exp.window_size, exp.step_size)[..., :exp.n_frames]
    # x = x.reshape(x.shape[0], 1, -1)
    # x = max_norm(x, dim=2)
    x = x.view(-1, 128, 128)
    return x



class SynthParams(object):
    def __init__(self, packed):
        super().__init__()
        self.packed = packed
    
    @staticmethod
    def random_events(n_events):
        '''
        p = torch.cat([f0, f0_fine, amp, harm_amp,
                       harm_decay, freq_env], dim=-1)
        '''

        if discrete_freqs:
            f0s = torch.zeros(n_events, 1, len(scale), device=device).uniform_(0, 1).view(-1, len(scale))
        else:
            f0s = torch.zeros(-1, 1, 1, device=device).uniform_(0, 1) ** 2
            f0s = f0s.view(-1, 1)

        f0_fine = torch.zeros(n_events, n_frames, device=device).uniform_(0, 1)
        
        amp_win = Window(n_frames, 0, 1)
        means = torch.zeros(n_events, 1, device=device).uniform_(0, 1)
        stds = torch.zeros(n_events, 1, device=device).uniform_(0, 0.1) ** 2
        amps = (amp_win.forward(means, stds).view(n_events, n_frames) + 1) * 0.5

        harm_amps = torch.cat(
            [
                harmonics(n_harmonics, choice(['triangle', 'square', 'sawtooth']), device=device)[None, :] 
                for _ in range(n_events)
            ], dim=0)

        harm_resonance = torch.zeros(n_events, 1, device=device).uniform_(0, 1).repeat(1, n_harmonics)

        freq_win = Window(freq_domain_filter_size, 0, 1)
        means = torch.zeros(n_events, 1, device=device).uniform_(0, 1) ** 2
        stds = torch.zeros(n_events, 1, device=device).uniform_(0, 1)
        freq_env = freq_win.forward(means, stds).view(n_events, freq_domain_filter_size)


        params = torch.cat([f0s, f0_fine, amps, harm_amps, harm_resonance, freq_env], dim=-1)
        return params


    @property
    def f0(self) -> Tensor:
        if discrete_freqs:
            x = self.packed[:, param_slices['f0']].view(-1, len(scale))
            x = softmax(x)
            x = x @ frequencies
            return x.view(-1, 1).repeat(1, n_frames) ** 2
        else:
            x = self.packed[:, param_slices['f0']
                            ].reshape(-1, 1).repeat(1, n_frames)
            return x ** 2

    @property
    def f0_fine(self) -> Tensor:
        return self.packed[:, param_slices['f0_fine']]

    @property
    def harmonic_amps(self) -> Tensor:
        return self.packed[:, param_slices['harmonic_amps']]

    @property
    def harmonic_decay(self) -> Tensor:
        return self.packed[:, param_slices['harmonic_decay']]

    @property
    def freq_domain_envelope(self) -> Tensor:
        return self.packed[:, param_slices['freq_domain_envelope']]

    @property
    def amp(self) -> Tensor:
        return self.packed[:, param_slices['amp']]


class ProxySpec(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            PosEncodedUpsample(
                total_synth_params + 39, exp.model_dim, size=128, out_channels=128, layers=5)
        )

        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x, verb):
        x = x.view(-1, n_events, total_synth_params)
        verb = verb.view(-1, 1, 39).repeat(1, n_events, 1)
        x = torch.cat([x, verb], dim=-1)
        x = x.view(-1, total_synth_params + 39)
        x = self.net(x)
        x = x.view(-1, n_events, 128, 128)
        x = torch.abs(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x.view(x.shape[0], -1)
        x = max_norm(x, dim=1)
        x = x.view(-1, 128, 128)
        return x


class Synthesizer(nn.Module):
    def __init__(self):
        super().__init__()

        self.global_amp = nn.Parameter(torch.zeros(1).fill_(5000))
        self.osc_amp = nn.Parameter(torch.zeros(1).fill_(50))

    def forward(self, synth_params: SynthParams):

        f0 = synth_params.f0.view(-1, 1, n_frames) * exp.samplerate.nyquist

        f0 = min_f0 + (f0 * f0_span)

        proportion = (f0 / exp.samplerate.nyquist) * 0.1
        f0_fine = synth_params.f0_fine.reshape(-1, 1, n_frames) * \
            exp.samplerate.nyquist * proportion
        f0 + f0 + f0_fine

        batch = f0.shape[0]

        osc = f0 * harmonic_factors[None, :, None]
        radians = (osc / exp.samplerate.nyquist) * np.pi

        indices = torch.where(radians >= np.pi)
        radians[indices] = 0

        radians = F.interpolate(radians, size=exp.n_samples, mode='linear')
        osc_bank = torch.sin(torch.cumsum(radians, dim=-1)) * self.osc_amp

        harm_decay = synth_params.harmonic_decay.view(-1, n_harmonics)
        harm_decay = min_resonance + (harm_decay * res_span)

        harm_amp = synth_params.harmonic_amps.view(-1, n_harmonics)

        amp = (synth_params.amp.view(-1, 1, n_frames) * 2) - 1
        logged['amp'] = amp
        amp_full = F.interpolate(amp, size=exp.n_samples, mode='linear')

        noise_filter = synth_params.freq_domain_envelope.view(
            -1, 1, freq_domain_filter_size)
        noise_filter = F.interpolate(
            noise_filter, exp.n_samples // 2, mode='linear')
        noise_filter = F.pad(noise_filter, (0, 1))

        noise = torch.zeros(batch, 1, exp.n_samples,
                            device=f0.device).uniform_(-1, 1) * self.global_amp
        noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')

        filtered_noise = noise_spec * noise_filter
        noise = torch.fft.irfft(filtered_noise, dim=-1, norm='ortho')
        noise = noise * amp_activation(amp_full)

        current = torch.zeros(batch, n_harmonics, 1, device=f0.device)
        output = []

        for i in range(n_frames):
            current = current + (amp[..., i:i + 1] * harm_amp[..., None])
            current = torch.clamp(current, 0, np.inf)
            output.append(current)
            # print(current.shape, harm_decay.shape)
            current = current * harm_decay[..., None]


        x = torch.cat(output, dim=-1).view(-1, n_harmonics, n_frames)

        logged['res'] = x

        

        x = F.interpolate(x, size=exp.n_samples, mode='linear')

        x = x * osc_bank
        x = (noise / n_harmonics) + x

        x = x.view(-1, n_events, n_harmonics, exp.n_samples)

        x = torch.sum(x, dim=(1, 2), keepdim=True).view(-1, 1, exp.n_samples)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)
        self.n_rooms = self.verb.n_rooms

        self.synth = Synthesizer()

        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])

        self.reduce = nn.Conv1d(exp.scale.n_bands + 33, exp.model_dim, 1, 1, 0)
        self.norm = ExampleNorm()

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=n_events, channels_last=False, dense=False, normalize=True)

        self.to_mix = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.to_room = LinearOutputStack(
            exp.model_dim, 2, out_channels=self.n_rooms)

        self.to_event_params = nn.Sequential(
            LinearOutputStack(
                exp.model_dim, 4, out_channels=total_synth_params)
        )

        mode = conv_mode
        self.to_f = LinearOutputStack(
            exp.model_dim, 3, out_channels=len(scale) if discrete_freqs else 1)
        self.to_fine = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=n_frames, mode=mode, out_channels=1)
        # self.to_fine = LinearOutputStack(exp.model_dim, 3, out_channels=n_frames)
        

        # self.to_harm_amp = ConvUpsample(
        #     exp.model_dim, exp.model_dim, start_size=4, end_size=n_harmonics, out_channels=1, mode=mode)
        # self.to_harm_decay = ConvUpsample(
        #     exp.model_dim, exp.model_dim, start_size=4, end_size=n_harmonics, out_channels=1, mode=mode)
        # self.to_freq_domain_env = ConvUpsample(
        #     exp.model_dim, exp.model_dim, start_size=4, end_size=freq_domain_filter_size, out_channels=1, mode=mode)

        self.to_harm_amp = LinearOutputStack(exp.model_dim, 3, out_channels=n_harmonics)
        self.to_harm_decay = LinearOutputStack(exp.model_dim, 3, out_channels=n_harmonics)
        self.to_freq_domain_env = LinearOutputStack(exp.model_dim, 3, out_channels=freq_domain_filter_size)

        self.to_amp = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=n_frames, out_channels=1, mode=mode)
        # self.to_amp = LinearOutputStack(exp.model_dim, 3, out_channels=n_frames)

        self.apply(lambda p: exp.init_weights(p))

    def pack_verb_params(self, rooms, mx):
        verb_params = torch.cat(
            [mx.view(-1, 1), rooms.view(-1, self.n_rooms)], dim=-1)
        return verb_params

    def encode(self, x):
        batch = x.shape[0]
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

        orig_verb_params, _ = torch.max(x, dim=1)
        verb_params = orig_verb_params

        # expand to params
        mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
        rooms = softmax(self.to_room(verb_params))

        verb_params = self.pack_verb_params(rooms, mx)

        latent = x = x.view(-1, exp.model_dim)

        f0 = self.to_f.forward(x).view(-1, len(scale) if discrete_freqs else 1)
        f0_fine = self.to_fine(x).view(-1, n_frames)
        harm_amp = self.to_harm_amp(x).view(-1, n_harmonics)
        harm_decay = self.to_harm_decay(x).view(-1, n_harmonics)
        freq_env = self.to_freq_domain_env(x).view(-1, freq_domain_filter_size)
        amp = self.to_amp(x).view(-1, n_frames)

        # print(f0.shape, f0_fine.shape, harm_amp.shape, harm_decay.shape, freq_env.shape, amp.shape)

        p = torch.cat([f0, f0_fine, amp, harm_amp, harm_decay, freq_env], dim=-1)
        p = activation(p)
        return p, rooms, mx, verb_params, latent

    def generate_random_params(self, batch_size):
        # p = torch.zeros(batch_size, n_events, total_synth_params, device=device).uniform_(
            # 0, 1).view(-1, total_synth_params)
        

        # p_mag = torch.zeros(batch_size, n_events, total_synth_params // 2 + 1).uniform_(-1, 1)
        # p_angle = torch.zeros(batch_size, n_events, total_synth_params // 2 + 1).uniform_(-np.pi, np.pi)
        # mag_env = torch.linspace(1, 0, total_synth_params // 2 + 1) ** 10
        # p_mag = p_mag * mag_env[None, None, :]

        # real = p_mag * torch.cos(p_angle)
        # imag = p_mag * torch.sin(p_angle)

        # tf = torch.complex(real, imag)
        # p = torch.fft.irfft(tf, dim=-1, norm='ortho')
        # p = torch.cat([p, torch.zeros(batch_size, n_events, 1)], dim=-1)
        # p = p.view(-1, total_synth_params)
        # # p = p - p.min()
        # p = p / (p.max() + 1e-8)
        # p = torch.relu(p)

        p = SynthParams.random_events(batch_size * n_events).view(-1, total_synth_params)


        rooms = F.gumbel_softmax(torch.zeros(
            batch_size, self.n_rooms, device=device).uniform_(0, 1), dim=-1, hard=True)
        mx = torch.zeros(batch_size, device=device).uniform_(0, 1).view(-1, 1, 1)
        packed = self.pack_verb_params(rooms, mx)
        return p, rooms, mx, packed

    def synthesize(self, p, rooms, mx, verb_params):
        params = SynthParams(p)
        samples = self.synth.forward(params)

        rm = softmax(rooms)
        wet = self.verb.forward(samples, rm)

        samples = (mx * wet) + ((1 - mx) * samples)

        # samples = max_norm(samples)

        return samples, p, verb_params

    def forward(self, x):

        p, rooms, mx, verb_params, latent = self.encode(x)
        samples, p, verb_params = self.synthesize(p, rooms, mx, verb_params)
        return samples, p, verb_params, latent

model = Model().to(device)
optim = optimizer(model, lr=learning_rate)

proxy = ProxySpec().to(device)
proxy_optim = optimizer(proxy, lr=learning_rate)

# multi = MultiscaleLoss(128, 128)


def train_proxy(batch):
    """
    Ensure that the renderer matches a real spectrogram
    """
    proxy_optim.zero_grad()

    with torch.no_grad():
        p, rooms, mx, packed = model.generate_random_params(batch.shape[0])
        samples, p, verb_params = model.synthesize(p, rooms, mx, packed)


    pred_spec = proxy.forward(p, verb_params)
    real_spec = fb_feature(samples)

    p_flat = pred_spec.view(batch.shape[0], -1)

    loss = F.mse_loss(pred_spec, real_spec)
    loss.backward()
    proxy_optim.step()
    return loss, pred_spec, samples



def train(batch):
    """
    Train the encoder to minimize loss in render space
    """

    optim.zero_grad()
    recon, params, verb, latent = model.forward(batch)

    # fake_spec = proxy.forward(params, verb)
    # fake_spec = fb_feature(recon)
    # real_spec = fb_feature(batch)

    # loss = F.mse_loss(fake_spec, real_spec)
    loss = exp.perceptual_loss(recon, batch) + latent_loss(latent)

    loss.backward()
    optim.step()


    return loss, recon


@readme
class ResonatorModelExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        self.pred = None
        self.synthetic = None

    def proxy_spec(self):
        return self.pred[0].view(128, 128).data.cpu().numpy().T

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def orig(self):
        return playable(self.real, exp.samplerate)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def amps(self):
        return logged['amp'].view(-1, n_events, n_frames)[0].data.cpu().numpy()
    
    def res(self):
        return logged['res'].view(-1, n_harmonics, n_frames)[0].data.cpu().numpy()
    
    def synth(self):
        return playable(self.synthetic, exp.samplerate)
    
    def synth_spec(self):
        return np.abs(zounds.spectral.stft(self.synth()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            print('-------------------------')

            l, recon = train(item)
            self.fake = recon
            print('R', i, l.item())

            # l, pred, synth = train_proxy(item)
            # self.pred = pred
            # self.synthetic = synth
            # print('P', i, l.item())
