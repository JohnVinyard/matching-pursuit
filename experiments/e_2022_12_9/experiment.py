from torch import Tensor, nn
from config.dotenv import Config
from config.experiment import Experiment
import zounds
import torch
import numpy as np
from torch.jit import ScriptModule, script_method
from torch.nn import functional as F
from modules.atoms import unit_norm
from modules.decompose import fft_frequency_decompose
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm, max_norm
from modules.pos_encode import pos_encoded
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from modules.shape import Reshape
from modules.sparse import VectorwiseSparsity
from modules.stft import stft

from train.optim import optimizer
from upsample import ConvUpsample
from util import playable
from util.music import MusicalScale
from util.readmedocs import readme
from util import device

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
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

min_resonance = 0.75
res_span = 1 - min_resonance
loss_type = 'perceptual'
discrete_freqs = False
learning_rate = 1e-3


def softmax(x):
    # return F.gumbel_softmax(x, dim=-1, hard=True)
    return torch.softmax(x, dim=-1)


class MultiscaleLoss(nn.Module):
    def __init__(self, kernel_size, n_bands):
        super().__init__()
        scale = zounds.LinearScale(
            zounds.FrequencyBand(1, exp.samplerate.nyquist), n_bands, False)
        self.fb = zounds.learn.FilterBank(
            exp.samplerate, kernel_size, scale, 0.25, normalize_filters=True).to(device)

    def _feature(self, x):
        x = x.view(-1, 1, exp.n_samples)
        bands = fft_frequency_decompose(x, 512)
        output = []
        for size, band in bands.items():
            spec = self.fb.forward(band, normalize=False)
            windowed = spec.unfold(-1, 128, 64)
            pooled = windowed.mean(dim=-1)
            windowed = torch.abs(torch.fft.rfft(
                windowed, dim=-1, norm='ortho'))
            windowed = unit_norm(windowed, axis=-1)
            output.append(pooled.view(-1))
            output.append(windowed.view(-1))
        return torch.cat(output)

    def forward(self, a, b):
        a = self._feature(a)
        b = self._feature(b)
        return torch.abs(a - b).sum()


def amp_activation(x):
    return torch.abs(x)
    # return torch.clamp(x, 0, 1)
    # return torch.relu(x)
    # return F.leaky_relu(x, 0.2)


def activation(x):
    # return torch.clamp(x, 0, 1)
    # return (torch.sin(x) + 1) * 0.5
    return torch.sigmoid(x)


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


'''
Resonator
-------------------
f0
f0_variance * n_frames
harm_amp * n_harmonics
decay * n_harmonics

Impulse
--------------------
amp * n_frames
freq domain envelope
window (if impulses must be localized in time)


This still needs to be viewed as individual events, because
of polyphonic instruments like the piano

It can be factored into

instrument
------------
harm_amp * n_harmonics
decay * n_harmonics
freq_domain_envelope

event
-----------
f0
f0_variance * n_frames
amp * n_frames


For the initial over-fitting experiment(s), 
we should just optimize the synth parameters directly
and exclude any latent generation/hierarchical network
stuff

https://freesound.org/people/LiftPizzas/sounds/586617/
https://freesound.org/people/Seidhepriest/sounds/232014/
https://freesound.org/people/Walter_Odington/sounds/25602/
https://freesound.org/people/hellska/sounds/328727/
https://freesound.org/people/ananth-pattabi/sounds/44335/
https://freesound.org/people/ldk1609/sounds/55944/
https://freesound.org/people/MTG/sounds/358332/

'''


class SynthParams(object):
    def __init__(self, packed):
        super().__init__()
        self.packed = packed

    @property
    def f0(self) -> Tensor:
        if discrete_freqs:
            x = self.packed[:, param_slices['f0']].view(-1, len(scale))
            x = softmax(x)
            x = x @ frequencies
            return x.view(-1, 1).repeat(1, n_frames) ** 2
        else:
            x = self.packed[:, param_slices['f0']
                            ].view(-1, 1).repeat(1, n_frames)
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

            ConvUpsample(
                total_synth_params + 39,
                exp.model_dim,
                4,
                end_size=128,
                mode='nearest',
                out_channels=128)

            # nn.Linear(total_synth_params + exp.model_dim, 512),
            # Reshape((128, 2, 2)),
            # nn.ConvTranspose2d(128, 64, (4, 4), (2, 2), (1, 1)),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(64, 32, (4, 4), (2, 2), (1, 1)),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(32, 16, (4, 4), (2, 2), (1, 1)),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(16, 8, (4, 4), (2, 2), (1, 1)),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(8, 4, (4, 4), (2, 2), (1, 1)),
            # nn.LeakyReLU(0.2),
            # nn.ConvTranspose2d(4, 1, (4, 4), (2, 2), (1, 1)),
        )

        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x, verb):

        x = x.view(-1, n_events, total_synth_params)
        verb = verb.view(-1, 1, 39).repeat(1, n_events, 1)

        x = torch.cat([x, verb], dim=-1)
        x = x.view(-1, total_synth_params + 39)
        x = self.net(x)
        x = x.view(-1, n_events, 128, 128)
        # x = torch.relu(x)
        # x = x ** 2
        x = torch.abs(x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = x.view(x.shape[0], -1)
        # x = max_norm(x, dim=1)
        x = x.view(-1, 128, 128)
        return x


class Synthesizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, synth_params: SynthParams):

        f0 = synth_params.f0.view(-1, 1, n_frames) * exp.samplerate.nyquist

        f0 = min_f0 + (f0 * f0_span)

        proportion = (f0 / exp.samplerate.nyquist) * 0.1
        f0_fine = synth_params.f0_fine.view(-1, 1, n_frames) * \
            exp.samplerate.nyquist * proportion
        f0 + f0 + f0_fine

        batch = f0.shape[0]

        osc = f0 * harmonic_factors[None, :, None]
        radians = (osc / exp.samplerate.nyquist) * np.pi
        radians = F.interpolate(radians, size=exp.n_samples, mode='linear')
        osc_bank = torch.sin(torch.cumsum(radians, dim=-1))

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
                            device=f0.device).uniform_(-1, 1)
        noise_spec = torch.fft.rfft(noise, dim=-1, norm='ortho')

        filtered_noise = noise_spec * noise_filter
        noise = torch.fft.irfft(filtered_noise, dim=-1, norm='ortho')
        noise = noise * amp_activation(amp_full)

        current = torch.zeros(batch, n_harmonics, 1, device=f0.device)
        output = []

        for i in range(n_frames):
            current = current + (amp[..., i:i + 1] * harm_amp[..., None])
            current = torch.clamp(current, 0, 1)
            output.append(current)
            current = current * harm_decay[..., None]

        x = torch.cat(output, dim=-1).view(-1, n_harmonics, n_frames)

        indices = torch.where(x >= np.pi)
        x[indices] = 0

        x = F.interpolate(x, size=exp.n_samples, mode='linear')

        x = x * osc_bank
        x = noise + x

        x = x.view(-1, n_events, n_harmonics, exp.n_samples)

        x = torch.sum(x, dim=(1, 2), keepdim=True).view(-1, 1, exp.n_samples)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.params = nn.Parameter(
        #     torch.zeros(n_events, total_synth_params).uniform_(0.01, 0.999))

        self.verb = NeuralReverb.from_directory(
            Config.impulse_response_path(), exp.samplerate, exp.n_samples)
        self.n_rooms = self.verb.n_rooms

        # self.rooms = nn.Parameter(torch.zeros(
        # self.verb.n_rooms).uniform_(-1, 1))
        # /elf.mix = nn.Parameter(torch.zeros(1).uniform_(0, 1))
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

        mode = 'nearest'
        self.to_f = LinearOutputStack(
            exp.model_dim, 3, out_channels=len(scale) if discrete_freqs else 1)
        self.to_fine = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=n_frames, mode=mode, out_channels=1)
        self.to_harm_amp = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=n_harmonics, out_channels=1, mode=mode)
        self.to_harm_decay = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=n_harmonics, out_channels=1, mode=mode)
        self.to_freq_domain_env = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=freq_domain_filter_size, out_channels=1, mode=mode)
        self.to_amp = ConvUpsample(
            exp.model_dim, exp.model_dim, start_size=4, end_size=n_frames, out_channels=1, mode=mode)

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

        # verb_params = torch.cat(
        # [mx.view(-1, 1), rooms.view(-1, self.n_rooms)], dim=-1)
        verb_params = self.pack_verb_params(rooms, mx)

        # event_params = self.to_event_params.forward(x)

        # p = event_params.view(-1, total_synth_params)

        x = x.view(-1, exp.model_dim)

        f0 = self.to_f.forward(x).view(-1, len(scale) if discrete_freqs else 1)
        f0_fine = self.to_fine(x).view(-1, n_frames)
        harm_amp = self.to_harm_amp(x).view(-1, n_harmonics)
        harm_decay = self.to_harm_decay(x).view(-1, n_harmonics)
        freq_env = self.to_freq_domain_env(x).view(-1, freq_domain_filter_size)
        amp = self.to_amp(x).view(-1, n_frames)

        p = torch.cat([f0, f0_fine, amp, harm_amp,
                       harm_decay, freq_env], dim=-1)
        p = activation(p)
        return p, rooms, mx, verb_params

    def generate_random_params(self, batch_size):
        p = torch.zeros(batch_size, n_events, total_synth_params).uniform_(
            0, 1).view(-1, total_synth_params)

        rooms = F.gumbel_softmax(torch.zeros(
            batch_size, self.n_rooms).uniform_(0, 1), dim=-1, hard=True)
        mx = torch.zeros(batch_size).uniform_(0, 1).view(-1, 1, 1)
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

        p, rooms, mx, verb_params = self.encode(x)
        samples, p, verb_params = self.synthesize(p, rooms, mx, verb_params)
        return samples, p, verb_params

        # batch = x.shape[0]
        # x = exp.fb.forward(x, normalize=False)
        # x = exp.fb.temporal_pooling(
        #     x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        # x = self.norm(x)
        # pos = pos_encoded(
        #     batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)

        # x = torch.cat([x, pos], dim=1)
        # x = self.reduce(x)

        # x = self.context(x)

        # x = self.norm(x)

        # x, indices = self.sparse(x)

        # orig_verb_params, _ = torch.max(x, dim=1)
        # verb_params = orig_verb_params

        # # expand to params
        # mx = torch.sigmoid(self.to_mix(verb_params)).view(batch, 1, 1)
        # rooms = softmax(self.to_room(verb_params))

        # verb_params = torch.cat(
        #     [mx.view(-1, 1), rooms.view(-1, self.n_rooms)], dim=-1)

        # # event_params = self.to_event_params.forward(x)

        # # p = event_params.view(-1, total_synth_params)

        # x = x.view(-1, exp.model_dim)

        # f0 = self.to_f.forward(x).view(-1, len(scale) if discrete_freqs else 1)
        # f0_fine = self.to_fine(x).view(-1, n_frames)
        # harm_amp = self.to_harm_amp(x).view(-1, n_harmonics)
        # harm_decay = self.to_harm_decay(x).view(-1, n_harmonics)
        # freq_env = self.to_freq_domain_env(x).view(-1, freq_domain_filter_size)
        # amp = self.to_amp(x).view(-1, n_frames)

        # p = torch.cat([f0, f0_fine, amp, harm_amp, harm_decay, freq_env], dim=-1)
        # p = activation(p)

        # if add_noise:
        #     p = (p * 0.5) + (0.5 * torch.zeros_like(p).uniform_(0, 1))

        # params = SynthParams(p)
        # samples = self.synth.forward(params)

        # rm = softmax(rooms)
        # wet = self.verb.forward(samples, rm)

        # samples = (mx * wet) + ((1 - mx) * samples)

        # samples = max_norm(samples)

        # return samples, p, orig_verb_params


model = Model().to(device)
optim = optimizer(model, lr=learning_rate)

proxy = ProxySpec().to(device)
proxy_optim = optimizer(proxy, lr=learning_rate)

multi = MultiscaleLoss(128, 128)


def train_proxy(batch):
    """
    Ensure that the renderer matches a real spectrogram
    """
    proxy_optim.zero_grad()

    recon, params, verb = model.forward(batch)

    pred_spec = proxy.forward(params, verb)
    real_spec = fb_feature(recon)

    loss = F.mse_loss(pred_spec, real_spec)
    loss.backward()
    proxy_optim.step()
    return loss, pred_spec


def train_encoder(batch):
    """
    Ensure that the encoder does not always produce the same
    output
    """
    optim.zero_grad()

    with torch.no_grad():
        p, rooms, mx, packed = model.generate_random_params(batch.shape[0])
        samples, p, verb_params = model.synthesize(p, rooms, mx, packed)

    fake_p, rooms, mx, fake_packed = model.encode(samples)

    p_loss = F.mse_loss(fake_p, p)
    v_loss = F.mse_loss(fake_packed, packed)

    loss = p_loss + v_loss
    loss.backward()
    optim.step()

    return loss


def train(batch):
    """
    Train the encoder to minimize loss in render space
    """

    optim.zero_grad()
    recon, params, verb = model.forward(batch)

    fake_spec = proxy.forward(params, verb)
    real_spec = fb_feature(batch)

    loss = F.mse_loss(fake_spec, real_spec)

    loss.backward()
    optim.step()

    # if loss_type == 'perceptual':
    #     loss = exp.perceptual_loss(recon, batch, norm='l1')
    # elif loss_type == 'multiscale':
    #     # loss = multiscale_loss(recon, batch)
    #     loss = multi.forward(recon, batch)
    # else:
    #     real_spec = stft(batch, 512, 256, pad=True, log_amplitude=False)
    #     fake_spec = stft(recon, 512, 256, pad=True, log_amplitude=False)
    #     # real_spec = fb_feature(batch)
    #     # fake_spec = fb_feature(recon)
    #     loss = F.mse_loss(fake_spec, real_spec)

    # loss.backward()
    # optim.step()

    return loss, recon


@readme
class ResonatorModelExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.real = None
        self.fake = None
        self.pred = None

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

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item

            print('-------------------------')

            l, recon = train(item)
            self.fake = recon
            print('R', i, l.item())

            l = train_encoder(item)
            print('E', i, l.item())

            l, pred = train_proxy(item)
            self.pred = pred
            print('P', i, l.item())
