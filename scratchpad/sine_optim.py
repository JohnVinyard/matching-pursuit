import zounds
import torch
from torch import nn
from torch.optim import Adam
import numpy as np
from scipy.signal.windows import tukey
from torch.nn import functional as F
from modules.linear import LinearOutputStack
from modules.psychoacoustic import PsychoacousticFeature
from util import device

from modules.stft import stft
from util.weight_init import make_initializer

sr = zounds.SR22050()

min_frequency = 20 / sr.nyquist
max_frequency = 1

scale = zounds.MelScale(zounds.FrequencyBand(20, sr.nyquist - 100), 512)
filter_bank = zounds.learn.FilterBank(
    sr, 
    512, 
    scale, 
    0.1, 
    normalize_filters=True, 
    a_weighting=True).to(device)

synth = zounds.SineSynthesizer(sr)

n_samples = 2**14

a_note = 440
c_note = 261.625565

table = torch.sin(torch.linspace(0, 2 * np.pi, 2048))

cs = torch.cumsum(torch.linspace(0, np.pi, n_samples), dim=-1)

feature = PsychoacousticFeature().to(device)

def filter_bank_loss(inp, t):
    inp = filter_bank.forward(inp, normalize=False)
    inp = filter_bank.temporal_pooling(inp, 512, 256)

    t = filter_bank.forward(t, normalize=False)
    t = filter_bank.temporal_pooling(t, 512, 256)

    return F.mse_loss(inp, t)

def time_domain_loss(inp, t):
    return F.mse_loss(inp, t)


def spectral_mag_loss(inp, t):
    inp = stft(inp)
    t = stft(t)
    return F.mse_loss(inp, t)


def phase_invariant_loss(inp, t):
    inp, _ = feature(inp.view(1, 1, n_samples))
    t, _ = feature(t.view(1, 1, n_samples))
    return F.mse_loss(inp, t)


def make_signal():

    duration = sr.frequency * (n_samples / 2)
    first_part = synth.synthesize(duration, [a_note / 2, a_note, a_note * 2])
    second_part = synth.synthesize(duration, [c_note / 2, c_note, c_note * 2])

    window = tukey(n_samples // 2)

    full = np.concatenate([first_part * window, second_part * window])
    samples = zounds.AudioSamples(full, sr)

    spec = np.abs(zounds.spectral.stft(samples))

    t = torch.from_numpy(samples).float().to(device)

    return samples, spec, t

def make_simple_signal():
    first_part = synth.synthesize(sr.frequency * n_samples, [220])

    window = tukey(n_samples)

    samples = zounds.AudioSamples(first_part * window, sr)

    spec = np.abs(zounds.spectral.stft(samples))

    t = torch.from_numpy(samples).float().to(device)

    return samples, spec, t


def identity(x):
    return x

def clamp(x):
    return torch.clamp(x, 0, 1)

def squared(x):
    return x ** 2

def abs(x):
    return torch.abs(x)

# settings ########################################
init_weights = make_initializer(0.01)
home_freqs = False
use_nn_approx = False
smooth_freq = False
smooth_amp = False
loss_func = spectral_mag_loss
prebuilt_accum = False
upsample_mode = 'linear'
learning_rate = 1e-3
wavetable = False
n_osc = 3
create_signal = make_signal

# LEARNING:  Success seems to hinge on using this as a nonlinearity and not
# constraining the values.  
# TODO: Check out the values.  Is this a situation where aliasing is playing a role?
amp_nl = abs
freq_nl = abs
###################################################


def erb(f):
    f = f * sr.nyquist
    return (0.108 * f) + 24.7

def scaled_erb(f):
    return erb(f) / sr.nyquist

erb_plot = scaled_erb(np.linspace(0, 1, 100))

class ApproxSine(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.approx_sine = LinearOutputStack(
            channels=32,
            layers=6,
            out_channels=1,
            in_channels=1,
            shortcut=True,
            activation=activation)

        initializer = make_initializer(0.1)

        self.apply(initializer)

    def forward(self, x):
        x = x.view(-1, 1)
        x = self.approx_sine(x)
        return x


class Model(nn.Module):
    def __init__(
            self,
            upsample=True,
            smooth_freq=False,
            smooth_amp=False,
            nn_approx=False,
            prebuilt_accum=False,
            wavetable=False,
            n_osc=n_osc,
            home_freqs=home_freqs):

        super().__init__()
        init_value = 0.1
        self.prebuilt_accum = prebuilt_accum
        self.upsample = upsample
        self.smooth_freq = smooth_freq
        self.smooth_amp = smooth_amp
        self.wavetable = wavetable
        self.n_osc = n_osc
        self.home_freqs = home_freqs
        self.latent = nn.Parameter(torch.FloatTensor(
            n_osc * 2, 32 if upsample else n_samples).uniform_(-init_value, init_value))
        
        if self.home_freqs:
            self.base_freqs = nn.Parameter(torch.FloatTensor(n_osc).uniform_(0, 0.5))

        self.nn_approx = nn_approx
        self.apply(init_weights)

    def forward(self, x):

        x = self.latent
        amp = amp_nl(x[:self.n_osc, :])
        freq = freq_nl(x[self.n_osc:, :])

        if self.home_freqs:
            freq = self.base_freqs[:, None] + (scaled_erb(self.base_freqs[:, None]) * freq)

        freq_params = freq
        amp_params = amp

        if self.upsample:
            amp = F.upsample(amp[None, ...], size=n_samples,
                             mode=upsample_mode).view(self.n_osc, n_samples)
            freq = F.upsample(freq[None, ...], size=n_samples,
                              mode=upsample_mode).view(self.n_osc, n_samples)

        if self.smooth_freq:
            freq = freq.view(1, self.n_osc, n_samples)
            freq = F.avg_pool1d(freq, 512, 1, 256)[..., :-1]
            freq = freq.view(self.n_osc, n_samples)

        if self.smooth_amp:
            amp = amp.view(1, self.n_osc, n_samples)
            amp = F.avg_pool1d(amp, 512, 1, 256)[..., :-1]
            amp = amp.view(self.n_osc, n_samples)
        

        if self.prebuilt_accum:
            phase_accum = freq * cs[None, :]
        else:
            phase_accum = torch.cumsum(freq * np.pi, dim=-1)

        if self.nn_approx:
            # neural network approximation
            signal = approx(phase_accum).view(self.n_osc, n_samples) * amp
        elif self.wavetable:
            # TODO: this doesn't work as there are no gradients
            # for the modulus or indexing operations
            unwrapped = phase_accum % (2 * np.pi)
            indices = ((unwrapped / (2 * np.pi)) * table.shape[0]).long()
            signal = table[indices].view(self.n_osc, n_samples)
        else:
            # sine of phase accumulator
            signal = torch.sin(phase_accum) * amp
            # signal = torch.sin(freq * cs[None, :]) * amp

        signal = signal.mean(axis=0)
        return signal, freq_params, amp_params


def fake():
    audio = zounds.AudioSamples(result.data.cpu().numpy().squeeze(), sr)
    return audio


def fake_spec():
    audio = zounds.AudioSamples(result.data.cpu().numpy().squeeze(), sr)
    sp = np.abs(zounds.spectral.stft(audio))
    return sp


def view_sine_approx():
    samples = torch.linspace(0, 2 * np.pi, 1024)
    a = approx(samples)
    return a.data.cpu().numpy().reshape((-1,))


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread()

    if use_nn_approx:
        # First, train the approx sine model
        approx = ApproxSine(activation=torch.sin).to(device)
        optim = Adam(approx.parameters(), lr=1e-4, betas=(0, 0.9))
        samples = np.linspace(0, 2 * np.pi, 1000)

        while True:
            optim.zero_grad()
            indices = np.random.permutation(samples.shape[0])[:128]
            batch = torch.from_numpy(samples[indices]).float().to(device)
            target = torch.sin(batch)

            recon = approx(batch)

            loss = F.mse_loss(recon.view(-1), target.view(-1))
            loss.backward()
            optim.step()
            print(loss.item())

            if loss <= 1e-4:
                break
    else:
        approx = None

    audio, spectrogram, target = create_signal()

    model = Model(
        upsample=True,
        smooth_freq=smooth_freq,
        smooth_amp=smooth_amp,
        prebuilt_accum=prebuilt_accum,
        nn_approx=use_nn_approx,
        wavetable=wavetable,
        home_freqs=home_freqs).to(device)

    optim = Adam(model.parameters(), lr=learning_rate, betas=(0, 0.9))

    while True:
        optim.zero_grad()
        result, fp, ap = model.forward(None)

        # discourage big jumps in a frequency channel
        delta = torch.abs(torch.diff(fp, axis=-1))
        delt = delta.data.cpu().numpy()

        # discourage big jumps in an amp channel
        amp_delta = torch.abs(torch.diff(ap, axis=-1))
        amp_delt = delta.data.cpu().numpy()

        # encourage frequencies to be at least an ERB apart
        f_params = fp.permute(1, 0).contiguous()
        erbs = scaled_erb(f_params)
        diff = torch.cdist(f_params[..., None], f_params[..., None]) # (32, 3, 3)
        diff = diff - erbs[:, :, None]
        diff = torch.clamp(diff, -np.inf, 0)
        erb_loss = -diff.mean()

        # encourage frequencies to be within the correct range


        f = fp.data.cpu().numpy().squeeze()
        a = ap.data.cpu().numpy().squeeze()


        # TODO: Consider an aliasing loss as well to 
        # keep values in a reasonable range
        loss = loss_func(result, target) + (delta.mean() * 1.0) + (erb_loss.mean() * 0.5)
        loss.backward()
        optim.step()
        print(loss.item())

        fake_spec()

    input('Waiting...')
