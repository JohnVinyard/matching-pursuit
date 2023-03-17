import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.ddsp import overlap_add
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.linear import LinearOutputStack
from modules.normalization import max_norm
from modules.reverb import ReverbGenerator
from modules.softmax import hard_softmax, sparse_softmax
from modules.sparse import sparsify, sparsify_vectors

from modules.stft import morlet_filter_bank
from modules.transfer import ImpulseGenerator, PosEncodedImpulseGenerator
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class Impulse(nn.Module):
    def __init__(self, n_frames, n_samples, n_impulses):
        super().__init__()
        self.n_frames = n_frames
        self.n_samples = n_samples
        self.n_impulses = n_impulses

        self.impulses = nn.Parameter(torch.zeros(
            n_impulses, n_frames).uniform_(0, 1))

    def forward(self, x):
        x = x.view(-1, self.n_impulses)
        x = x @ (self.impulses ** 2)
        x = x.view(-1, 1, self.n_frames)
        x = F.interpolate(x, size=self.n_samples, mode='linear')
        noise = torch.zeros_like(x).uniform_(-1, 1)
        x = x * noise
        return x


class Bandpass(nn.Module):
    def __init__(self, filter_size, n_filters):
        super().__init__()
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.filters = nn.Parameter(torch.zeros(self.n_filters, self.filter_size).uniform_(-1, 1))

    def forward(self, x, signal):
        batch = x.shape[0]
        signal = signal.view(1, batch, signal.shape[-1])
        x = x.view(-1, self.n_filters)
        x = x @ self.filters
        x = x.view(batch, 1, self.filter_size)
        x = x * torch.hamming_window(self.filter_size, device=x.device)[None, None, :]
        x = F.conv1d(signal, x, padding=self.filter_size // 2, groups=batch)
        x = x.view(batch, 1, -1)
        return x


class BandLimitedNoise(nn.Module):
    def __init__(self, n_frames, n_samples, n_impulses, filter_size, n_filters):
        super().__init__()
        self.n_samples = n_samples
        self.impulses = Impulse(n_frames, n_samples, n_impulses)
        self.bandpass = Bandpass(filter_size, n_filters)

    def forward(self, impulse_choice, filter_choice):
        impulse = self.impulses.forward(impulse_choice)
        filtered = self.bandpass.forward(filter_choice, impulse)
        return filtered[..., :self.n_samples]


class Resonance(nn.Module):
    def __init__(self, n_samples, n_frequencies, samplerate: zounds.SampleRate):
        super().__init__()
        self.n_samples = n_samples
        self.n_frequencies = n_frequencies

        band = zounds.FrequencyBand(20, samplerate.nyquist)
        scale = zounds.MelScale(band, n_frequencies)
        bank = morlet_filter_bank(
            samplerate, n_samples, scale, 0.01, normalize=False)
    
        bank = torch.from_numpy(bank.real).float().view(n_frequencies, n_samples) 
        bank = max_norm(bank, dim=-1)

        self.register_buffer('bank', bank)

    def forward(self, balance: torch.Tensor, decay: torch.Tensor):
        balance = balance.view(-1, self.n_frequencies)
        decay = decay.view(-1, 1)
        decay = torch.clamp(decay, 0, 1)

        x = balance @ self.bank
        x = x.view(-1, 1, self.n_samples)
        x = F.pad(x, (0, 256))
        windowed = \
            x.unfold(-1, size=512, step=256) * torch.hamming_window(512,
                                                                   device=x.device)[None, None, None, :]

        decay = decay.repeat(1, windowed.shape[-2])

        decay = torch.exp(torch.cumsum(torch.log(decay + 1e-8), dim=-1))

        x = windowed * decay[:, None, :, None]

        x = overlap_add(x, apply_window=False)
        x = x[..., :self.n_samples]
        return x


class Event(nn.Module):
    def __init__(
            self,
            n_samples,
            n_frequencies,
            samplerate,
            n_frames,
            impulse_samples,
            n_impulses,
            filter_size,
            n_filters):

        super().__init__()
        self.n_samples = n_samples
        self.impulse_samples = impulse_samples
        self.resonance = Resonance(
            n_samples, n_frequencies, samplerate)
        self.impulse = BandLimitedNoise(
            n_frames, impulse_samples, n_impulses, filter_size, n_filters)
    
    def forward(self, balance, decay, impulse_choice, filter_choice, amp):
        batch = balance.shape[0]

        res = self.resonance.forward(balance, decay)
        impulse = self.impulse.forward(impulse_choice, filter_choice)

        res = res.view(1, batch, self.n_samples)
        impulse = impulse.view(batch, 1, self.impulse_samples)

        x = F.conv1d(res, impulse, padding=self.impulse_samples // 2, groups=batch)[..., :self.n_samples]
        x = x.view(-1, 1, self.n_samples)
        imp = F.pad(impulse.reshape(x.shape[0], 1, -1), (0, self.n_samples - self.impulse_samples))
        imp = imp.view(-1, 1, self.n_samples)
        x = x + imp
        x = x * amp.view(-1, 1, 1)
        x = x.view(-1, 1, self.n_samples)
        return x



class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.encoder = DilatedStack(
            channels, 
            [1, 3, 9, 27, 81, 1], 
            dropout=0.1, 
            soft_sparsity=False, 
            internally_sparse=False, 
            sparsity_amt=1)
        

        self.event = Event(
            exp.n_samples, 
            n_frequencies=512, 
            samplerate=exp.samplerate, 
            n_frames=128, 
            impulse_samples=2048, 
            n_impulses=512, 
            filter_size=32, 
            n_filters=512)
        
        self.to_balance = LinearOutputStack(channels, 3, out_channels=512)
        self.to_decay = LinearOutputStack(channels, 3, out_channels=1)
        self.to_impulse = LinearOutputStack(channels, 3, out_channels=512)
        self.to_filter = LinearOutputStack(channels, 3, out_channels=512)
        self.to_amp = LinearOutputStack(channels, 3, out_channels=1)

        self.to_time = LinearOutputStack(channels, 3, out_channels=33)

        self.to_pos = PosEncodedImpulseGenerator(
            exp.n_samples, exp.n_samples, softmax=lambda x: hard_softmax(x, invert=True, tau=0.01))
        # self.to_pos = ImpulseGenerator(exp.n_samples, softmax=lambda x: hard_softmax(x, invert=True, tau=0.01))
    
        self.attn = nn.Conv1d(channels, 1, 1, 1, 0)
    
        self.verb = ReverbGenerator(channels, 3, exp.samplerate, exp.n_samples)
        self.apply(lambda p: exp.init_weights(p))
    
    def forward(self, x, debug=False):
        batch = x.shape[0]
        x = exp.pooled_filter_bank(x)
        x = x[..., :exp.n_samples]
        encoded = self.encoder(x)

        context = torch.mean(encoded, dim=-1)
        attn = self.attn.forward(encoded)
        events, indices = sparsify_vectors(encoded, attn, n_to_keep=16)

        b = torch.relu(self.to_balance.forward(events))
        d = torch.sigmoid(self.to_decay.forward(events))
        i = torch.softmax(self.to_impulse.forward(events), dim=-1)
        f = torch.softmax(self.to_filter.forward(events), dim=-1)
        a = self.to_amp.forward(events) ** 2
        t = self.to_time.forward(events)

        events = self.event.forward(
            b.view(-1, 512), 
            d.view(-1, 1), 
            i.view(-1, 512), 
            f.view(-1, 512), 
            a.view(-1, 1))
        
        impulses, _ = self.to_pos.forward(t.view(batch * 16, -1))

        impulses = impulses.view(batch, 16, exp.n_samples)
        events = events.view(batch, 16, exp.n_samples)
        final = fft_convolve(events, impulses)
        final = torch.sum(final, dim=1, keepdim=True)

        final = self.verb.forward(context, final)
        return final

model = Model(exp.model_dim).to(device)

optim = optimizer(model, lr=1e-4)


def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch, debug=False)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class PhysicalModel(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None