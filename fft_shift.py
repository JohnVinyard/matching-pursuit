import zounds
from data import AudioIterator
import numpy as np
import torch
from torch import nn
from modules.dilated import DilatedStack
from modules.pos_encode import pos_encoded
from train import optimizer
from modules.normal_pdf import pdf
from upsample import ConvUpsample
from torch.nn import functional as F
from util import make_initializer
from modules.stft import stft
from util import playable

from torch.distributions.normal import Normal

samplerate = zounds.SR22050()
n_samples = 2 ** 15

init = make_initializer(0.05)


def fft_shift(a, shift):
    n_samples = a.shape[-1]
    shift_samples = shift * n_samples
    a = F.pad(a, (0, n_samples))
    spec = torch.fft.rfft(a, dim=-1, norm='ortho')

    n_coeffs = spec.shape[-1]
    shift = (torch.arange(0, n_coeffs, device=a.device) * 2j * np.pi) / n_coeffs
    shift = torch.exp(-shift * shift_samples)

    spec = spec * shift

    samples = torch.fft.irfft(spec, dim=-1, norm='ortho')
    # samples = samples[..., :n_samples]
    return samples

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.shifts = nn.Parameter(torch.zeros(2, 1, 1).normal_(0, np.pi))
    
    def forward(self, x, gen_signals):
        gen_signals = gen_signals.view(2, 1, 8192)
        gen_signals = F.pad(gen_signals, (0, n_samples - 8192))
        signal = fft_shift(gen_signals, torch.sin(self.shifts))
        signal = signal.sum(dim=0).view(1, 1, n_samples * 2)
        return signal


model = Model()
optim = optimizer(model, lr=1e-4)


def does_this_work():
    attack = np.zeros(2**16)
    attack[:64] = np.random.uniform(-1, 1, 64) * np.hamming(64)
    print(attack.max())

    t = np.zeros(2**16)
    synth = zounds.SineSynthesizer(samplerate)
    tone = synth.synthesize(samplerate.frequency * 8192, [220])
    t[:8192] = tone * np.linspace(1, 0, 8192) ** 2
    print(t.max())

    loc = np.zeros(2**16)
    loc[2**13] = 1
    print(loc.max())

    a = np.fft.rfft(attack, axis=-1, norm='ortho')
    t = np.fft.rfft(t, axis=-1, norm='ortho')
    l = np.fft.rfft(loc, axis=-1, norm='ortho')

    spec = a * t * l
    final = np.fft.irfft(spec, axis=-1, norm='ortho')
    final = final[:2**15]
    return zounds.AudioSamples(final, samplerate).pad_with_silence()



    

def train(batch, a, b):
    optim.zero_grad()

    batch = F.pad(batch, (0, n_samples))

    a = a * np.hamming(8192)
    b = b * np.hamming(8192)

    gen_signals = np.concatenate([a[None, ...], b[None, ...]], axis=0)
    gen_signals = torch.from_numpy(gen_signals).float()

    x = model.forward(batch, gen_signals)

    spec = stft(batch.view(1, 1, -1), pad=True).view(-1, 257)
    fspec = stft(x.view(1, 1, -1), pad=True).view(-1, 257)

    r_sim = torch.cdist(spec, spec)
    f_sim = torch.cdist(fspec, fspec)

    sim_loss = F.mse_loss(f_sim, r_sim)

    loss = ((spec - fspec) ** 2).mean() + sim_loss

    spec = torch.fft.rfft(batch.view(1, 1, -1), dim=-1, norm='ortho')
    fspec = torch.fft.rfft(x.view(1, 1, -1), dim=-1, norm='ortho')
    spec = torch.cat([spec.real, spec.imag], dim=-1)
    fspec = torch.cat([fspec.real, fspec.imag], dim=-1)
    full_loss = F.mse_loss(fspec, spec)
    
    loss = loss + full_loss

    # loss = sim_loss


    loss.backward()
    optim.step()
    print(loss.item())

    return x

'''
def phase_shift(coeffs, samplerate, time_shift, axis=-1, frequency_band=None):
    frequency_dim = coeffs.dimensions[axis]
    if not isinstance(frequency_dim, FrequencyDimension):
        raise ValueError(
            'dimension {axis} of coeffs must be a FrequencyDimension instance, '
            'but was {cls}'.format(axis=axis, cls=frequency_dim.__class__))

    n_coeffs = coeffs.shape[axis]
    shift_samples = int(time_shift / samplerate.frequency)
    shift = (np.arange(0, n_coeffs) * 2j * np.pi) / n_coeffs
    shift = np.exp(-shift * shift_samples)
    shift = ArrayWithUnits(shift, [frequency_dim])

    frequency_band = frequency_band or slice(None)
    new_coeffs = coeffs.copy()

    if coeffs.ndim == 1:
        new_coeffs[frequency_band] *= shift[frequency_band]
        return new_coeffs

    slices = [slice(None) for _ in range(coeffs.ndim)]
    slices[axis] = frequency_band
    new_coeffs[tuple(slices)] *= shift[frequency_band]
    return new_coeffs
'''




if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    # x = does_this_work()

    synth = zounds.SineSynthesizer(samplerate)

    a = synth.synthesize(samplerate.duration * 8192, [440])
    b = synth.synthesize(samplerate.duration * 8192, [210])

    canvas = np.zeros(n_samples)
    start_a = 10
    start_b = 16444

    canvas[start_a: start_a + 8192] = a * np.hamming(8192)
    canvas[start_b: start_b + 8192] = b * np.hamming(8192)


    samples = zounds.AudioSamples(canvas, samplerate).pad_with_silence()

    t_samples = torch.from_numpy(canvas).float()

    # rng = torch.linspace(0, 1, 128)
    # means = torch.zeros(16).uniform_(0, 1)
    # stds = torch.zeros(16).uniform_(0, 0.1)

    # pdf_example = pdf(rng[None, :], means[:, None], stds[:, None])
    # mx, _ = torch.max(pdf_example, dim=-1, keepdim=True)
    # pdf_example = pdf_example / (mx + 1e-8)
    # pdf_example = pdf_example.data.cpu().numpy()

    while True:
        result = train(t_samples, a, b)
        r = result.data.cpu().numpy().squeeze()
        rr = zounds.AudioSamples(r, samplerate).pad_with_silence()

    input('waiting')

