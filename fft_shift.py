import zounds
from data import AudioIterator
import numpy as np
import torch
from torch import nn
from train import optimizer
from modules.normal_pdf import pdf
from upsample import ConvUpsample
from torch.nn import functional as F
from util import make_initializer
from modules.stft import stft

samplerate = zounds.SR22050()
n_samples = 2 ** 15

init = make_initializer(0.1)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.means = nn.Parameter(torch.zeros(2, 1).uniform_(0, 1))
        self.register_buffer('stds', torch.zeros(1).fill_(1 / n_samples))

        self.latents = nn.Parameter(torch.zeros(2, 16).normal_(0, 1))

        self.gen = ConvUpsample(16, 16, 4, 8192, mode='learned', out_channels=1)
        self.apply(init)
    
    def forward(self, x):
        r = torch.linspace(0, 1, n_samples)

        impulses = pdf(
            r, 
            torch.clamp(self.means, 0, 1), 
            self.stds).view(2, 1, n_samples)
        
        gen_signals = self.gen.forward(self.latents).view(2, 1, 8192)
        gen_signals = F.pad(gen_signals, (0, n_samples - 8192))

        impulse_spec = torch.fft.rfft(impulses, norm='ortho', dim=-1)
        gen_spec = torch.fft.rfft(gen_signals, norm='ortho', dim=-1)
        spec = impulse_spec * gen_spec
        inverse = torch.fft.irfft(spec, norm='ortho', dim=-1)

        return torch.sum(inverse, dim=0).view(1, 1, n_samples), impulses


model = Model()
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    x, impulses = model.forward(None)
    diff = ((x - batch) ** 2).mean()

    spec = stft(batch.view(1, 1, n_samples), pad=True).view(128, 257)
    fspec = stft(x.view(1, 1, n_samples), pad=True).view(128, 257)

    rsim = spec @ spec.T
    fsim = fspec @ fspec.T

    sim_loss = ((rsim - fsim) ** 2).mean()

    loss = sim_loss + diff
    loss.backward()
    optim.step()
    print(loss.item())
    return x, impulses


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    synth = zounds.SineSynthesizer(samplerate)

    a = synth.synthesize(samplerate.duration * 8192, [440])
    b = synth.synthesize(samplerate.duration * 8192, [220])

    canvas = np.zeros(n_samples)
    start_a = 10
    start_b = 16444

    canvas[start_a: start_a + 8192] = a * np.hamming(8192)
    canvas[start_b: start_b + 8192] = b * np.hamming(8192)

    samples = zounds.AudioSamples(canvas, samplerate).pad_with_silence()

    t_samples = torch.from_numpy(canvas).float()

    while True:
        result, impulses = train(t_samples)
        ii = impulses.data.cpu().numpy().squeeze()
        r = result.data.cpu().numpy().squeeze()
        rr = zounds.AudioSamples(r, samplerate).pad_with_silence()

    input('waiting')
    