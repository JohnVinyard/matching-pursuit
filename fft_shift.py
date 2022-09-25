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

from torch.distributions.normal import Normal

samplerate = zounds.SR22050()
n_samples = 2 ** 15

init = make_initializer(0.05)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.means = nn.Parameter(torch.zeros(2, 1).uniform_(0, 1))
        # self.register_buffer('stds', torch.zeros(1).fill_(1 / n_samples))
        # self.stds = nn.Parameter(torch.zeros(2, 1).uniform_(0, 0.01))

        # self.latents = nn.Parameter(torch.zeros(2, 16).normal_(0, 1))

        self.impulses = nn.Parameter(torch.zeros(2, n_samples).normal_(0, 1))

        # self.gen = ConvUpsample(16, 16, 4, 8192, mode='learned', out_channels=1)
        # self.apply(init)

        # self.down = nn.Conv1d(257, 32, 1, 1, 0)
        # self.up = nn.Conv1d(33, 32, 1, 1, 0)
        # self.stack = DilatedStack(32, [1, 3, 9, 1])
        # self.to_means = nn.Linear(32, 2)
        # self.to_stds = nn.Linear(32, 2)
        # self.apply(init)
    
    def forward(self, x, gen_signals):

        # x = stft(x, 512, 256, pad=True).permute(0, 2, 1)
        # pos = pos_encoded(x.shape[0], 128, 16, device=x.device).permute(0, 2, 1).view(-1, 33, 128)
        # pos = self.up(pos)
        # x = self.down(x)
        # x = x + pos
        # x = self.stack(x)
        # x, _ = torch.max(x, dim=-1)

        # means = self.to_means(x).view(2, 1)
        # stds = self.to_stds(x).view(2, 1)


        # r = torch.linspace(0, 1, n_samples)
        # dist = Normal(torch.sigmoid(means), 1e-12 + torch.sigmoid(stds))
        # impulses = torch.exp(dist.log_prob(r[None, :])).view(2, 1, n_samples)
        # mx, _ = torch.max(impulses, dim=-1, keepdim=True)
        # impulses = impulses / (mx + 1e-12)

        # impulses = F.gumbel_softmax(self.impulses, dim=-1, hard=False)
        impulses = torch.softmax(self.impulses, dim=-1)
        impulses = impulses.view(2, 1, n_samples)
        
        gen_signals = gen_signals.view(2, 1, 8192)
        gen_signals = F.pad(gen_signals, (0, n_samples - 8192))

        impulse_spec = torch.fft.rfft(impulses, norm='ortho', dim=-1)
        gen_spec = torch.fft.rfft(gen_signals, norm='ortho', dim=-1)
        spec = impulse_spec * gen_spec
        inverse = torch.fft.irfft(spec, norm='ortho', dim=-1)

        signal = torch.sum(inverse, dim=0).view(1, 1, n_samples)
        signal = signal / (signal.max() + 1e-8)
        return signal, impulses


model = Model()
optim = optimizer(model, lr=1e-4)


def train(batch, a, b):
    optim.zero_grad()

    a = a * np.hamming(8192)
    b = b * np.hamming(8192)

    gen_signals = np.concatenate([a[None, ...], b[None, ...]], axis=0)
    gen_signals = torch.from_numpy(gen_signals).float()

    x, impulses = model.forward(batch, gen_signals)
 
    # print(10 + 4096, 16444 + 4096)
    # print(torch.argmax(impulses, dim=-1))

    spec = stft(batch.view(1, 1, n_samples), pad=True).view(128, 257)
    fspec = stft(x.view(1, 1, n_samples), pad=True).view(128, 257)

    diff = ((spec - fspec) ** 2).mean()

    # rsim = spec @ spec.T
    # fsim = fspec @ fspec.T

    # rsim = torch.cdist(spec, spec)
    # fsim = torch.cdist(fspec, fspec.detach())

    # sim_loss = ((rsim - fsim) ** 2).mean()

    loss = diff

    # spec = torch.fft.rfft(batch, dim=-1, norm='ortho')
    # fspec = torch.fft.rfft(x, dim=-1, norm='ortho')

    # mag_loss = F.mse_loss(fspec.real, spec.real)
    # phase_loss = F.mse_loss(fspec.imag, spec.imag)

    # loss = mag_loss + phase_loss


    loss.backward()
    optim.step()
    print(loss.item())
    # print('===================================')

    return x, impulses


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

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

    rng = torch.linspace(0, 1, 128)
    means = torch.zeros(16).uniform_(0, 1)
    stds = torch.zeros(16).uniform_(0, 0.1)

    pdf_example = pdf(rng[None, :], means[:, None], stds[:, None])
    mx, _ = torch.max(pdf_example, dim=-1, keepdim=True)
    pdf_example = pdf_example / (mx + 1e-8)
    pdf_example = pdf_example.data.cpu().numpy()

    while True:
        result, impulses = train(t_samples, a, b)
        ii = impulses.data.cpu().numpy().squeeze()
        r = result.data.cpu().numpy().squeeze()
        rr = zounds.AudioSamples(r, samplerate).pad_with_silence()

    input('waiting')

