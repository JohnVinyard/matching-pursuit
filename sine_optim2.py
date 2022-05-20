import zounds
import torch
from torch import nn
from modules import MelScale, AudioCodec, stft
from torch.nn import functional as F
import numpy as np

from train.optim import optimizer
from util import playable

sr = zounds.SR22050()
n_samples = 2**14

scale = MelScale()
codec = AudioCodec(scale)


band = zounds.FrequencyBand(40, sr.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(sr, 512, scale, 0.1, normalize_filters=True, a_weighting=False)

def features(x):
    x = x.view(-1, n_samples)
    spec = codec.to_frequency_domain(x)
    return spec[..., 0]


def loss(inp, target):
    inp_spec = features(inp)
    target_spec = features(target)
    return F.mse_loss(inp_spec, target_spec)


class Model(nn.Module):
    def __init__(self, n_osc=3, mode='complex'):
        super().__init__()
        self.params = nn.Parameter(torch.zeros(n_osc, 2).uniform_(0.0001, 0.99999))
        self.n_osc = n_osc
        self.mode = mode

    def forward(self, x):

        if self.mode == 'clamp':
            freq = torch.tanh(self.params[:, 0])
            amp = torch.sigmoid(self.params[:, 1])
        elif self.mode == 'sin':
            freq = torch.sin(self.params[:, 0])
            amp = torch.sigmoid(self.params[:, 1])
        elif self.mode == 'complex':
            freq = self.params[:, 0]
            amp = self.params[:, 1]

            amp = torch.norm(self.params, dim=1)
            # freq = torch.cosine_similarity(self.params, torch.ones_like(self.params))
            freq = (torch.angle(torch.complex(amp, freq)) / np.pi)
            # print('amp', amp.item(), 'freq', freq.item())
        else:
            raise ValueError('Unsupported mode')

        accum = torch.zeros(1, self.n_osc, n_samples)
        accum[:, :] = freq[None, :, None]

        signal = amp[None, :, None] * torch.sin(torch.cumsum(accum, dim=-1) * np.pi)
        signal = signal.sum(dim=1)
        return signal


model = Model(n_osc=3)
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    synth = zounds.SineSynthesizer(sr)
    samples = synth.synthesize(sr.duration * n_samples, [220, 440, 880])

    samples = torch.from_numpy(samples).view(1, 1, n_samples).float()

    def listen():
        return playable(inp, sr)

    def look():
        return features(inp).data.cpu().numpy().squeeze()
    
    def real():
        return features(samples).data.cpu().numpy().squeeze()

    while True:
        optim.zero_grad()
        inp = model.forward(None)
        l = loss(inp, samples)
        l.backward()
        optim.step()
        print(l.item())
