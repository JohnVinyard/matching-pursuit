import zounds
import torch
from modules import diff_index
from torch import nn
import numpy as np
from torch.nn import functional as F

from train.optim import optimizer
from util import playable

samplerate = zounds.SR22050()
n_samples = 2**15

n_frames = 128

wavetable_size = 1024
wavetable = torch.sin(torch.linspace(-np.pi, np.pi, steps=wavetable_size))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(n_frames).uniform_(-0.001, 0.001))

    def forward(self, x):
        p = self.p.view(-1, 1, n_frames)
        p = F.upsample(p, size=n_samples, mode='linear').view(-1)
        indices = torch.cumsum(p, dim=-1)
        # indices = torch.sin(indices)

        values = diff_index(wavetable, indices)
        return values


model = Model()
optim = optimizer(model, lr=1e-4)

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    synth = zounds.SineSynthesizer(samplerate)
    target = synth.synthesize(samplerate.frequency * n_samples, [220])
    target = torch.from_numpy(target).float()

    def wt():
        return wavetable.data.cpu().numpy().squeeze()

    def listen():
        return playable(estimate.view(1, -1), samplerate)

    while True:
        optim.zero_grad()
        estimate = model.forward(None)
        loss = F.mse_loss(estimate, target)
        loss.backward()
        optim.step()
        print(loss.item())

        listen()
