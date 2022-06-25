from torch.nn import Module
import torch
import zounds
from torch import nn
from modules.phase import morlet_filter_bank


from train.optim import optimizer

samplerate = zounds.SR22050()
n_samples = 2 ** 15
band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)

bank = morlet_filter_bank(samplerate, 1024, scale, 0.1).real
atom = torch.from_numpy(bank[10])


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(n_samples - 1024).normal_(0, 1))
    
    def forward(self, x):
        p = torch.softmax(self.p, dim=-1)
        values, indices = torch.max(p, dim=-1)
        v = values.mean()
        output = torch.zeros(n_samples)
        output[indices: indices + 1024] = v * atom
        return output

if __name__ == '__main__':
    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(8888)

    target = torch.zeros(n_samples)
    target[233: 233 + 1024] = atom

    model = Model()
    optim = optimizer(model, lr=1e-4)

    while True:
        optim.zero_grad()
        real = target.data.cpu().numpy()
        o = model.forward(None)
        fake = o.data.cpu().numpy()
        loss = torch.abs(target - o).sum()
        loss.backward()
        optim.step()
        print(loss.item(), o.argmax().item())





    