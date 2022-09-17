import zounds
from data import AudioIterator
import numpy as np
import torch
from torch import nn
from train import optimizer
from modules.normal_pdf import pdf
from torch.nn import functional as F
from upsample import ConvUpsample, PosEncodedUpsample
from util import make_initializer
from modules.stft import stft
from util.playable import playable

samplerate = zounds.SR22050()
n_samples = 2 ** 15
n_events = 8

init = make_initializer(0.02)

def fft_convolve(a, b):
    a = torch.fft.rfft(a, dim=-1, norm='ortho')
    b = torch.fft.rfft(b, dim=-1, norm='ortho')
    spec = a * b
    td = torch.fft.irfft(spec, dim=-1, norm='ortho')
    return td


def get_batch_and_positions(n_segments, segment_length, n_samples):
    ai = AudioIterator(n_segments, segment_length, samplerate, normalize=True)
    batch = next(ai.__iter__())
    positions = np.random.randint(0, n_samples, n_segments)

    canvas = torch.zeros((1, 1, n_samples * 2))
    for b, p in zip(batch, positions):
        canvas[0, 0, p: p + segment_length] += b
    
    canvas = canvas[..., :n_samples]

    return canvas, batch, positions 


# def impulses(means, n_samples):
#     rng = torch.linspace(0, 1, n_samples)
#     std = torch.zeros(1).fill_(1 / n_samples)
#     backward_std = std * 1000000
    
#     forward_impulse = pdf(rng[None, None, :], means[None, :, None], std[None, :, None]).view(-1, 1, n_samples)
#     mx = torch.max(forward_impulse)
#     forward_impulse = forward_impulse / (mx + 1e-8)

#     backward_impulse = pdf(rng[None, None, :], means[None, :, None], backward_std[None, :, None]).view(-1, 1, n_samples)
#     mx = torch.max(backward_impulse)
#     backward_impulse = backward_impulse / (mx +1e-8)


#     x = forward_impulse
#     x = backward_impulse + (forward_impulse - backward_impulse).detach()
#     return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.impulses = nn.Parameter(torch.zeros(n_events, 128).uniform_(0, 0.1))     

        self.latents = nn.Parameter(torch.zeros(n_events, 16).normal_(0, 1))
        self.up = ConvUpsample(16, 32, 8, 128, out_channels=1, mode='learned')   
        # self.up = PosEncodedUpsample(
            # 16, 32, 128, out_channels=1, layers=4)

        self.apply(init)
    
    def forward(self, batch):
        batch = batch.view(-1, n_events, 4096)
        batch = F.pad(batch, (0, n_samples - 4096))

        impulses = self.up(self.latents)
        imp = torch.clamp(impulses.view(n_events, 1, 128), 0, 2)
        # imp = F.gumbel_softmax(imp, tau=1, hard=True, dim=-1)
        i = imp

        imp = F.interpolate(imp, size=n_samples, mode='nearest').view(1, n_events, n_samples)

        signal = fft_convolve(batch, imp)
        signal = torch.sum(signal, dim=1, keepdim=True)
        # mx = torch.max(signal)
        # signal = signal / (mx + 1e-8)
        return signal, i.squeeze()


model = Model()
optim = optimizer(model, lr=1e-3)


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    canvas, batch, pos = get_batch_and_positions(n_events, 4096, n_samples)

    def listen():
        return playable(result, samplerate)
    
    def target():
        return playable(canvas, samplerate)

    while True:
        optim.zero_grad()

        result, imp = model.forward(batch)
        i = imp.data.cpu().numpy().squeeze()
        real_spec = stft(canvas, 512, 256, pad=True)
        fake_spec = stft(result, 512, 256, pad=True)
        loss = F.mse_loss(fake_spec, real_spec)
        loss.backward()
        optim.step()
        print(loss.item())


    input('waiting')
    