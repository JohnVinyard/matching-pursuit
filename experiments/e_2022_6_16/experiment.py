import torch
from modules.overfitraw import OverfitRawAudio
from modules.phase import morlet_filter_bank
from modules.scattering import batch_fft_convolve, scattering_transform
from train.optim import optimizer
from util import playable, readme
import zounds

samplerate = zounds.SR22050()
band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 64)
from torch.nn import functional as F

bank = torch.from_numpy(morlet_filter_bank(samplerate, 512, scale, 0.1).real)

n_samples = 2**15


model = OverfitRawAudio((1, n_samples), std=0.1)
optim = optimizer(model, lr=1e-2)

@readme
class ScatteringTransformTest(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.c1 = None
        self.c2 = None
        self.audio = None
        self.real = None

    def view1(self):
        return self.c1.data.cpu().numpy()[0].T

    def view2(self):
        return self.c2.data.cpu().numpy()[0].T
    
    def listen(self):
        return playable(self.audio, samplerate)
    
    def r(self):
        return playable(self.real, samplerate)

    def run(self):
        for item in self.stream:
            self.c1, self.c2 = scattering_transform(item, bank)
            self.real = item

            while True:
                optim.zero_grad()

                self.audio = model.forward(None)
                c1, c2 = scattering_transform(self.audio, bank)


                loss = F.mse_loss(c1, self.c1) + F.mse_loss(c2, self.c2) * 100
                loss.backward()
                optim.step()
                print(loss.item())

