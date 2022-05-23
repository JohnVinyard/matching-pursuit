import torch
from torch import nn
import zounds

from data.audiostream import audio_stream
from modules.ddsp import NoiseModel, UnconstrainedOscillatorBank
from modules.phase import AudioCodec, MelScale
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import playable
from util.weight_init import make_initializer
from torch.nn import functional as F

n_samples = 2**15
sr = zounds.SR22050()
osc_frames = n_samples // 256 
noise_frames = osc_frames * 4
channels = 128
n_osc = 64

mel_scale = MelScale()
codec = AudioCodec(mel_scale)

init_weights = make_initializer(0.1)

def feature(x):
    x = codec.to_frequency_domain(x)
    return x[..., 0]

def loss(a, b):
    return F.mse_loss(a, b)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.osc_params = nn.Parameter(torch.zeros(1, channels, osc_frames).normal_(0, 0.001))
        self.noise_params = nn.Parameter(torch.zeros(1, channels, osc_frames).normal_(0, 1))
        self.room_params = nn.Parameter(torch.zeros(8).uniform_(0, 1))
        self.mix_params = nn.Parameter(torch.ones(1))

        self.osc = UnconstrainedOscillatorBank(
            channels, 
            n_osc, 
            n_samples, 
            fft_upsample=False, 
            baselines=True)
        
        self.noise = NoiseModel(channels, osc_frames, noise_frames, n_samples, channels)

        self.verb = NeuralReverb(n_samples, 8)

        self.apply(init_weights)

    def forward(self, _):
        osc = self.osc(self.osc_params)
        noise = self.noise(self.noise_params)
        signal = osc + noise

        rooms = torch.softmax(self.room_params, dim=-1)
        wet = self.verb.forward(signal, rooms[None, ...])
        mix = torch.sigmoid(self.mix_params)

        x = (signal * mix) + (wet * (1 - mix))
        return x

model = Model()
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    stream = audio_stream(
        1, n_samples, overfit=True, normalize=True, as_torch=True)
    target = next(stream)


    def f():
        return playable(fake, sr)
    
    def real():
        return playable(target, sr)

    while True:
        optim.zero_grad()
        fake = model.forward(None)
        l = loss(fake, target)
        l.backward()
        optim.step()

        print(l.item())
