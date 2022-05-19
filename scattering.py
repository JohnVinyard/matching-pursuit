from pickletools import optimize
import zounds
import torch
from torch import nn
from torch.nn import functional as F
from data import audio_stream
from modules.overfitraw import OverfitRawAudio
from train.optim import optimizer
from util import playable


sr = zounds.SR22050()
n_samples = 2**14

band = zounds.FrequencyBand(40, sr.nyquist)
scale = zounds.MelScale(band, 128)
fb = zounds.learn.FilterBank(
    sr, 512, scale, 0.1, normalize_filters=True, a_weighting=False)

def feature(x):
    x = fb.forward(x, normalize=False)
    pooled = fb.temporal_pooling(x, 512, 256)

    up = F.upsample(pooled, size=n_samples)
    x = x - up

    x = x.permute(1, 0, 2) # (128, 1, 16384)
    x = fb.forward(x, normalize=False).permute(1, 0, 2) # (128, 128, 16384)
    p2 = fb.temporal_pooling(x, 512, 256)

    x = torch.cat([pooled.view(-1), p2.view(-1)])
    return x

def loss(inp, target):
    inp_feat = feature(inp)
    target_feat = feature(target)
    return F.mse_loss(inp_feat, target_feat)


model = OverfitRawAudio((1, 1, n_samples), std=0.1)
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)

    stream = audio_stream(1, n_samples, overfit=True, normalize=True, as_torch=True)

    audio = next(stream)

    def listen():
        return playable(current, sr)

    
    def real():
        return playable(audio, sr)
    
    while True:
        optim.zero_grad()
        current = model.forward(None)
        l = loss(current, audio)
        l.backward()
        optim.step()
        print(l.item())