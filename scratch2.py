from enum import auto
import torch
import zounds
import numpy as np
from data.audiostream import audio_stream
from modules.normal_pdf import pdf
from modules.phase import AudioCodec, MelScale
from modules.recurrent import RecurrentSynth, Conductor
from torch import nn
from torch.nn import functional as F

from train.optim import optimizer
from util import playable
from torch import autograd
from torch.nn.utils.clip_grad import clip_grad_norm_, clip_grad_value_


sr = zounds.SR22050()

n_samples = 2**15
samples_per_frame = 256
frames = n_samples // samples_per_frame
channels = 128
voices = 8

mel_scale = MelScale()
codec = AudioCodec(mel_scale)

def feature(x):
    x = codec.to_frequency_domain(x)
    return x[..., 0]

def loss(a, b):
    return torch.abs(a - b).sum()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, frames, channels).normal_(0, 0.02))
        self.con = Conductor(
            layers=3, 
            channels=channels, 
            voices=voices, 
            samples_per_frame=samples_per_frame, 
            total_frames=frames)
    
    def forward(self, _):
        return self.con(self.p)


model = Model()
optim = optimizer(model, lr=1e-3)

if __name__ == '__main__':

    app = zounds.ZoundsApp(globals=globals(), locals=locals())
    app.start_in_thread(9999)


    stream = audio_stream(1, n_samples, overfit=True, normalize=True, as_torch=True)
    batch = next(stream).view(1, n_samples)

    def real():
        return playable(batch, sr)
    
    def fake():
        return playable(recon, sr)

    while True:
        optim.zero_grad()
        recon = model.forward(None).view(1, n_samples)

        l = loss(recon, batch)
        # loss = F.mse_loss(recon, batch)
        # loss.backward()

        # found_nan = False
        # for name, p in model.named_parameters():
        #     if p.grad is not None:
        #         print(name, p.grad.std().item())
        #         if torch.isnan(p.grad).sum().item() > 0:
        #             found_nan = True
        #     else:
        #         print(name, 'no grad')
        
        # if found_nan:
        #     raise ValueError('NaN detected')
            
        l.backward()
        # clip_grad_value_(model.parameters(), clip_value=0.5)
        optim.step()
        print(l.item())


    input('Waiting...')
