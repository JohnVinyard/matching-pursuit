import zounds

from data.audioiter import AudioIterator
from modules import MoreCorrectScattering
from modules.overfitraw import OverfitRawAudio
from train.optim import optimizer
from torch.nn import functional as F
import torch
from util import playable
import numpy as np

n_samples = 2 ** 15
samplerate = zounds.SR22050()

band = zounds.FrequencyBand(20, samplerate.nyquist)
scale = zounds.MelScale(band, 128)

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    stream = AudioIterator(1, n_samples, samplerate, normalize=True, overfit=True)

    def nl(x):
        x = torch.relu(x)
        return x
    
    scatter = MoreCorrectScattering(
        samplerate, scale, 512, non_linearity=nl, scaling_factors=0.01)

    model = OverfitRawAudio((1, 1, n_samples), std=0.0001)
    optim = optimizer(model, lr=1e-2)

    for item in stream:
        item = item.view(-1, 1, n_samples)

        optim.zero_grad()
        recon = model.forward(None)
        fake_feat = scatter.forward(recon)
        real_feat = scatter.forward(item)
        loss = F.mse_loss(fake_feat, real_feat)
        loss.backward()
        print(loss.item())
        optim.step()

        rf = real_feat.data.cpu().numpy().squeeze().T
        ff = fake_feat.data.cpu().numpy().squeeze().T
        a = playable(recon, samplerate)
        o = playable(item, samplerate)