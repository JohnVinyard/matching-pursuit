import zounds
from config.dotenv import Config
from data.datastore import batch_stream
from modules import stft
from modules.psychoacoustic import PsychoacousticFeature
from util import device
from modules.reverb import NeuralReverb
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam

feature = PsychoacousticFeature().to(device)

def loss_func(inp, target):
    inp = feature.compute_feature_dict(inp)
    target = feature.compute_feature_dict(target)
    loss = 0
    for k, v in inp.items():
        loss = loss + torch.abs(inp[k] - target[k]).sum()
    
    return loss

    inp = stft(inp)
    target = stft(target)
    return F.mse_loss(inp, target)


class VerbWrapper(nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.verb = NeuralReverb(n_samples, 1)
        self.mix = nn.Parameter(torch.FloatTensor(1, 1).fill_(1))
    
    def forward(self, x):
        x = self.verb.forward(x, self.mix)
        return x


if __name__ == '__main__':

    # settings
    n_samples = 2**15
    sr = zounds.SR22050()

    # start the app
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    # get some audio
    bs = batch_stream(Config.audio_path(), '*.wav', 1, n_samples)
    audio = next(bs)
    audio /= (np.abs(audio).max() + 1e-12)
    audio = torch.from_numpy(audio).to(device)

    # get the impulse
    impulse = zounds.AudioSamples\
        .from_file('/home/john/Downloads/reverbs/Five Columns.wav')\
        .mono[:n_samples]
    impulse /= np.abs(impulse.max())
    
    # apply a known impulse response
    with torch.no_grad():
        static_verb = NeuralReverb(
            n_samples, 1, impulses=impulse.reshape((1, n_samples)))
        dry = zounds.AudioSamples(audio.data.cpu().numpy().squeeze(), sr).pad_with_silence()
        with_verb = static_verb(audio, torch.FloatTensor(1, 1).fill_(1))

    # create a model to learn reverb settings
    model = VerbWrapper(n_samples).to(device)
    optim = Adam(model.parameters(), lr=1e-4, betas=(0, 0.9))


    def real():
        return zounds.AudioSamples(with_verb.data.cpu().numpy().squeeze(), sr).pad_with_silence()
    
    def fake():
        return zounds.AudioSamples(f.data.cpu().numpy().squeeze(), sr).pad_with_silence()
    
    def real_impulse():
        return np.array(impulse.squeeze())

    def fake_impulse():
        return model.verb.rooms.data.cpu().numpy().squeeze()

    while True:
        optim.zero_grad()
        f = model.forward(audio)
        loss = loss_func(f, with_verb)
        loss.backward()
        optim.step()
        print(loss.item())



    input('waiting..')
