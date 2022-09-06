import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.ddsp import AudioModel
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.sparse import VectorwiseSparsity
from modules.waveguide import WaveguideSynth
from train.optim import optimizer
from upsample import ConvUpsample

from util import device, playable
from modules import pos_encoded

from util.readmedocs import readme
import numpy as np

exp = Experiment(zounds.SR22050(), 2**15, weight_init=0.1)



class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self):
        super().__init__()

        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])

        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)

        self.sparse = VectorwiseSparsity(
            exp.model_dim, keep=16, channels_last=False, dense=False)
        

        self.impulse = ConvUpsample(exp.model_dim, exp.model_dim, 4, exp.n_frames, 'nearest', out_channels=1)
        self.delay_selection = ConvUpsample(exp.model_dim, exp.model_dim, 4, exp.n_frames, 'nearest', out_channels=512)
        self.damping = LinearOutputStack(exp.model_dim, 2, out_channels=1)
        self.filt = LinearOutputStack(exp.model_dim, 2, out_channels=512)

        self.decode = WaveguideSynth(
            max_delay=512, 
            n_samples=exp.n_samples, 
            filter_kernel_size=512)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, exp.n_samples)
        x = torch.abs(exp.fb.convolve(x))
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        # x = self.decode(x)
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)
        x = self.context(x)
        x, indices = self.sparse(x)

        
        x = x.view(-1, exp.model_dim)

        impulse = self.impulse(x)
        delay_selection = self.delay_selection(x)
        damping = self.damping(x)
        filt = self.filt(x)

        x = self.decode.forward(impulse, delay_selection, damping, filt)
        print(x.shape)
        x = x.view(batch, 16, exp.n_samples)

        output = torch.zeros(batch, 1, exp.n_samples * 2)
        for b in range(batch):
            for i in range(16):
                start = indices[b, i] * 256
                end = start + exp.n_samples
                output[b, :, start: end] += x[b, i]
        
        output = output[..., :exp.n_samples]
        return output


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x):
        x = self.summary(x)
        return x


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class WaveguideSynthesisExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None

    def orig(self):
        return playable(self.real, exp.samplerate, normalize=True)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, exp.samplerate, normalize=True)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.real = item
            loss, self.fake = train_model(item)
            print('GEN', i, loss.item())
