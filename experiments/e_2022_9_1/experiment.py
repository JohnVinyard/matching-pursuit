import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.ddsp import AudioModel
from modules.dilated import DilatedStack
from modules.sparse import sparsify, sparsify_vectors
from train.optim import optimizer
from upsample import PosEncodedUpsample

from util import device, playable
from modules import pos_encoded

from util.readmedocs import readme
import numpy as np

exp = Experiment(zounds.SR22050(), 2**15, weight_init=0.1)


class ElementwiseSparsity(nn.Module):
    def __init__(self, high_dim=2048, keep=64):
        super().__init__()
        self.expand = nn.Conv1d(exp.model_dim, high_dim, 1, 1, 0)
        self.contract = nn.Conv1d(high_dim, exp.model_dim, 1, 1, 0)
        self.keep = keep

    def forward(self, x):
        x = self.expand(x)
        x = sparsify(x, self.keep)
        x = self.contract(x)
        return x


class VectorwiseSparsity(nn.Module):
    def __init__(self, keep=16, channels_last=True):
        super().__init__()
        self.channels_last = channels_last
        self.attn = nn.Linear(exp.model_dim, 1)
        self.keep = keep

    def forward(self, x):
        if not self.channels_last:
            x = x.permute(0, 2, 1)

        batch, time, channels = x.shape

        attn = self.attn(x).view(batch, time)
        attn = torch.softmax(attn, dim=1)

        x = sparsify_vectors(
            x, attn, n_to_keep=self.keep, dense=True, normalize=False)

        if not self.channels_last:
            x = x.permute(0, 2, 1)

        return x


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

        self.sparse = ElementwiseSparsity(high_dim=2048, keep=32)
        # self.sparse = VectorwiseSparsity(keep=16, channels_last=False)

        self.decode = nn.Sequential(
            DilatedStack(exp.model_dim, [1, 3, 9, 27, 1]),
            AudioModel(
                exp.n_samples,
                exp.model_dim,
                exp.samplerate,
                exp.n_frames,
                exp.n_frames * 8)
        )

        # self.decode = PosEncodedUpsample(
        #     latent_dim=exp.model_dim,
        #     channels=exp.model_dim,
        #     size=exp.n_samples,
        #     out_channels=1,
        #     layers=1,
        #     concat=False,
        #     learnable_encodings=True,
        #     multiply=False,
        #     transformer=False,
        #     filter_bank=True)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, exp.n_samples)
        x = torch.abs(exp.fb.convolve(x))
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)
        x = self.context(x)
        x = self.sparse(x)
        x = self.decode(x)
        return x


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
class SparseRepresentationExperiment(object):
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
