import torch
from torch import nn
import zounds
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.normalization import ExampleNorm
from modules.sparse import sparsify, to_sparse_vectors_with_context
from modules.waveguide import TransferFunctionSegmentGenerator
from train.optim import optimizer

from util import device, playable
from modules import pos_encoded

from util.readmedocs import readme
import numpy as np

exp = Experiment(zounds.SR22050(), 2**15, weight_init=0.1)

n_events = 16

class Summarizer(nn.Module):
    """
    Summarize a batch of audio samples into a set of sparse
    events, with each event described by a single vector.

    Output will be (batch, n_events, event_vector_size)

    """

    def __init__(self):
        super().__init__()


        self.norm = ExampleNorm()

        # encoder = nn.TransformerEncoderLayer(
        #     exp.model_dim, 4, exp.model_dim, batch_first=True)
        # encoder.norm1 = ExampleNorm()
        # encoder.norm2 = ExampleNorm()
        # self.context = nn.TransformerEncoder(encoder, 4, norm=None)

        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])

        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)

        # self.expand = nn.Parameter(torch.zeros(exp.model_dim, 2048).uniform_(-1, 1))

        self.expand = nn.Linear(exp.model_dim, 2048, bias=False)

        self.embed_one_hot = nn.Linear(2048, exp.model_dim, bias=False)
        self.embed_context = nn.Linear(2048, exp.model_dim, bias=False)
        self.reduce_again = nn.Linear(exp.model_dim * 2, exp.model_dim)

        self.tf = TransferFunctionSegmentGenerator(
            exp.model_dim, exp.n_frames, 512, exp.n_samples)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, 1, exp.n_samples)
        x = torch.abs(exp.fb.convolve(x))
        x = exp.fb.temporal_pooling(
            x, exp.window_size, exp.step_size)[..., :exp.n_frames]
        x = self.norm(x)
        pos = pos_encoded(
            batch, exp.n_frames, 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.reduce(x)

        # x = x.permute(0, 2, 1)
        x = self.context(x)
        x = self.norm(x)

        x = x.permute(0, 2, 1)
        x = self.expand(x)
        x = x.permute(0, 2, 1)

        # x = torch.dropout(x, 0.05, train=self.training)

        raw = x

        x = torch.softmax(x, dim=1)

        sparse = x = sparsify(x, n_events)

        one_hot, context, positions = to_sparse_vectors_with_context(x, n_events)

        oh = self.embed_one_hot(one_hot)
        # c = self.embed_context(context)
        # x = torch.cat([oh, c], dim=-1)
        # x = self.reduce_again(x)
        x = oh

        segments = self.tf.forward(x)
        segments = segments.view(batch, n_events, exp.n_samples)

        output = torch.zeros(batch, 1, exp.n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(n_events):
                pos_index = (b * n_events) + i
                start = positions[pos_index] * exp.step_size
                end = start + exp.n_samples
                output[b, :, start: end] += segments[b, i]

        
        return output[..., :exp.n_samples], sparse, raw



class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x):
        x, sparse, raw = self.summary(x)
        return x, sparse, raw


model = Model().to(device)
optim = optimizer(model, lr=1e-4)


def train_model(batch):
    optim.zero_grad()
    recon, sparse, raw = model.forward(batch)

    mn = torch.mean(raw, dim=-1)
    mx, _ = torch.max(mn, dim=-1, keepdim=True)

    diff = torch.abs(mn - mx).mean()


    # mx, _ = torch.max(raw, dim=1)
    # diff = torch.abs(1 - mx).mean()

    # sm = torch.sum(raw, dim=-1)
    # mn = torch.mean(sm, dim=-1)
    # mx, _ = torch.max(sm, dim=-1)

    loss = exp.perceptual_loss(recon, batch) + diff
    loss.backward()
    optim.step()
    return loss, recon, sparse


@readme
class SparseRepresentationExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.real = None
        self.sparse = None

    def orig(self):
        return playable(self.real, exp.samplerate, normalize=True)

    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, exp.samplerate, normalize=True)

    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def encoded(self):
        return self.sparse.data.cpu().numpy()[0].T
    
    def nonzero(self):
        return np.nonzero(self.encoded())[1]

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.real = item
            loss, self.fake, self.sparse = train_model(item)
            print('GEN', i, loss.item())
