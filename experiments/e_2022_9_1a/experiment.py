import torch
from torch import nn
from torch.nn import functional as F
from config import Experiment
import zounds
from modules import LinearOutputStack
from modules.dilated import DilatedStack
from modules.normal_pdf import pdf
from modules.pos_encode import pos_encoded
from modules.sparse import ElementwiseSparsity, VectorwiseSparsity, sparsify_vectors
from modules.stft import stft
from train.optim import optimizer
from util import device, playable, readme
import numpy as np
from scipy.signal import square, sawtooth

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1
)

class WavetableSynth(nn.Module):
    def __init__(self, n_tables=16, table_size=512):
        super().__init__()

        # sine = torch.sin(torch.linspace(-np.pi, np.pi, table_size))[None, :]
        # st = torch.from_numpy(sawtooth(np.linspace(-np.pi, np.pi, table_size))).float()[None, :]
        # sq = torch.from_numpy(square(np.linspace(-np.pi, np.pi, table_size))).float()[None, :]
        # tri = torch.from_numpy(
        #     np.concatenate([
        #         np.linspace(-1, 1, table_size // 2),
        #         np.linspace(1, -1, table_size // 2)
        #     ])
        # ).float()[None, :]

        self.n_tables = n_tables
        self.table_size = table_size

        # self.register_buffer('wavetables', torch.cat([
        #     sine, st, sq, tri
        # ], dim=0))

        self.wavetables = nn.Parameter(torch.zeros(n_tables, table_size).uniform_(-1, 1))

        self.table_choice = LinearOutputStack(
            exp.model_dim, 3, out_channels=n_tables * 16)
        self.to_env = LinearOutputStack(
            exp.model_dim, 3, out_channels=exp.n_frames)
        self.embed_pos = LinearOutputStack(exp.model_dim, 3, in_channels=33)
        self.to_frequency = LinearOutputStack(exp.model_dim, 3, out_channels=exp.n_frames)

    def get_tables(self):
        wt = self.wavetables
        mx, _ = torch.max(wt, dim=-1, keepdim=True)
        wt = wt / (mx + 1e-8)
        return wt

    def forward(self, x):
        batch = x.shape[0]

        # each vector should be turned into a full-length "event"
        x = x.view(-1, exp.model_dim)

        # TODO: Mixture of tables over time
        # c = self.table_choice(x)
        # c = torch.softmax(c, dim=-1)
        # values, indices = torch.max(c, dim=-1, keepdim=False)
        tc = self.table_choice(x).view(batch, self.n_tables, 16)

        wt = self.get_tables()

        # Selected tables should be (n_tables, n_samples)
        selected_tables = wt[indices][:, None, :]

        env = self.to_env(x).view(-1, 1, exp.n_frames)
        env = env ** 2
        env = F.interpolate(env, size=exp.n_samples, mode='linear')
        env = env.view(-1, exp.n_samples)

        freq = self.to_frequency.forward(x).view(batch, 1, -1)
        freq = freq ** 2
        freq = F.interpolate(freq, size=exp.n_samples, mode='linear')
        freq = torch.cumsum(freq, dim=-1) % 1
        stds = torch.zeros(1).fill_(0.01)
        
        sampling_kernel = pdf(torch.linspace(0, 1, self.table_size)[None, :, None], freq, stds).permute(0, 2, 1)
        print(sampling_kernel.shape) 

        sampled = (selected_tables @ sampling_kernel.permute(0, 2, 1))

        sampled = sampled * env[:, None, :] * values[:, None, None]
        return sampled, sampling_kernel


class Summarizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.context = DilatedStack(exp.model_dim, [1, 3, 9, 27, 1])
        self.reduce = nn.Conv1d(exp.model_dim + 33, exp.model_dim, 1, 1, 0)
        self.attend = nn.Linear(exp.model_dim, 1)

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
        x = x.permute(0, 2, 1)
        
        attn = torch.softmax(self.attend(x).view(batch, exp.n_frames), dim=-1)
        x, indices = sparsify_vectors(x, attn, 16, normalize=False)
        norms = torch.norm(x, dim=-1, keepdim=True)
        x = x / (norms + 1e-8)

        return x, indices


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.wt = WavetableSynth()
        self.summary = Summarizer()
        self.apply(lambda p: exp.init_weights(p))

    def forward(self, x):
        batch = x.shape[0]
        x, indices = self.summary(x)
        events, sk = self.wt.forward(x)
        events = events.view(-1, 16, exp.n_samples)

        output = torch.zeros(events.shape[0], 1, exp.n_samples * 2, device=x.device)

        for b in range(batch):
            for i in range(16):
                start = indices[b, i]
                end = start + exp.n_samples
                output[b, :, start: end] += events[b, i]

        x = output[..., :exp.n_samples]
        mx, _ = torch.max(x, dim=-1, keepdim=True)
        x = x / (mx + 1e-8)
        return x, sk


model = Model().to(device)
optim = optimizer(model)

# TODO: common base class, maybe?


def train(batch):
    optim.zero_grad()
    recon, sk = model.forward(batch)

    # spec = torch.abs(torch.fft.rfft(sk, dim=1, norm='ortho'))
    # full_mean = torch.mean(spec, dim=(1, 2), keepdim=True)
    # featurewise_mean = torch.mean(spec, dim=1, keepdim=True)
    # sk_loss = torch.abs(full_mean - featurewise_mean).sum()
    # full_mean = torch.mean(sk, dim=(1, 2), keepdim=True)
    # featurewise_mean = torch.mean(sk, dim=1, keepdim=True)
    # sk_loss + sk_loss + torch.abs(full_mean - featurewise_mean).sum()
    

    real_spec = stft(batch.view(-1, 1, exp.n_samples))
    fake_spec = stft(recon.view(-1, 1, exp.n_samples))
    loss = F.mse_loss(fake_spec, real_spec)
    loss.backward()
    optim.step()
    return recon, loss, sk

# TODO: Common base class


def make_sparse(x, to_keep, dim=-1):
    values, indices = torch.topk(x, to_keep, dim=dim)
    n = torch.zeros_like(x)
    n = torch.scatter(n, dim=-1, index=indices, src=values)
    return n

@readme
class WavetableSynthExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None
        self.sk = None

    def listen(self):
        return playable(self.fake, exp.samplerate)
    
    def kernel(self):
        return self.sk.data.cpu().numpy()

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.fake, loss, self.sk = train(item)
            print(loss.item())
