import torch
from torch import nn
from torch.nn import functional as F
from config import Experiment
import zounds
from modules import LinearOutputStack
from modules.dilated import DilatedStack
from modules.pos_encode import pos_encoded
from modules.sparse import ElementwiseSparsity, VectorwiseSparsity, sparsify_vectors
from modules.stft import stft
from train.optim import optimizer
from util import device, playable, readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.02
)


class WavetableSynth(nn.Module):
    def __init__(self, n_tables=8, table_size=512):
        super().__init__()

        self.n_tables = n_tables
        self.table_size = table_size

        self.wavetables = nn.Parameter(torch.zeros(
            n_tables, table_size).uniform_(-1, 1))
        self.table_choice = LinearOutputStack(
            exp.model_dim, 3, out_channels=n_tables)
        self.to_env = LinearOutputStack(
            exp.model_dim, 3, out_channels=exp.n_frames)
        self.embed_pos = LinearOutputStack(exp.model_dim, 3, in_channels=33)
        self.to_sampling_kernel = LinearOutputStack(
            exp.model_dim, 3, out_channels=self.table_size)

    def get_tables(self):
        wt = self.wavetables
        mx, _ = torch.max(wt, dim=-1, keepdim=True)
        wt = wt / (mx + 1e-8)
        return wt

    def forward(self, x):
        # each vector should be turned into a full-length "event"
        x = x.view(-1, exp.model_dim)

        c = self.table_choice(x)
        c = torch.softmax(c, dim=-1)
        values, indices = torch.max(c, dim=-1, keepdim=False)

        wt = self.get_tables()

        selected_tables = wt[indices][:, None, :]

        env = self.to_env(x).view(-1, 1, exp.n_frames)
        env = torch.abs(env)
        env = F.interpolate(env, size=exp.n_samples, mode='linear')
        env = env.view(-1, exp.n_samples)

        pos = pos_encoded(x.shape[0], exp.n_samples, 16, device=x.device)
        pos = self.embed_pos(pos)

        # TODO: Would a weight and bias work better here?
        pos = pos + x[:, None, :]
        sampling_kernel = self.to_sampling_kernel(pos)
        sampling_kernel = torch.softmax(sampling_kernel, dim=-1)

        sampled = (selected_tables @ sampling_kernel.permute(0, 2, 1))

        sampled = sampled * env[:, None, :] * values[:, None, None]
        return sampled


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
        attn = torch.softmax(self.attend(x).view(batch, exp.n_frames), dim=-1)
        x = x.permute(0, 2, 1)
        x, indices = sparsify_vectors(x, attn, 16, normalize=False)

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
        events = self.wt.forward(x).view(-1, 16, exp.n_samples)

        output = torch.zeros(events.shape[0], 1, exp.n_samples * 2, device=x.device)


        for b in range(batch):
            for i in range(16):
                start = indices[b, i]
                end = start + exp.n_samples
                output[b, :, start: end] += events[b, i]

        x = output[..., :exp.n_samples]
        mx, _ = torch.max(x, dim=-1, keepdim=True)
        x = x / (mx + 1e-8)
        return x


model = Model().to(device)
optim = optimizer(model)

# TODO: common base class, maybe?


def train(batch):
    optim.zero_grad()
    recon = model.forward(batch)
    print(batch.shape, recon.shape)
    real_spec = stft(batch.view(-1, 1, exp.n_samples))
    fake_spec = stft(recon.view(-1, 1, exp.n_samples))
    loss = F.mse_loss(fake_spec, real_spec)
    loss.backward()
    optim.step()
    return recon, loss

# TODO: Common base class


@readme
class WavetableSynthExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.fake = None

    def listen(self):
        return playable(self.fake, exp.samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, exp.n_samples)

            self.fake, loss = train(item)
            print(loss.item())
