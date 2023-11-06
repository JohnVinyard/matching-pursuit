import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules import max_norm
from train import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2 ** 15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def kernel_at_freq(delay_size, hz, decay_exp):
    kernel = torch.zeros(1, 1, delay_size, device=device)
    spacing = int(exp.samplerate) // hz
    kernel[:, :, ::spacing] = 1

    env = torch.linspace(0, 1, delay_size, device=device) ** decay_exp
    kernel = kernel * env[None, None, :]
    return kernel


def cross_fade(kernel1, kernel2):
    size = kernel1.shape[-1]
    stacked = torch.concatenate([kernel1[..., None], kernel2[..., None]], dim=-1)
    up = torch.linspace(0, 1, size, device=kernel1.device)
    down = torch.linspace(1, 0, size, device=kernel1.device)
    envs = torch.concatenate([up[..., None], down[..., None]], dim=-1)
    x = (stacked * envs).sum(dim=-1)

    x = torch.cat([kernel1[..., :size // 2], kernel2[..., size // 2:]], dim=-1)
    return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.blah = nn.Parameter(torch.zeros(10).uniform_(-1, 1))

    def forward(self, x):

        impulse_size = 128
        delay_size = exp.n_samples

        noise = torch.zeros(1, 1, impulse_size, device=x.device).uniform_(-1, 1)
        signal = F.pad(noise, (0, exp.n_samples - noise.shape[-1]))

        k1 = kernel_at_freq(delay_size, 110, 1)
        k2 = kernel_at_freq(delay_size, 100, 1)
        kernel = cross_fade(k1, k2)

        # kernel = torch.zeros(1, 1, delay_size, device=x.device)
        # spacing = int(exp.samplerate) // 110
        # kernel[:, :, ::spacing] = 1
        #
        # env = torch.linspace(0, 1, delay_size, device=x.device) ** 1
        # kernel = kernel * env[None, None, :]

        wet = F.conv1d(signal, kernel, stride=1, padding=delay_size // 2)[..., :exp.n_samples]

        final = signal + wet

        filtered = F.conv1d(
            final,
            exp.fb.filter_bank[:10].sum(dim=0).view(1, 1, exp.kernel_size),
            padding=exp.kernel_size // 2)[..., :exp.n_samples]

        final = max_norm(filtered)
        final.requires_grad = True
        return final.view(1, 1, exp.n_samples)


model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = F.mse_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon


@readme
class GraphRepresentation4(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
