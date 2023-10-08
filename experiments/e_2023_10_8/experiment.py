
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.mixer import MixerStack
from modules.stft import stft
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 64

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.analysis = MixerStack(257, 256, sequence_length=128, layers=4, attn_blocks=4)

        self.up = nn.Conv1d(256, 4096, 1, 1, 0)
        self.down = nn.Conv1d(4096, 256, 1, 1, 0)

        self.planner = MixerStack(256, 256, sequence_length=128, layers=4, attn_blocks=4)

        self.synthesis = ConvUpsample(
            256, 256, 128, exp.n_samples, mode='learned', out_channels=1, from_latent=False, batch_norm=True)
        
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        spec = stft(x, 512, 256, pad=True).view(-1, 128, 257)
        x = self.analysis(spec)

        x = x.permute(0, 2, 1)
        x = self.up(x)
        x = self.down(x)
        x = x.permute(0, 2, 1)

        x = self.planner(x)
        x = x.permute(0, 2, 1)

        x = self.synthesis(x)

        x = torch.sum(x, dim=1, keepdim=True)
        return x

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)
    loss = exp.perceptual_loss(recon, batch)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class Independent(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    