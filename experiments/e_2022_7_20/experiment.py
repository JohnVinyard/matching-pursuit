from torch import nn
import torch
from modules.pos_encode import pos_encoded
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from torch.nn import functional as F
import zounds

from util.weight_init import make_initializer

n_samples = 2 ** 14
samplerate = zounds.SR22050()
model_dim = 128

n_steps = 25

positions = pos_encoded(1, n_steps, 16).view(n_steps, 33).to(device)

init_weights = make_initializer(0.1)

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
    def forward(self, x, step, cond=None):
        batch = x.shape[0]

        skip = x

        step = step.view(batch, self.channels, 1)

        x = x + step

        scale = self.scale(x + cond)
        gate = self.gate(x + cond)

        x = torch.tanh(scale) * torch.sigmoid(gate)

        out = self.out(x)
        next = self.next(x) + skip

        return next, out

class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        c = channels

        self.initial = nn.Conv1d(1, channels, 1, 1, 0)
        self.embed_step = nn.Conv1d(33, channels, 1, 1, 0)

        self.stack = nn.Sequential(
            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 27),
            DilatedBlock(c, 81),
            DilatedBlock(c, 243),

            DilatedBlock(c, 1),
            DilatedBlock(c, 3),
            DilatedBlock(c, 9),
            DilatedBlock(c, 27),
            DilatedBlock(c, 81),
            DilatedBlock(c, 243),

            DilatedBlock(c, 1),
        )

        self.final = nn.Sequential(
            nn.Conv1d(c, c, 1, 1, 0),
            nn.LeakyReLU(0.2),
            nn.Conv1d(c, 1, 1, 1, 0),
        )
    
    def forward(self, x, step, cond=None):
        batch = x.shape[0]

        n = F.leaky_relu(self.initial(x), 0.2)

        if cond is None:
            cond = torch.zeros(1, self.channels, 1, device=x.device)

        step = step.view(batch, 33, 1)
        step = self.embed_step(step)

        outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)

        for layer in self.stack:
            n, o = layer.forward(n, step, cond)
            outputs = outputs + (o / len(self.stack))
        
        x = self.final(outputs)
        return x

model = Model(model_dim).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch):
    optim.zero_grad()
    pos = torch.randint(0, n_steps, (batch.shape[0],))
    p = positions[pos]
    pred = model.forward(batch, p)
    print(pred.shape)

@readme
class DiffusionExperimentTwo(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
    
    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            train(item)