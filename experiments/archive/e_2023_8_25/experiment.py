
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.hypernetwork import HyperNetworkLayer
from modules.linear import LinearOutputStack
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    a_weighting=True)


class Block(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.dilation = dilation
        self.conv1 = nn.Conv1d(
            channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        orig = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x


class Stack(nn.Module):
    def __init__(self, channels, dilations):
        super().__init__()
        self.channels = channels
        self.dilation = dilations
        self.net = nn.Sequential(*[Block(channels, d) for d in dilations])
    
    def __iter__(self):
        return iter(self.net)
    
    def forward(self, x):
        x = self.net(x)
        return x

class Model(nn.Module):
    def __init__(self, channels, hyperchannels, hyperlatent, n_layers):
        super().__init__()
        self.channels = channels
        self.hyperchannels = hyperchannels
        self.hyperlatent = hyperlatent
        self.n_layers = n_layers

        self.norm = nn.LayerNorm((exp.n_samples, self.hyperchannels))

        self.embed_pos = nn.Linear(33, self.hyperchannels)

        self.p1 = nn.Parameter(torch.zeros(1, 33, 128).uniform_(-1, 1))
        self.p2 = nn.Parameter(torch.zeros(1, 33, exp.n_samples).uniform_(-1, 1))

        self.hyper = nn.ModuleDict({str(i): nn.Sequential(
            LinearOutputStack(
                channels, 3, out_channels=channels, norm=nn.LayerNorm((channels,))),
            HyperNetworkLayer(
                channels, hyperlatent, hyperchannels, hyperchannels)
        ) for i in range(n_layers)})
        self.embed = nn.Conv1d(exp.n_bands + 33, channels, 1, 1, 0)

        # encoder = nn.TransformerEncoderLayer(channels, 4, channels, batch_first=True)
        # self.context = nn.TransformerEncoder(encoder, 4, norm=nn.LayerNorm((128, channels)))
        self.context = Stack(channels, [1, 3, 9, 1, 3, 9, 1])

        # TODO: would it be better to include these weights in the hypernetwork?
        self.to_samples = nn.Linear(hyperchannels, 1)


        self.apply(lambda x: exp.init_weights(x))

    
    def forward(self, x):
        x = exp.pooled_filter_bank(x)
        batch, _, time = x.shape
        # pos = pos_encoded(batch, x.shape[-1], 16, device=x.device, channels_last=False)

        pos = self.p1.repeat(batch, 1, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.embed(x).permute(0, 2, 1).view(-1, time, self.channels)

        # x = self.context(x)
        x = x.permute(0, 2, 1)
        x = self.context(x)
        x = x.permute(0, 2, 1)

        # get latent for entire segment
        # x = torch.sum(x, dim=1)
        x = x[:, -1, :]

        # generate hypernetwork
        fwds = [m.forward(x) for m in self.hyper.values()]

        # generate pos encodings for audio samples
        # x = pos_encoded(batch, exp.n_samples, 16, device=x.device)
        x = self.p2.repeat(batch, 1, 1).permute(0, 2, 1)
        x = self.embed_pos(x)

        for i, pair in enumerate(fwds):
            w, func = pair
            x = func(x)
            if i < len(fwds) - 1:
                # x = torch.sin(x * 30)
                x = F.leaky_relu(x, 0.2)
                x = self.norm(x)
        
        
        x = self.to_samples(x)
        x = x.view(batch, -1, exp.n_samples)
        # x = max_norm(x)
        return x



model = Model(
    channels=256, 
    hyperchannels=256, 
    hyperlatent=16, 
    n_layers=8).to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    recon = model.forward(batch)

    loss = exp.perceptual_loss(recon, batch)

    loss.backward()
    optim.step()
    return loss, recon


# TODO: Make it easier to add view elements

@readme
class HyperNerf(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    