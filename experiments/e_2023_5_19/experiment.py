
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.normalization import max_norm
from modules.overfitraw import OverfitRawAudio
from modules.pointcloud import CanonicalOrdering
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from modules.phase import MelScale, AudioCodec
from modules.sparse import to_key_points
from modules.stft import stft

scale = MelScale()
codec = AudioCodec(scale)

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


# ordering = CanonicalOrdering(3).to(device)

def transform(signal):
    spec = codec.to_frequency_domain(signal.view(-1, exp.n_samples))
    spec = torch.abs(spec[..., 0])
    kp = to_key_points(spec, n_to_keep=256)
    # kp = ordering.forward(kp)
    return kp

def loss(x, y):
    x = transform(x)
    y = transform(y)
    return F.mse_loss(x, y)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent = nn.Parameter(torch.zeros(1, 128).normal_(0, 1))
        self.net = ConvUpsample(128, 128, 8, exp.n_samples, mode='learned', out_channels=1, batch_norm=False)
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        output = self.net(self.latent)
        output = max_norm(output)
        return output

# model = Model().to(device)
model = OverfitRawAudio((1, exp.n_samples), std=1e-1, normalize=False).to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    recon = model.forward(None)
    l = loss(recon, batch)
    l.backward()
    optim.step()
    return l, recon


@readme
class KeyPointLoss(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    