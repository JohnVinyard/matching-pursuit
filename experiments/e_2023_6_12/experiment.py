
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.atoms import unit_norm
from modules.phase import AudioCodec, MelScale
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample, FFTUpsampleBlock
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


class UpsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        batch, channels, time = x.shape

        return F.interpolate(x, scale_factor=4, mode='nearest')
        
        new_time = time * 4
        new_coeffs = new_time // 2 + 1

        spec = torch.fft.rfft(x, dim=-1, norm='ortho')
        new_spec = torch.zeros(batch, channels, new_coeffs, dtype=spec.dtype, device=spec.device)
        new_spec[:, :, :spec.shape[-1]] = spec

        result = torch.fft.irfft(new_spec, dim=-1, norm='ortho')
        return result

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(257, 8)

        self.prepare = nn.Sequential(
            nn.Conv1d(1024, 256, 3, padding=1, dilation=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 256, 3, padding=3, dilation=3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 256, 3, padding=9, dilation=9),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 256, 3, padding=27, dilation=27),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),

            nn.Conv1d(256, 1024, 1, 1, 0)
        )

        self.up = nn.Sequential(

            nn.Conv1d(1024, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            # nn.Upsample(scale_factor=4, mode='nearest'), # 512
            UpsampleBlock(),

            nn.Conv1d(512, 256, 7, 1, 3),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            # nn.Upsample(scale_factor=4, mode='nearest'), # 2048
            UpsampleBlock(),

            nn.Conv1d(256, 128, 7, 1, 3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            # nn.Upsample(scale_factor=4, mode='nearest'), # 8192
            UpsampleBlock(),

            nn.Conv1d(128, 64, 7, 1, 3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            # nn.Upsample(scale_factor=4, mode='nearest'), # 32768
            UpsampleBlock(),

            nn.Conv1d(64, 1, 7, 1, 3),
        )        
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        # torch.Size([16, 128, 128, 257])
        batch, channels, time, period = x.shape
        x = self.embed(x).permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)

        # sparsify
        # x = self.prepare(x)
        # x = sparsify(x, n_to_keep=128, return_indices=False)

        
        x = self.up(x)
        return x

model = Model().to(device)
optim = optimizer(model, lr=1e-3)

def train(batch, i):
    optim.zero_grad()
    with torch.no_grad():
        spec = exp.perceptual_feature(batch)

    recon = model.forward(spec)
    recon_spec = exp.perceptual_feature(recon)
    loss = F.mse_loss(recon_spec, spec)
    loss.backward()
    optim.step()
    return loss, recon

@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):
    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    