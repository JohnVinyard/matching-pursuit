
import zounds
from config.experiment import Experiment
from modules.activation import Sine
from modules.ddsp import AudioModel
from modules.decompose import fft_frequency_recompose
from modules.dilated import DilatedStack
from modules.linear import LinearOutputStack
from modules.normalization import ExampleNorm
from modules.perceptual import PerceptualAudioModel
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from upsample import ConvUpsample
from torch import nn
import torch
from torch.nn import functional as F

from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.05,
    model_dim=128,
    kernel_size=512)


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_periodicity = LinearOutputStack(exp.model_dim, 3, out_channels=8)

        self.audio = \
            nn.Sequential(
                DilatedStack(exp.model_dim, [1, 3, 9, 1]), 
                AudioModel(exp.n_samples, exp.model_dim, exp.samplerate, exp.n_frames, exp.n_frames * 4)
            )

        self.embed = nn.Conv1d(1024, exp.model_dim, 1, 1, 0)


        self.upsample = ConvUpsample(
            exp.model_dim, 
            512, 
            128, 
            end_size=exp.n_samples, 
            out_channels=1, 
            from_latent=False,
            mode='learned',
            activation_factory=lambda: Sine()
        )

        
        band_sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768]

        self.bands = nn.ModuleDict(
            {str(k): nn.Conv1d(exp.model_dim, exp.model_dim, 7, 1, 3) for k in band_sizes})
        
        self.apply(lambda x: exp.init_weights(x))

    def forward(self, pooled, spec):
        x = self.embed_periodicity(spec)
        x = x.permute(0, 3, 1, 2).reshape(-1, exp.model_dim * 8, pooled.shape[-1])
        x = self.embed(x)

        return self.audio(x)

        # x = self.upsample(x)

        output = {}

        for layer in self.upsample:
            x = layer(x)
            key = str(x.shape[-1])

            try:
                to_samples = self.bands[key]
                s = to_samples.forward(x)
                x = x - s
                samples = exp.fb.transposed_convolve(s)
                output[key] = samples
            except KeyError:
                pass
        
        x = fft_frequency_recompose(output, exp.n_samples)
        return x


loss_model = PerceptualAudioModel(exp, norm_second_order=False).to(device)

model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch):
    optim.zero_grad()

    with torch.no_grad():
        r_pooled, r_spec = loss_model.forward(batch)

    recon = model.forward(r_pooled, r_spec)

    f_pooled, f_spec = loss_model.forward(recon)

    loss = F.mse_loss(f_pooled, r_pooled) + F.mse_loss(f_spec, r_spec)
    loss.backward()
    optim.step()

    return loss, recon

@readme
class PerceptualFeatureDecoder(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
