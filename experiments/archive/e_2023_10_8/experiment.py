
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.decompose import fft_resample
from modules.fft import fft_convolve
from modules.refractory import make_refractory_filter
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme
from conjure import numpy_conjure, SupportedContentType


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

n_events = 64

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed_period = nn.Linear(257, 8)
        

        self.embed_cond = nn.Conv1d(4096, 256, 1, 1, 0)
        self.embed_spec = nn.Conv1d(1024, 256, 1, 1, 0)

        self.net = nn.Sequential(

            # 64
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 32
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(256)
            ),

            # 8
            nn.Sequential(
                nn.Conv1d(256, 256, 7, 2, 3),
                nn.LeakyReLU(0.2),
            ),


            nn.Conv1d(256, 1, 1, 1, 0)            
        )

        self.apply(lambda x: exp.init_weights(x))

    def forward(self, cond, audio):
        batch_size = cond.shape[0]

        spec = exp.perceptual_feature(audio)

        x = self.embed_period(spec)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        spec = self.embed_spec(x)

        cond = self.embed_cond(cond)
        x = cond + spec
        j = self.net(x)
        return j


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Linear(257, 8)
        self.encoder = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 2), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 6), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 18), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((0, 2), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),


            nn.Conv1d(1024, 4096, 1, 1, 0)
        )
        self.down = nn.Conv1d(4096, 1024, 1, 1, 0)

        self.planner = nn.Sequential(
            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((2, 0), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((6, 0), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((18, 0), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Dropout(0.05),
                nn.ConstantPad1d((2, 0), 0),
                nn.Conv1d(1024, 1024, 3, 1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

        )
        self.down_again = nn.Conv1d(1024, 256, 1, 1, 0)

        self.synthesis = ConvUpsample(
            1024, 256, 128, exp.n_samples, mode='learned', out_channels=1, from_latent=False, batch_norm=True)

        
        self.refractory_period = 8
        self.register_buffer('refractory', make_refractory_filter(self.refractory_period, power=10, device=device))
        
        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):
        batch_size = x.shape[0]

        if len(x.shape) != 4:
            x = exp.perceptual_feature(x)

        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)

        # compute a sparse coding by only looking at future values
        encoded = self.encoder.forward(x)

        # ref = F.pad(self.refractory, (0, encoded.shape[-1] - self.refractory_period))
        # encoded = fft_convolve(encoded, ref)[..., :encoded.shape[-1]]
        # encoded = sparsify(encoded, n_to_keep=n_events)
        
        x = self.down(encoded)

        # plan output samples by only looking at past parts of the sparse code
        x = self.planner(x)
        x = self.down_again(x)
        x = self.synthesis(x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x, encoded


model = Model().to(device)
optim = optimizer(model, lr=1e-3)

disc = Discriminator().to(device)
disc_optim = optimizer(disc, lr=1e-3)


def perlin():
    scale = 1.0

    start = torch.zeros(1, 4096, 4, device=device).uniform_(0, scale)

    while start.shape[-1] < 128:
        # start = F.interpolate(start, scale_factor=2, mode='linear')
        start = fft_resample(start, desired_size=start.shape[-1] * 2, is_lowest_band=True)
        scale = scale / 2
        start = start + torch.zeros_like(start).uniform_(0, scale)
    
    start = sparsify(start, n_to_keep=n_events)
    return start

def train(batch, i):
    optim.zero_grad()
    disc_optim.zero_grad()

    # with torch.no_grad():
    #     feat = exp.perceptual_feature(batch)

    if i % 2 == 0:
        recon, encoded = model.forward(batch)

        j = disc.forward(encoded.clone().detach(), recon)
        adv_loss = (torch.abs(1 - j).mean() * 1)
        spec_loss = exp.perceptual_loss(recon, batch)

        loss = adv_loss + spec_loss

        loss.backward()
        optim.step()
        print('GEN', loss.item())
        return loss, recon, encoded
    else:
        
        with torch.no_grad():
            recon, encoded = model.forward(batch)
        
        rj = disc.forward(encoded, batch)
        fj = disc.forward(encoded, recon)
        loss = (torch.abs(1 - rj) + torch.abs(0 - fj)).mean()
        loss.backward()
        disc_optim.step()
        print('DISC', loss.item())
        return None, None, None

def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):
        x = x[:, None, :, :]
        x = F.max_pool2d(x, (16, 8), (16, 8))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        return x
    return (encoded,)

@readme
class Independent(BaseExperimentRunner):

    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)
            
            if l is None:
                continue

            self.real = item
            self.fake = r
            self.encoded = e
            # print(i, l.item())
            self.after_training_iteration(l)