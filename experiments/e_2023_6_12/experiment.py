
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.refractory import make_refractory_filter
from modules.reverb import ReverbGenerator
from modules.sparse import encourage_sparsity_loss
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512,
    a_weighting=False,
    windowed_pif=True,
    norm_periodicities=False)


features_per_band = 8


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
    
    def forward(self, x):
        return F.interpolate(x, scale_factor=4, mode='nearest')


class Model(nn.Module):
    def __init__(self):
        super().__init__()


        self.embed = nn.Linear(257, features_per_band)

        self.salience = nn.Conv1d(1024, 1024, 1, 1, 0)

        
        self.context = nn.Sequential(
            nn.Sequential(
                nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Conv1d(1024, 1024, 3, 1, padding=3, dilation=3),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Conv1d(1024, 1024, 3, 1, padding=9, dilation=9),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Conv1d(1024, 1024, 3, 1, padding=1, dilation=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(1024)
            ),

            nn.Sequential(
                nn.Conv1d(1024, 1024, 7, 4, 3),
                # nn.LeakyReLU(0.2),
                # nn.BatchNorm1d(1024)
            )
        )


        self.verb = ReverbGenerator(1024, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((1024,)))

        
        

        self.up = nn.Sequential(
            nn.Conv1d(1024, 128, 1, 1, 0),
            ConvUpsample(
                128, 128, 128, exp.n_samples, mode='nearest', out_channels=1024, from_latent=False, batch_norm=True)
        )

        self.atom_size = 16384


        band = zounds.FrequencyBand(40, 2000)
        scale = zounds.MelScale(band, 1024)
        bank = morlet_filter_bank(exp.samplerate, self.atom_size, scale, 0.25, normalize=True).real.astype(np.float32)
        bank = torch.from_numpy(bank)

        self.atoms = nn.Parameter(bank)
        # self.atoms = nn.Parameter(
        #     unit_norm(torch.zeros(1, 1024, self.atom_size).uniform_(-1, 1))
        # )

        self.register_buffer('refractory', make_refractory_filter(32, power=10, device=device, channels=1024))

        self.norm = nn.LayerNorm((1024, 128))


        self.apply(lambda x: exp.init_weights(x))
    
    def embed_features(self, x, iteration):
        encoding = None

        # torch.Size([16, 128, 128, 257])
        batch, channels, time, period = x.shape
        x = self.embed(x) # (batch, channels, time, 8)

        x = x.permute(0, 3, 1, 2).reshape(batch, 8 * channels, time)

        # gather context
        x = self.context(x)

        # sparsify
        salience = F.dropout(x, 0.1)
        salience = self.salience(salience)
        salience = torch.relu(salience)
        encoding = x = salience

        return x, encoding

    def generate(self, x):

        size = x.shape[-1]

        # ref = F.pad(self.refractory, (0, size - 32))
        # x = fft_convolve(x, ref)[..., :size]

        # upsample to audio sample rate
        rate = x.shape[-1]
        step = exp.n_samples // rate
        new_x = torch.zeros(x.shape[0], x.shape[1], exp.n_samples, device=x.device)
        new_x[:, :, ::step] = x
        x = new_x

        x = torch.relu(x)

        atoms = self.atoms
        atoms = F.pad(atoms, (0, exp.n_samples - self.atom_size))

        x = fft_convolve(atoms, x)
        x = torch.sum(x, dim=1, keepdim=True)
        return x
    
    def forward(self, x, iteration):

        # torch.Size([16, 128, 128, 257])
        encoded, encoding = self.embed_features(x, iteration)
        
        ctx = torch.sum(encoded, dim=-1)

        x = self.generate(encoded)

        x = self.verb.forward(ctx, x)

        return x, encoding

model = Model().to(device)
optim = optimizer(model, lr=1e-3)



def train(batch, i):
    batch_size = batch.shape[0]

    optim.zero_grad()

    with torch.no_grad():
        feat = exp.perceptual_feature(batch)

    recon, encoding = model.forward(feat, i)
    r = exp.perceptual_feature(recon)

    sparsity_loss = encourage_sparsity_loss(encoding, 0, sparsity_loss_weight=0.0000115) 

    loss = F.mse_loss(r, feat) + sparsity_loss

    loss.backward()
    optim.step()

    encoding = (encoding != 0).float()

    recon = max_norm(recon)

    return loss, recon, encoding
    
def make_conjure(experiment: BaseExperimentRunner):

    @numpy_conjure(experiment.collection, SupportedContentType.Spectrogram.value)
    def encoding(x: torch.Tensor):
        return x[0].data.cpu().numpy()
    
    return (encoding,)

@readme
class PhaseInvariantFeatureInversion(BaseExperimentRunner):

    encoding_view = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    

    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            self.real = item
            l, r, e = train(item, i)
            self.fake = r
            self.encoding_view = e
            print(l.item())
            self.after_training_iteration(l)
    
    
            

                
    
    