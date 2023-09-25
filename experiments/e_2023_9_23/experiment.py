
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from fft_basis import morlet_filter_bank
from modules.fft import fft_convolve
from modules.normalization import max_norm
from modules.reverb import ReverbGenerator
from modules.sparse import encourage_sparsity_loss, sparsify
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from upsample import ConvUpsample
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    scaling_factor=np.linspace(0.01, 0.5, 128),
    kernel_size=512)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        # x = F.dropout(x, 0.1)
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(257, 8)
        self.encoder = nn.Sequential(
            DilatedBlock(1024, 1),
            DilatedBlock(1024, 3),
            DilatedBlock(1024, 9),
            DilatedBlock(1024, 1),
            nn.Conv1d(1024, 4096, 1, 1, 0)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(4096, 1024, 1, 1, 0),
            # Adding dilated blocks here seems to harm performance
            # DilatedBlock(1024, 1),
            # DilatedBlock(1024, 3),
            # DilatedBlock(1024, 9),
            # DilatedBlock(1024, 1),
            nn.Conv1d(1024, 256, 1, 1, 0)
        )

        self.up = ConvUpsample(
            256, 
            256, 
            128, 
            exp.n_samples, 
            mode='nearest', 
            out_channels=256, 
            from_latent=False, 
            batch_norm=True)
        
        self.n_long_atoms = 512
        self.n_short_atoms = 512

        self.long_atoms_size = 8192
        self.short_atoms_size = 128


        self.to_short = nn.Conv1d(256, self.n_short_atoms, 1, 1, 0)
        self.to_long = nn.Conv1d(256, self.n_long_atoms, 1, 1, 0)


        
        band = zounds.FrequencyBand(20, 2000)
        scale = zounds.MelScale(band, self.n_long_atoms)
        bank = morlet_filter_bank(exp.samplerate, self.long_atoms_size, scale, 0.1, normalize=True).real.astype(np.float32)
        bank = torch.from_numpy(bank)
        noise = torch.zeros_like(bank).uniform_(-1, 1)
        band = fft_convolve(noise, bank)[..., :self.long_atoms_size]

        self.long_atoms = nn.Parameter(
            band / torch.abs(band).max()
        )
        
        self.short_atoms = nn.Parameter(
            torch.zeros(self.n_short_atoms, self.short_atoms_size).uniform_(-1, 1)
        )

        self.embed_verb = nn.Linear(4096, 32)
        self.verb = ReverbGenerator(32, 3, exp.samplerate, exp.n_samples, norm=nn.LayerNorm((32,)))

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):

        batch_size = x.shape[0]

        if len(x.shape) != 4:
            x = exp.perceptual_feature(x)
        
        x = self.embed(x)
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(batch_size, 8 * exp.n_bands, -1)
        encoded = self.encoder.forward(x)

        encoded = sparsify(encoded, n_to_keep=512)

        ctxt = torch.sum(encoded, dim=-1)
        ctxt = self.embed_verb(ctxt)

        decoded = self.decoder.forward(encoded)

        final = self.up.forward(decoded)

        la = self.to_long(final)

        sa = self.to_short(final)


        long_atoms = F.pad(self.long_atoms, (0, exp.n_samples - self.long_atoms_size))
        short_atoms = F.pad(self.short_atoms, (0, exp.n_samples - self.short_atoms_size))

        long = fft_convolve(long_atoms, la)[..., :exp.n_samples]
        short = fft_convolve(short_atoms, sa)[..., :exp.n_samples]

        # This sounds WORLDS better
        # atoms = F.pad(self.atoms, (0, exp.n_samples - 4096))
        # final = fft_convolve(atoms, final)[..., :exp.n_samples]
        # final = torch.sum(final, dim=1, keepdim=True)

        final = torch.sum(long + short, dim=1, keepdim=True)

        final = self.verb.forward(ctxt, final)

        # final = F.pad(final, (0, 1))
        # final = exp.fb.transposed_convolve(final * 0.001)[..., :exp.n_samples]

        return final, encoded
        
    
        
model = Model().to(device)
optim = optimizer(model, lr=1e-3)


def train(batch, i):
    optim.zero_grad()

    with torch.no_grad():
        feat = exp.perceptual_feature(batch)
    
    recon, encoded = model.forward(feat)


    # This is perceptually/subjectively better    
    # a, b, c = exp.perceptual_triune(recon)
    # d, e, f = exp.perceptual_triune(batch)
    # loss = (F.mse_loss(a, d) * 10) + F.mse_loss(b, e) + (F.mse_loss(c, f) * 10)

    # This feature captures more dynamics, but is responsible for the
    # griding audio
    r = exp.perceptual_feature(recon)
    loss = F.mse_loss(r, feat)


    loss.backward()
    optim.step()

    recon = max_norm(recon)

    return loss, recon, encoded


def make_conjure(experiment: BaseExperimentRunner):
    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def encoded(x: torch.Tensor):

        x = x[:, None, :, :]
        x = F.max_pool2d(x, (4, 4), (4, 4))
        x = x.view(x.shape[0], x.shape[2], x.shape[3])
        x = x.data.cpu().numpy()[0]
        x = (x != 0).astype(np.float32)
        return x
    
    return (encoded,)

@readme
class SparseV5(BaseExperimentRunner):

    encoded = MonitoredValueDescriptor(make_conjure)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples)
            l, r, e = train(item, i)
            self.real = item
            self.fake = r
            self.encoded = e
            print(l.item())
            self.after_training_iteration(l)