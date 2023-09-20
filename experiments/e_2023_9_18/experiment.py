
import numpy as np
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from modules.normalization import max_norm, unit_norm
from modules.pos_encode import pos_encoded
from modules.sparse import encourage_sparsity_loss, sparsify
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme
from torch import jit

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

    
class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
        self.norm = nn.BatchNorm1d(channels)
    
    def forward(self, x):
        x = F.dropout(x, 0.1)

        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, channels, encoding_channels, atom_size):
        super().__init__()
        self.channels = channels
        self.encoding_channels = encoding_channels
        self.periodicity_embedding_dim = 8
        self.atom_size = atom_size

        self.embed_periodicity = nn.Linear(257, self.periodicity_embedding_dim)
        self.reduce = nn.Conv1d(self.periodicity_embedding_dim * exp.n_bands, self.channels, 1, 1, 0)

        self.context = nn.Sequential(
            DilatedBlock(channels, 1),
            DilatedBlock(channels, 3),
            DilatedBlock(channels, 9),
            DilatedBlock(channels, 1),
        )


        self.salience = nn.Conv1d(channels, encoding_channels, 1, 1, 0)
        self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)
        self.atoms = nn.Parameter(
            torch.zeros(1, self.encoding_channels, self.atom_size).uniform_(-1, 1))

        self.apply(lambda x: exp.init_weights(x))
    
    def forward(self, x):

        if len(x.shape) == 3:
            x = exp.perceptual_feature(x)
        else:
            x = x
        
        # compute perceptual feature
        batch_size = x.shape[0]
        x = self.embed_periodicity(x)
        x = x.permute(0, 3, 1, 2).reshape(batch_size, self.periodicity_embedding_dim * exp.n_bands, -1)
        x = self.reduce(x)

        # gather context
        x = self.context(x)

        # sparsify
        sal = self.salience(x)
        sal = torch.softmax(sal.view(batch_size, -1), dim=-1).view(batch_size, self.encoding_channels, -1)

        x = encoding = self.up(x)
        x = F.dropout(x, 0.05)
        x = torch.relu(x)

        # local competition
        # x = x[:, None, :, :]
        # pooled = F.avg_pool2d(x, (9, 9), (1, 1), (4, 4))
        # x = x - pooled 
        # x = x.view(batch_size, self.encoding_channels, -1)

        # encoding = x = sparsify(
        #     x, 
        #     n_to_keep=32, 
        #     salience=sal * 25
        # )
        encoding = x

        d = encoding[0].sum(dim=-1)
        nz = torch.nonzero(d)
        print('alive', set(nz.data.cpu().numpy().reshape((-1,))))
        
        
        full = torch.zeros(batch_size, self.encoding_channels, exp.n_samples, device=x.device)
        ratio = exp.n_samples // x.shape[-1]
        full[:, :, ::ratio] = x

        # atoms = self.atoms
        # atoms = self.atoms * torch.hamming_window(self.atom_size, device=x.device)[None, None, :]
        # atoms = unit_norm(atoms, dim=-1)
        atoms = F.pad(self.atoms, (0, exp.n_samples - self.atom_size))
        signal = fft_convolve(atoms, full)[..., :exp.n_samples]

        signal = torch.sum(signal, dim=1, keepdim=True)

        return signal, encoding

        
model = Model(
    channels=256, 
    encoding_channels=512, 
    atom_size=4096
).to(device)

optim = optimizer(model, lr=1e-3)


def train(batch, i):
    optim.zero_grad()
    feat = exp.perceptual_feature(batch)
    recon, encoding = model.forward(feat)

    # e = torch.sum(encoding, dim=-1)
    # e = torch.softmax(e, dim=-1)
    # sim = e[:, None, :] * e[:, :, None]
    # sim = torch.triu(sim)
    # sim = sim.mean() * 100

    sp = encourage_sparsity_loss(encoding, 0, sparsity_loss_weight=0.00000005)

    # loss = F.mse_loss(exp.perceptual_feature(recon), feat) + sp 
    loss = F.mse_loss(recon, batch) + sp
    loss.backward()
    optim.step()

    # model.atoms.data[:] = unit_norm(model.atoms)

    # recon = max_norm(recon)

    return loss, recon, encoding


def build_conjure_funcs(experiment: BaseExperimentRunner):
    
    @numpy_conjure(
            experiment.collection, 
            content_type=SupportedContentType.Spectrogram.value, 
            identifier='sparsefeaturemap')
    def sparsefeaturemap(x: torch.Tensor):
        x = x.data.cpu().numpy()
        x = (x != 0).astype(np.float32)
        # x = x - x.min()
        # x = x / (x.max() + 1e-12)
        return x

    return (sparsefeaturemap,)


@readme
class SparseV4(BaseExperimentRunner):

    sfm = MonitoredValueDescriptor(build_conjure_funcs)

    def __init__(self, stream, port=None):
        super().__init__(stream, train, exp, port=port)
    
    def run(self):
        
        for i, item in enumerate(self.iter_items()):
            item = item.view(-1, 1, exp.n_samples).to(device)
            l, r, fm = train(item, i)

            self.fake = r
            self.real = item
            self.sfm = fm[0]

            print(l.item())
            self.after_training_iteration(l)
    