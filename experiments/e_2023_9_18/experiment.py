
import torch
from conjure import numpy_conjure, SupportedContentType
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from modules.normalization import unit_norm
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
from train.experiment_runner import BaseExperimentRunner, MonitoredValueDescriptor
from train.optim import optimizer
from util import device
from util.readmedocs import readme

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class Contextual(nn.Module):
    def __init__(self, channels, n_layers=6, n_heads=4):
        super().__init__()
        self.channels = channels
        self.down = nn.Conv1d(channels + 33, channels, 1, 1, 0)
        encoder = nn.TransformerEncoderLayer(channels, n_heads, channels, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, n_layers, norm=nn.LayerNorm((128, channels)))

    def forward(self, x):
        pos = pos_encoded(x.shape[0], x.shape[-1], 16, device=x.device).permute(0, 2, 1)
        x = torch.cat([x, pos], dim=1)
        x = self.down(x)
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)
        return x
    
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
        self.reduce = nn.Conv1d(self.periodicity_embedding_dim * self.channels, self.channels, 1, 1, 0)
        # self.context = nn.Sequential(
        #     DilatedBlock(channels, 1),
        #     DilatedBlock(channels, 3),
        #     DilatedBlock(channels, 9),
        #     DilatedBlock(channels, 1),
        # )

        self.context = Contextual(channels)

        self.salience = nn.Conv1d(channels, channels, 1, 1, 0)
        self.act = nn.Conv1d(channels, channels, 1, 1, 0)
        self.up = nn.Conv1d(channels, encoding_channels, 1, 1, 0)


        self.atoms = nn.Parameter(torch.zeros(1, self.encoding_channels, self.atom_size).uniform_(-1, 1))

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
        # sal = self.salience(x)
        # sal = torch.softmax(sal.view(batch_size, -1), dim=-1).view(batch_size, self.channels, -1)
        # act = self.act(x)
        # x = sal * act

        
        x = self.up(x)


        encoding = x = sparsify(x, n_to_keep=128)

        # x = torch.relu(x)
        # encoding = x
        
        full = torch.zeros(batch_size, self.encoding_channels, exp.n_samples, device=x.device)
        ratio = exp.n_samples // x.shape[-1]
        full[:, :, ::ratio] = x

        atoms = unit_norm(self.atoms)
        atoms = F.pad(self.atoms, (0, exp.n_samples - self.atom_size))
        signal = fft_convolve(atoms, full)[..., :exp.n_samples]

        signal = torch.sum(signal, dim=1, keepdim=True)

        return signal, encoding

        
model = Model(
    channels=128, 
    encoding_channels=512, 
    atom_size=1024
).to(device)

optim = optimizer(model, lr=1e-3)



def train(batch, i):
    optim.zero_grad()

    feat = exp.perceptual_feature(batch)

    recon, encoding = model.forward(feat)

    loss = F.mse_loss(exp.perceptual_feature(recon), feat)

    loss.backward()
    optim.step()
    return loss, recon, encoding


def build_conjure_funcs(experiment: BaseExperimentRunner):
    
    @numpy_conjure(
            experiment.collection, 
            content_type=SupportedContentType.Spectrogram.value, 
            identifier='sparsefeaturemap')
    def sparsefeaturemap(x: torch.Tensor):
        x = x.data.cpu().numpy()
        x = x - x.min()
        x = x / (x.max() + 1e-12)
        x = x * 255
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
    