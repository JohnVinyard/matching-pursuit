import numpy as np
import zounds
import torch
from torch import nn
from modules.pif import AuditoryImage
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
from torch.nn import functional as F

from util.weight_init import make_initializer

n_samples = 2 ** 15
samplerate = zounds.SR22050()

atom_size = 512
n_atoms = 512

kernel_size = 128
n_bands = 128

model_dim = 64

atoms_to_keep = 256

sample_index_factor = n_samples // 128

band = zounds.FrequencyBand(40, samplerate.nyquist)
scale = zounds.MelScale(band, n_bands)
fb = zounds.learn.FilterBank(
    samplerate,
    kernel_size,
    scale,
    0.1,
    normalize_filters=True,
    a_weighting=False).to(device)

aim = AuditoryImage(512, 128, do_windowing=False, check_cola=False)

init_weights = make_initializer(0.05)


def perceptual_feature(x):
    # x = fb.forward(x, normalize=False)
    # x = aim.forward(x)
    return x


def perceptual_loss(a, b):
    a = perceptual_feature(a)
    b = perceptual_feature(b)
    loss = F.mse_loss(a, b)
    return loss



class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.channels = channels
        self.out = nn.Conv1d(channels, channels, 1, 1, 0)
        self.next = nn.Conv1d(channels, channels, 1, 1, 0)
        self.scale = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
        self.gate = nn.Conv1d(channels, channels, 3, 1, dilation=dilation, padding=dilation)
    
    def forward(self, x):
        batch = x.shape[0]
        skip = x
        scale = self.scale(x)
        gate = self.gate(x)
        x = torch.tanh(scale) * F.sigmoid(gate)
        out = self.out(x)
        next = self.next(x) + skip
        return next, out

class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        c = channels

        self.initial = nn.Conv1d(1, channels, 7, 1, 3)

        self.embed_step = nn.Conv1d(33, channels, 1, 1, 0)

        self.stack = nn.Sequential(
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
            nn.Conv1d(c, n_atoms, 1, 1, 0),
        )
    
        self.apply(init_weights)
    
    def forward(self, x):
        batch = x.shape[0]

        n = F.leaky_relu(self.initial(x), 0.2)


        outputs = torch.zeros(batch, self.channels, x.shape[-1], device=x.device)
        for layer in self.stack:
            n, o = layer.forward(n)
            outputs = outputs + o
        
        x = self.final(outputs)
        return x


class SparseAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Model(model_dim)
        self.atoms = nn.Parameter(torch.zeros(
            n_atoms, atom_size).uniform_(-1, 1))
        self.apply(init_weights)

    def unit_norm_atoms(self):
        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-12)
        return atoms

    def orthogonal_loss(self):
        x = self.unit_norm_atoms().view(n_atoms, atom_size)
        sim = x @ x.T
        return sim.mean()
    
    def clip_atoms(self):
        atoms = self.unit_norm_atoms()
        self.atoms.data[:] = atoms
    
    def from_features(self, x, sparse=False):
        # norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        # atoms = self.atoms / (norms + 1e-8)
        atoms = self.atoms.reshape(n_atoms, 1, atom_size)

        if sparse:
            values, indices = torch.topk(x, atoms_to_keep, -1)
            f = torch.zeros_like(x)
            f = torch.scatter(f, -1, indices, values)
            output = f
        else:
            output = x
        
        output = F.pad(output, (0, 1))
        output = F.conv_transpose1d(output, atoms, padding=atom_size // 2)
        return output
    
    def compute_features(self, x):
        output = self.model(x)
        return output
    
    
    def forward(self, x):
        feat = self.compute_features(x)
        full = self.from_features(feat, sparse=False)
        sparse = self.from_features(feat, sparse=True)
        return feat, full, sparse


model = SparseAudioModel().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    feat, full, sparse = model.forward(batch)

    # TODO: If this doesn't work well, consider
    # trying the earlier l1 norm of all but top-k
    # approach
    recon_loss = perceptual_loss(full, batch)
    sparse_loss = perceptual_loss(sparse, full)


    loss = recon_loss + sparse_loss
    loss.backward()
    optim.step()
    model.clip_atoms()
    return loss, full, sparse, feat


@readme
class SparseAudioExperimentWithSparsityConstraint(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None

        self.sparse = None
        self.fake = None
        self.features = None

    def r(self):
        return playable(self.real, samplerate)

    def listen(self, sparse=False):
        if sparse:
            return playable(self.sparse, samplerate)
        else:
            return playable(self.fake, samplerate)
    
    def feat(self):
        with torch.no_grad():
            x = F.avg_pool1d(self.features, 512, 256, 256)
            return np.abs(x.data.cpu().numpy()[0])

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss, self.fake, self.sparse, self.features = train(item)
            print(loss.item())
