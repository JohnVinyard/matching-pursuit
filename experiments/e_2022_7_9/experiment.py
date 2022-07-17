import zounds
import torch
from torch import nn
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
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

model_dim = 128

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

init_weights = make_initializer(0.1)


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
        self.dilated = nn.Conv1d(
            channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)

    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        return x


class SparseAudioModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = LinearOutputStack(
            model_dim, 3, in_channels=257, out_channels=8)
        self.conv = nn.Conv1d(model_dim * 8, model_dim, 1, 1, 0)

        self.atoms = nn.Parameter(torch.zeros(
            n_atoms, atom_size).uniform_(-1, 1))

        self.context = nn.Sequential(
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            DilatedBlock(model_dim, 9),
        )

        self.up = nn.Sequential(
            nn.Upsample(scale_factor=4),
            nn.Conv1d(model_dim, model_dim, 7, 1, 3),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=4),
            nn.Conv1d(model_dim, model_dim, 7, 1, 3),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=4),
            nn.Conv1d(model_dim, model_dim, 7, 1, 3),
            nn.LeakyReLU(0.2),

            nn.Upsample(scale_factor=4),
            nn.Conv1d(model_dim, n_atoms, 7, 1, 3),
        )

        self.to_values = nn.Conv1d(n_atoms, n_atoms, 1, 1, 0)
        self.apply(init_weights)

    def unit_norm_atoms(self):
        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-12)
        return atoms

    def orthogonal_loss(self):
        x = self.unit_norm_atoms().view(n_atoms, atom_size)
        sim = x @ x.T
        return sim.mean()
    
    def forward(self, x):
        x = fb.forward(x, normalize=False)
        x = aim.forward(x)
        x = self.reduce(x)
        x = x.permute(0, 3, 2, 1).reshape(-1, model_dim * 8, 128)
        x = self.conv(x)
        x = self.context(x)
        x = self.up(x)

        # v = torch.abs(self.to_values(x).view(x.shape[0], -1))
        
        
        # x = torch.softmax(x, 1)
        x = x.view(x.shape[0], -1)
        values, indices = torch.topk(x, atoms_to_keep, dim=-1)
        # v = torch.gather(v, -1, indices)
        v = values

        output = torch.zeros(x.shape[0], n_atoms * n_samples).to(device)
        output = output.scatter(-1, indices, v)
        output = output.reshape(-1, n_atoms, n_samples)

        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-8)
        atoms = atoms.reshape(n_atoms, 1, atom_size)

        output = F.pad(output, (0, 1))
        output = F.conv_transpose1d(output, atoms, padding=atom_size // 2)

        return output


model = SparseAudioModel().to(device)
optim = optimizer(model, lr=1e-4)


def train(batch):
    optim.zero_grad()
    result = model.forward(batch)
    loss = perceptual_loss(result, batch) + model.orthogonal_loss()
    loss.backward()
    optim.step()
    return loss, result


@readme
class SparseAudioExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.real = None
        self.fake = None

    def r(self):
        return playable(self.real, samplerate)

    def listen(self):
        return playable(self.fake, samplerate)

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.real = item
            loss, self.fake = train(item)

            print(loss.item())
