import torch
from torch import nn
from torch.nn.init import orthogonal_, normal_
from torch.nn import functional as F
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from modules.scattering import batch_fft_convolve
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
import zounds

samplerate = zounds.SR22050()
pif = PsychoacousticFeature(kernel_sizes=[128] * 6).to(device)

n_samples = 2 ** 15

# [32768, 16384, 8192, 4096, 2048, 1024]
band_sizes = [2 ** (15 - i) for i in range(6)]
atoms_counts = {
    1024: 64,
    2048: 64,
    4096: 64,
    8192: 64,
    16384: 64,
    32768: 64
}

n_rooms = 8


class AtomCollection(nn.Module):
    def __init__(
            self,
            n_atoms,
            kernel_size,
            atoms_to_keep,
            atoms_to_apply=None):

        super().__init__()
        self.n_atoms = n_atoms
        self.kernel_size = kernel_size
        self.atoms_to_keep = atoms_to_keep
        self.atoms_to_apply = atoms_to_apply or n_atoms

        self.atoms = nn.Parameter(
            torch.zeros(n_atoms, 1, self.kernel_size).normal_(0, 0.01))

    def unit_norm_atoms(self):
        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-12)
        return atoms

    def orthogonal_loss(self):
        x = self.unit_norm_atoms().view(self.n_atoms, self.kernel_size)
        sim = x @ x.T
        return sim.mean()

    def forward(self, x):

        batch_size = x.shape[0]

        # give atoms unit norm
        atoms = self.unit_norm_atoms()

        # use only a subset of the atoms
        atom_indices_to_apply = torch.randperm(
            self.n_atoms)[:self.atoms_to_apply]
        atoms = self.atoms[atom_indices_to_apply]

        # frequency-domain convolution
        # atom_spec = torch.fft.rfft(torch.flip(atoms, dims=(-1,)), dim=-1, norm='ortho')
        # signal_spec = torch.fft.rfft(x, dim=-1, norm='ortho')
        # freq_domain_conv = atom_spec[None, :, :, :] * signal_spec[:, None, :, :atom_spec.shape[-1]]
        # full_size = (batch_size, self.atoms_to_apply, 1, signal_spec.shape[-1])
        # spec = torch.complex(torch.zeros(full_size), torch.zeros(full_size))
        # spec[:, :, :, :atom_spec.shape[-1]] = freq_domain_conv
        # feature_map = torch.fft.irfft(spec, dim=-1, norm='ortho').view(batch_size, self.atoms_to_apply, x.shape[-1])


        # TODO: keep the atoms in the frequency domain and do
        # the convolution there
        full = feature_map = x = F.conv1d(
            x, atoms, bias=None, stride=1, padding=self.kernel_size // 2)
        
        # minimize sum of columns of full feature map 
        # (encourage a single atom at a time)
        
        shape = feature_map.shape
        feature_map = feature_map.reshape(batch_size, -1)

        # TODO: Neural network analyzes audio and produces a
        # feature map from which we choose the top k, instead
        # of direct, linear correlation

        values, indices = torch.topk(
            torch.abs(feature_map), k=self.atoms_to_keep, dim=-1)
        
        values = feature_map.gather(-1, indices)

        new_feature_map = torch.zeros_like(feature_map)
        new_feature_map.scatter_(-1, indices, values)

        feature_map = new_feature_map.reshape(shape)

        x = F.conv_transpose1d(feature_map, atoms, bias=None,
                               stride=1, padding=self.kernel_size // 2)
        return feature_map, x, full


class MultiBandSparseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # [32768, 16384, 8192, 4096, 2048, 1024]

        n_atoms = 512

        self.bands = nn.ModuleDict({
            str(bs): AtomCollection(
                n_atoms=n_atoms,
                kernel_size=256,
                atoms_to_keep=atoms_counts[bs]) for bs in band_sizes
        })

        self.verb = NeuralReverb(n_samples, n_rooms)
        self.to_rooms = LinearOutputStack(128, 3, out_channels=n_rooms, in_channels=n_atoms * len(band_sizes))
        self.to_mix = LinearOutputStack(128, 3, out_channels=1, in_channels=n_atoms * len(band_sizes))

    def orthogonal_loss(self):
        loss = 0
        for band in self.bands.values():
            loss = loss + band.orthogonal_loss()
        return loss

    def forward(self, x):

        bands = fft_frequency_decompose(x, band_sizes[-1])
        n_samples = x.shape[-1]

        feature_maps = {}
        recon = {}
        full_maps = []
        sparse = []

        # TODO: Use global avg pooling on full (not sparse)
        # feature maps to choose reverb parameters
        for size, band in bands.items():
            fm, x, full = self.bands[str(size)].forward(band)
            sparse.append(full.sum(dim=1))
            full_maps.append(full.mean(dim=-1))
            recon[size] = x
            feature_maps[size] = fm

        full_features = torch.cat(full_maps, dim=-1)

        rooms = torch.softmax(self.to_rooms(full_features), dim=-1)
        mix = torch.sigmoid(self.to_mix(full_features)).view(-1, 1, 1)

        signal = fft_frequency_recompose(recon, n_samples)

        wet = self.verb.forward(signal, rooms)
        signal = (signal * mix) + (wet * (1 - mix))

        return feature_maps, signal, sparse


model = MultiBandSparseModel().to(device)
optim = optimizer(model, lr=1e-3)


def train_model(batch):
    optim.zero_grad()

    feature_maps, signal, sparse = model.forward(batch)

    # sparse_loss = sum([s.sum() for s in sparse])
    # print(sparse_loss.item())

    fake = pif.scattering_transform(signal)
    fake = torch.cat(list(fake.values()))

    real = pif.scattering_transform(batch)
    real = torch.cat(list(real.values()))

    ortho_loss = model.orthogonal_loss()
    recon_loss = F.mse_loss(fake, real)

    loss = recon_loss + ortho_loss
    loss.backward()

    optim.step()

    return loss, signal, feature_maps


@readme
class MultiBandMatchingPursuitExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.signal = None
        self.feature_maps = None

    def listen(self):
        return playable(self.signal, samplerate)

    def feature_map(self, n):
        return self.feature_maps[n].data.cpu().numpy()[0]

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)

            loss, self.signal, self.feature_maps = train_model(item)

            print(loss.item())
