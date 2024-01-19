import torch
from torch import nn
from torch.nn import functional as F
from modules.decompose import fft_frequency_decompose, fft_frequency_recompose
from modules.linear import LinearOutputStack
from modules.pif import AuditoryImage
from modules.psychoacoustic import PsychoacousticFeature
from modules.reverb import NeuralReverb
from train.optim import optimizer
from util import device, playable
from util.readmedocs import readme
import zounds

from util.weight_init import make_initializer

samplerate = zounds.SR22050()

n_samples = 2 ** 15

# [32768, 16384, 8192, 4096, 2048, 1024]
band_sizes = [2 ** (15 - i) for i in range(6)]
base_keep = 8
atoms_counts = {
    1024: base_keep * 4,
    2048: base_keep * 4,
    4096: base_keep * 8,
    8192: base_keep * 8,
    16384: base_keep * 16,
    32768: base_keep * 16
}

kernel_sizes = {
    1024: 512,
    2048: 512,
    4096: 512,
    8192: 512,
    16384: 512,
    32768: 512
}

band_1 = zounds.FrequencyBand(1, samplerate.nyquist)
band_2 = zounds.FrequencyBand(samplerate.nyquist / 2, samplerate.nyquist)

scale_1 = zounds.LinearScale(band_1, 128)
scale_2 = zounds.LinearScale(band_2, 128)

fb_1 = zounds.learn.FilterBank(
    samplerate, 128, scale_1, 0.1, normalize_filters=True, a_weighting=False).to(device)
fb_2 = zounds.learn.FilterBank(
    samplerate, 128, scale_2, 0.1, normalize_filters=True, a_weighting=False).to(device)

# aim = AuditoryImage(256, 128, do_windowing=False, check_cola=False)

n_rooms = 8

# pif = PsychoacousticFeature([128] * 6)

init_weights = make_initializer(0.1)

class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.dilated = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv = nn.Conv1d(channels, channels, 1, 1, 0)
    
    def forward(self, x):
        orig = x
        x = self.dilated(x)
        x = self.conv(x)
        x = x + orig
        x = F.leaky_relu(x, 0.2)
        return x


class AtomCollection(nn.Module):
    def __init__(
            self,
            n_atoms,
            kernel_size,
            atoms_to_keep,
            band_size,
            atoms_to_apply=None):

        super().__init__()
        self.band_size = band_size

        model_dim = 32
        self.net = nn.Sequential(
            nn.Conv1d(128, model_dim, 1, 1, 0),
            DilatedBlock(model_dim, 1),
            DilatedBlock(model_dim, 3),
            nn.Conv1d(model_dim, n_atoms, 1, 1, 0),
        )


        self.n_atoms = n_atoms
        self.kernel_size = kernel_size
        self.atoms_to_keep = atoms_to_keep
        self.atoms_to_apply = atoms_to_apply or n_atoms

        self.atoms = nn.Parameter(
            torch.zeros(n_atoms, 1, self.kernel_size).uniform_(-1, 1))
        
        self.apply(init_weights)
    
    def clip_atom_norms(self):
        self.atoms.data[:] = self.unit_norm_atoms()

    def unit_norm_atoms(self):
        norms = torch.norm(self.atoms, dim=-1, keepdim=True)
        atoms = self.atoms / (norms + 1e-12)
        return atoms

    def orthogonal_loss(self):
        x = self.unit_norm_atoms().view(self.n_atoms, self.kernel_size)
        sim = x @ x.T
        return sim.mean()
    
    def clip_atom_norms(self):
        pass

    def forward(self, x):

        batch_size = x.shape[0]

        # give atoms unit norm
        atoms = self.unit_norm_atoms()


        if self.band_size == band_sizes[0]:
            filter = fb_1
        else:
            filter = fb_2

        x = filter.forward(x, normalize=False)
        
        feature_map = x

        full = feature_map = self.net(feature_map)
        c = self.choice(feature_map)
        v = self.value(feature_map)

        shape = feature_map.shape
        feature_map = feature_map.reshape(batch_size, -1)

        c = c.reshape(batch_size, -1)
        v = v.reshape(batch_size, -1)

        c = torch.softmax(c, dim=-1)


        # TODO: Would a mask be better to allow gradients
        # to continue to flow?
        values, indices = torch.topk(
            c, k=self.atoms_to_keep, dim=-1)

        logits = c.gather(-1, indices)
        logits = logits + (1 - logits)
        values = v.gather(-1, indices)

        values = values * logits

        new_feature_map = torch.zeros_like(feature_map)
        new_feature_map.scatter_(-1, indices, values)

        feature_map = new_feature_map.reshape(shape)

        x = F.conv_transpose1d(
            feature_map, atoms, bias=None, stride=1, padding=self.kernel_size // 2)
        

        return feature_map, x, full


class MultiBandSparseModel(nn.Module):
    def __init__(self):
        super().__init__()
        # [32768, 16384, 8192, 4096, 2048, 1024]

        n_atoms = 256

        self.bands = nn.ModuleDict({
            str(bs): AtomCollection(
                n_atoms=n_atoms,
                kernel_size=kernel_sizes[bs],
                atoms_to_keep=atoms_counts[bs],
                band_size=bs) for bs in band_sizes
        })

        # self.verb = NeuralReverb(n_samples, n_rooms)
        # self.to_rooms = LinearOutputStack(
        #     128, 2, out_channels=n_rooms, in_channels=n_atoms * len(band_sizes))
        # self.to_mix = LinearOutputStack(
        #     128, 2, out_channels=1, in_channels=n_atoms * len(band_sizes))

    def clip_atom_norms(self):
        for size, band in self.bands.items():
            band.clip_atom_norms()
        
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

        # full_features = torch.cat(full_maps, dim=-1)

        # rooms = torch.softmax(self.to_rooms(full_features), dim=-1)
        # mix = 0.9 + (torch.sigmoid(self.to_mix(full_features)).view(-1, 1, 1) * 0.1)

        signal = fft_frequency_recompose(recon, n_samples)

        # wet = self.verb.forward(signal, rooms)
        # signal = (signal * mix) + (wet * (1 - mix))

        return feature_maps, signal, sparse


model = MultiBandSparseModel().to(device)
optim = optimizer(model, lr=1e-3)


def perceptual_feature(x):
    final_bands = {}
    bands = fft_frequency_decompose(x, band_sizes[-1])

    for k, v in bands.items():
        filt = fb_1 if k == band_sizes[0] else fb_2
        z = filt.forward(v, normalize=False)
        z = z.unfold(-1, 128, 64)
        z = torch.abs(torch.fft.rfft(z, dim=-1, norm='ortho'))
        final_bands[k] = z
    return final_bands



def perceptual_loss(fake, real):
    fake_bands = perceptual_feature(fake)
    real_bands = perceptual_feature(real)
    loss = 0
    for k, v in fake_bands.items():
        loss = loss + F.mse_loss(v, real_bands[k])
    return loss


def train_model(batch):
    optim.zero_grad()
    feature_maps, signal, sparse = model.forward(batch)

    #ortho_loss = model.orthogonal_loss() * 10


    loss = perceptual_loss(signal, batch) #+ ortho_loss
    loss.backward()
    optim.step()
    with torch.no_grad():
        model.clip_atom_norms()
    return loss, signal, feature_maps


@readme
class MultiBandMatchingPursuitExperiment(object):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream

        self.signal = None
        self.feature_maps = None
        self.r = None
        self.model = model

    def listen(self):
        return playable(self.signal, samplerate)
    
    def real(self):
        return playable(self.r, samplerate)

    def feature_map(self, n):
        return self.feature_maps[n].data.cpu().numpy()[0]

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, n_samples)
            self.r = item

            loss, self.signal, self.feature_maps = train_model(item)

            if i % 10 == 0:
                print(loss.item())
