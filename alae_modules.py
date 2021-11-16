import torch
from torch import nn
import zounds
from get_encoded import iter_training_examples, learn_dict
from learn_atom_embeddings import learn_atom_embeddings
from modules import PositionalEncoding
from modules2 import Expander

from modules3 import LinearOutputStack
from modules4 import AttentionStack, MyBatchNorm, unit_norm
from itertools import chain, cycle
from torch.optim import Adam
from torch.nn import functional as F
from multilevel_sparse import multilevel_sparse_decode
from sparse2 import freq_recompose

from train import BatchGenerator, decode, nn_decode
from enum import Enum
from os.path import exists
from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

def init_weights(p):
    try:
        if not p.requires_grad:
            print('NO INIT FOR', p)
            return
    except AttributeError:
        pass

    with torch.no_grad():
        try:
            p.weight.uniform_(-0.11, 0.11)
        except AttributeError:
            pass

        try:
            p.bias.uniform_(-0.0001, 0.0001)
        except AttributeError:
            pass

'''
Training steps:
----------------------

1) discriminator - ensure that E and D together correctly predict real vs. fake
2) generator     - ensure that F and G together fool the discriminator
3) latent        - z -> f() -> [g() -> e()], loss equals the MSE between latent and output of E
'''
sr = zounds.SR22050()
sparse_dict = learn_dict()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 32
max_atoms = 128
signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]


def decode(encoded, sparse_dict, return_audio=False):
    def a(): return defaultdict(list)
    d = defaultdict(a)

    # signal_sizes = sorted(list(set(e[0] for e in encoded)))
    sig_size_indices = {ss: i for i, ss in enumerate(signal_sizes)}

    batch_size = 1
    batch_num = 0

    for sig_size, atom, pos, mag in encoded:
        key = sig_size_indices[sig_size]
        d[key][atom].append((atom, pos, mag, batch_num))

    if not return_audio:
        return d

    bands = multilevel_sparse_decode(
        batch_size, signal_sizes, sparse_dict, d)

    recomposed = freq_recompose(bands)[0]
    return zounds.AudioSamples(recomposed, sr).pad_with_silence()


def nn_encode(encoded, max_atoms=max_atoms):
    """
    Transform the encoding into a format 
    suitable for the neural network to manipulate
    """
    atoms = []
    positions = []
    mags = []

    sig_sizes = signal_sizes
    sig_size_indices = {i: ss for i, ss in enumerate(sig_sizes)}

    for band_index, atom_dict in encoded.items():
        signal_size = sig_size_indices[band_index]

        for atom_index, atom_list in atom_dict.items():
            for atom, pos, mag, _ in atom_list:
                atoms.append(512 * band_index + atom)
                positions.append((pos / float(signal_size)) * 2 - 1)
                mags.append(np.clip(mag, 0, 20))

    atoms = np.array(atoms)
    positions = np.array(positions)
    mags = np.array(mags)

    # get the N loudest
    indices = np.argsort(mags)[::-1][:max_atoms]
    atoms = atoms[indices]
    positions = positions[indices]
    mags = mags[indices]

    # shuffle the loudest
    # indices = np.random.permutation(atoms.shape[0])
    # atoms = atoms[indices]
    # positions = positions[indices]
    # mags = mags[indices]

    # sort by time
    indices = np.argsort(positions)
    atoms = atoms[indices]
    positions = positions[indices]
    mags = mags[indices]

    atoms = torch.from_numpy(atoms).long().to(device)
    positions = (torch.from_numpy(positions).float().to(device))
    mags = torch.from_numpy(mags).float().to(device)

    return atoms, positions, mags


def _nn_decode(encoded, visualize=False, save=True, plot_mags=False):
    a, p, m = encoded

    a = a[0].data.cpu().numpy()
    p = p[0].data.cpu().numpy()
    m = m[0].data.cpu().numpy()

    if a.ndim == 2:
        dist = cdist(unit_norm(encoder.atom_embedding.weight.data.cpu().numpy()), unit_norm(a))
        atom_indices = np.argmin(dist, axis=0)
    else:
        atom_indices = a
    pos = (p + 1) / 2
    mags = m

    if visualize:
        t = ((pos * signal_sizes[-1])).astype(np.int32)
        sizes = list(mags * 5)

        plt.xlim([0, signal_sizes[-1]])

        if plot_mags:
            plt.ylim([0, 20])
            plt.scatter(t, mags, alpha=0.5)
        else:
            plt.ylim([0, 3072])
            plt.scatter(t, atom_indices, sizes, alpha=0.5)

        if save:
            plt.savefig('vis.png')
            plt.clf()
    else:
        return atom_indices.reshape(-1), pos.reshape(-1), mags.reshape(-1)


def nn_decode(encoded):
    """
    Transform the neural network encoding into one 
    """

    keys = signal_sizes

    atom_indices, pos, mags = _nn_decode(encoded)

    band_indices = atom_indices // 512
    atom_indices = atom_indices % 512

    band_keys = np.array([keys[i] for i in band_indices])
    sample_pos = (pos * band_keys).astype(np.int32)
    for b, a, m, p in zip(band_indices, atom_indices, mags, sample_pos):
        yield (keys[b], a, p, m)


def vis_fake():
    vis = _nn_decode(recon, visualize=True)
    return vis


def vis_real():
    vis = _nn_decode(orig, visualize=True)
    return vis


def vis():
    o, r = orig, recon
    _nn_decode(r, visualize=True, save=False)
    _nn_decode(o, visualize=True, save=True)


def listen():
    encoded = list(nn_decode(recon))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def real():
    encoded = list(nn_decode(orig))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


class FNet(nn.Module):
    def __init__(self, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.net = LinearOutputStack(latent_size, 5)
        self.apply(init_weights)

    def forward(self, x):
        x = self.net(x)
        return x


class Generator(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.channels = channels
        self.max_atoms = max_atoms
        self.pos = PositionalEncoding(1, max_atoms, 16)
        self.mlp = LinearOutputStack(channels, 8, in_channels=33 + channels)

        self.atom = LinearOutputStack(channels, 3, out_channels=128)
        self.time = LinearOutputStack(channels, 3, out_channels=1)
        self.mag = LinearOutputStack(channels, 3, out_channels=1)

        self.decoder = nn.Sequential(
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            Expander(channels, 2),
            MyBatchNorm(channels),
            nn.LeakyReLU(0.2),
            LinearOutputStack(channels, 2)
        )

        self.apply(init_weights)


    def forward(self, x):
        # encoded = x.view(-1, 1, self.channels).repeat(1, self.max_atoms, 1)
        # pos = self.pos.pos_encode.view(
        #     1, self.max_atoms, -1).repeat(encoded.shape[0], 1, 1)
        # x = torch.cat([encoded, pos], dim=-1)
        # x = self.mlp(x)

        x = x.view(-1, 1, 128)
        x = self.decoder(x)

        a = unit_norm(self.atom(x))
        p = torch.tanh(self.time(x))
        m = F.relu(self.mag(x)) * 20
        return a, p, m


class Encoder(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.channels = channels
        self.max_atoms = max_atoms

        self.pos = PositionalEncoding(1, max_atoms, 16)
        self.pos_embedding = LinearOutputStack(channels, 3, in_channels=33)

        self.atom_embedding = nn.Embedding(3072, channels)

        self.pos_mag = LinearOutputStack(channels, 3, in_channels=2)

        self.comb = LinearOutputStack(channels, 3, in_channels=channels * 3)
        self.attn = AttentionStack(
            channels,
            attention_heads=8,
            attention_layers=8,
            intermediate_layers=1)
        self.final = LinearOutputStack(channels, 3)

        self.encoder = nn.Sequential(
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 2, 2),
            # nn.BatchNorm1d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv1d(channels, channels, 1, 1)
        )
        self.apply(init_weights)


    def forward(self, a, p, m):

        a = a.view(batch_size, self.max_atoms, -1)
        p = p.view(-1, self.max_atoms, 1)
        m = m.view(-1, self.max_atoms, 1)


        pos = self.pos_embedding(self.pos.pos_encode)
        pos = pos.view(-1, self.max_atoms, self.channels).repeat(a.shape[0], 1, 1)

        if a.shape[-1] == 1:
            atom = self.atom_embedding(a).view(-1, self.max_atoms, self.channels)
        else:
            atom = a

        atom = unit_norm(atom)

        pm = torch.cat([p, m], dim=-1)
        pm = self.pos_mag(pm)

        x = torch.cat([pos, atom, pm], dim=-1)
        x = self.comb(x)

        # attn encoder
        # x = self.attn(x)
        # x = x.mean(dim=1, keepdim=True)
        # x = self.final(x)
        # return x

        # conv encoder
        x = x.permute(0, 2, 1)
        x = self.encoder(x)
        x = x.permute(0, 2, 1)

        return x


class Discriminator(nn.Module):
    def __init__(self, channels, max_atoms):
        super().__init__()
        self.net = LinearOutputStack(channels, 5, out_channels=1)
        self.apply(init_weights)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


fnet = FNet(128).to(device)
gen = Generator(128, 128).to(device)
encoder = Encoder(128, 128).to(device)
disc = Discriminator(128, 128).to(device)


disc_optim = Adam(chain(encoder.parameters(),
                  disc.parameters()), lr=1e-4, betas=(0, 0.9))
gen_optim = Adam(chain(fnet.parameters(), gen.parameters()),
                 lr=1e-4, betas=(0, 0.9))
# latent_optim = Adam(
#     chain(gen.parameters(), encoder.parameters()), lr=1e-4, betas=(0, 0.9))

# one-sided label smoothing
real_target = 0.9
fake_target = 0


def least_squares_generator_loss(j):
    # return 0.5 * ((j - real_target) ** 2).mean()
    return torch.abs(1 - j).mean()


def least_squares_disc_loss(r_j, f_j):
    # return 0.5 * (((r_j - real_target) ** 2).mean() + ((f_j - fake_target) ** 2).mean())
    return torch.abs(1 - r_j).mean() + torch.abs(0 - f_j).mean()


def latent():
    return torch.FloatTensor(batch_size, 1, 128).normal_(0, 1).to(device)


def train_disc(batch):
    disc_optim.zero_grad()

    with torch.no_grad():
        z = latent()
        inner_latent = fnet(z)
        fake = gen(inner_latent)

    r_encoded = encoder(*batch)
    f_encoded = encoder(*fake)

    rj = disc(r_encoded)
    fj = disc(f_encoded)

    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    disc_optim.step()
    print('Disc: ', loss.item())
    return batch, fake, r_encoded


def train_gen(batch):
    gen_optim.zero_grad()

    z = latent()
    inner_latent = fnet(z)
    fake = gen(inner_latent)

    encoded = encoder(*fake)
    fj = disc(encoded)

    loss = least_squares_generator_loss(fj)
    loss.backward()
    gen_optim.step()
    print('Gen: ', loss.item())


def train_latent(batch):
    disc_optim.zero_grad()
    gen_optim.zero_grad()

    z = latent()
    inner_latent = fnet(z)
    fake = gen(inner_latent)
    encoded = encoder(*fake)

    loss = F.mse_loss(encoded.view(batch_size, -1), inner_latent.view(batch_size, -1).clone().detach())
    loss.backward()
    disc_optim.step()
    gen_optim.step()
    print('Latent: ', loss.item())
    return z, inner_latent, encoded


class BatchGenerator(object):
    def __init__(self, overfit=False):
        self.overfit = overfit
        self._iter = \
            cycle([next(iter_training_examples())]) \
            if overfit else iter_training_examples()

    def __call__(self, batch_size, max_atoms, embed_atoms=True):

        if self.overfit:
            batch_size = 1

        atoms = []
        pos = []
        mag = []

        for example in self._iter:
            encoded = decode(example, sparse_dict)
            a, p, m = nn_encode(encoded, max_atoms=max_atoms)
            if a.shape[0] != max_atoms:
                continue

            atoms.append(a[None, ...])
            pos.append(p[None, ...])
            mag.append(m[None, ...])

            if len(atoms) == batch_size:
                break

        atoms = torch.cat(atoms, dim=0)
        pos = torch.cat(pos, dim=0)
        mag = torch.cat(mag, dim=0)
        return atoms, pos, mag


get_batch = BatchGenerator()


def listen():
    encoded = list(nn_decode(recon))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def real():
    encoded = list(nn_decode(orig))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


class Turn(Enum):
    GEN = 'gen'
    DISC = 'disc'
    LATENT = 'latent'


if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)


    steps = cycle([Turn.DISC, Turn.GEN, Turn.LATENT])

    while True:
        batch = get_batch(batch_size=batch_size, max_atoms=max_atoms)
        t = next(steps)
        if t == Turn.GEN:
            train_gen(batch)
        elif t == Turn.DISC:
            orig, recon, r_encode = train_disc(batch)
            rz = r_encode.data.cpu().numpy().squeeze()
            r = recon[0].data.cpu().numpy()[0]
        else:
            lat, f_latent, d_latent = train_latent(batch)
            l = lat.data.cpu().numpy().squeeze()
            fz = f_latent.data.cpu().numpy().squeeze()
            dz = d_latent.data.cpu().numpy().squeeze()
