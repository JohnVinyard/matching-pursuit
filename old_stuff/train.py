from collections import defaultdict
from learn_atom_embeddings import learn_atom_embeddings
from modules import pos_encode_feature
from modules4 import Discriminator, Generator, AutoEncoder
from set_alignment import aligned, reorder
from sparse2 import freq_recompose
from multilevel_sparse import multilevel_sparse_decode
from get_encoded import iter_training_examples, learn_dict
import zounds
import numpy as np
import torch
from torch.optim import Adam
from itertools import cycle
from enum import Enum
from matplotlib import pyplot as plt
from torch.nn.utils.clip_grad import clip_grad_value_
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.nn import functional as F
from os.path import exists

sr = zounds.SR22050()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


sparse_dict = learn_dict()

torch.backends.cudnn.benchmark = True


def unit_norm(x):
    if isinstance(x, np.ndarray):
        n = np.linalg.norm(x, axis=-1, keepdims=True)
    else:
        n = torch.norm(x, dim=-1, keepdim=True)
    return x / (n + 1e-12)


embedding_size = 32


max_atoms = 128
embedding_dim = 8
n_atoms = 512 * 6  # 512 atoms * 6 bands
mag_embedding_dim = 1
pos_embedding_dim = 1
total_vector_dim = embedding_dim + mag_embedding_dim + pos_embedding_dim
batch_size = 32
overfit = False

# OPTIONS
dense_judgements = True
gen_uses_disc_embeddings = False
one_hot = False

noise_level = 0.05

signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]


ae = AutoEncoder(128, max_atoms).to(device)
ae_optim = Adam(ae.parameters(), lr=1e-3, betas=(0, 0.9))

try:
    ae.load_state_dict(torch.load('ae.dat'))
    print('Loaded model from saved')
except IOError:
    print('Could not load model')
    pass


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


def nn_encode(encoded, max_atoms=max_atoms, embed_atoms=True):
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

    if embed_atoms:
        atoms = torch.from_numpy(coeffs[atoms]).to(device).float()
    else:
        atoms = torch.from_numpy(atoms).long().to(device)

    positions = (torch.from_numpy(positions).float().to(device))
    mags = torch.from_numpy(mags).float().to(device)

    return atoms, positions, mags


def _nn_decode(encoded, visualize=False, save=True, plot_mags=False):
    a, p, m = encoded

    a = a[0].data.cpu().numpy()
    p = p[0].data.cpu().numpy()
    m = m[0].data.cpu().numpy()

    dist = cdist(coeffs, a)
    a = np.argmin(dist, axis=0)

    if a.ndim == 2:
        atom_indices = np.argmax(a, axis=-1)
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




# atom_embeddings = np.zeros((3072, 3072))


def train_ae(batch):
    ae_optim.zero_grad()

    encoded, decoded = ae(batch)

    fa, fp, fm = decoded
    a, p, m = batch

    fpm = torch.cat([fp.view(batch_size, max_atoms, 1),
                    fm.view(batch_size, max_atoms, 1)], dim=-1)
    rpm = torch.cat([p.view(batch_size, max_atoms, 1),
                    m.view(batch_size, max_atoms, 1)], dim=-1)

    

    mse_loss = F.mse_loss(fpm, rpm)

    # ce_loss = F.cross_entropy(fa.view(-1, 3072), a.view(-1))
    ce_loss = F.mse_loss(fa, a)
    

    loss = mse_loss + ce_loss

    loss.backward()
    ae_optim.step()
    print(
        f'MSE: {mse_loss.item():.4f}, ATOM: {ce_loss.item():.4f}, TOTAL: {loss.item():.4f}')

    return batch, encoded, decoded


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
            a, p, m = nn_encode(encoded, max_atoms=max_atoms, embed_atoms=embed_atoms)
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


get_batch = BatchGenerator(overfit=overfit)


class Turn(Enum):
    GEN = 'gen'
    DISC = 'disc'


if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    if not exists('ae.dat'):
        print('Initializing embeddings')
        a, p, m = get_batch(batch_size=256, max_atoms=max_atoms, embed_atoms=False)
        atom_embeddings, recon_embeddings, coeffs = learn_atom_embeddings(
            a, m, 128)
        print(coeffs.std())

        # with torch.no_grad():
        #     ae.atom_embedding.weight[:] = torch.from_numpy(coeffs).to(device)

    while True:
        batch = get_batch(batch_size=batch_size, max_atoms=max_atoms)
        orig, decoded, recon = train_ae(batch)
        z = decoded.data.cpu().numpy().squeeze()

