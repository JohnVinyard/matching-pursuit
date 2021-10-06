from collections import defaultdict
from modules4 import Discriminator, Generator
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

sr = zounds.SR22050()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
max_atoms = 128
embedding_dim = 8
n_atoms = 512 * 6  # 512 atoms * 6 bands
mag_embedding_dim = 1
pos_embedding_dim = 1
total_vector_dim = embedding_dim + mag_embedding_dim + pos_embedding_dim
batch_size = 16
overfit = False

# OPTIONS
dense_judgements = False
gen_uses_disc_embeddings = False
one_hot = False
embedding_size = 17
noise_level = 0.05
evaluate_disc = False

signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]

disc = Discriminator(
    128,
    dense_judgements,
    embedding_size,
    one_hot,
    noise_level).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

gen = Generator(
    128,
    disc.atom_embedding,
    use_disc_embeddings=gen_uses_disc_embeddings,
    embedding_size=embedding_size,
    one_hot=one_hot,
    noise_level=noise_level,
    max_atoms=max_atoms).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))


class LatentGenerator(object):
    def __init__(self, overfit=False):
        self.overfit = overfit
        self._fixed = self._generate()

    def _generate(self):
        return torch.FloatTensor(
            batch_size, 128).normal_(0, 1).to(device)

    def __call__(self):
        if self.overfit:
            return self._fixed.clone()
        return self._generate()


latent = LatentGenerator(overfit=overfit)

# def latent():
#     return torch.FloatTensor(batch_size, 128).normal_(0, 1).to(device)


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


def nn_encode(encoded, max_atoms=100, pack=False):
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
    indices = np.random.permutation(atoms.shape[0])
    atoms = atoms[indices]
    positions = positions[indices]
    mags = mags[indices]

    atoms = torch.from_numpy(atoms).long().to(device)
    positions = (torch.from_numpy(positions).float().to(device))
    mags = torch.from_numpy(mags).float().to(device)

    if pack:
        return disc.get_embeddings([atoms, positions, mags])
    else:
        return atoms, positions, mags


def _nn_decode(encoded, visualize=False, save=True, plot_mags=False):

    size = embedding_size if not one_hot else 3072
    if isinstance(encoded, list):
        a, p = encoded
    else:
        a, p = \
            encoded[:, :size], \
            encoded[:, size:size * 2]

    
    atom_indices = disc.get_atom_keys(a).data.cpu().numpy()
    # translate from embeddings to time and magnitude
    pos = np.clip((disc.get_times(p).data.cpu().numpy() + 1) / 2, -1, 1)
    mags = disc.get_mags(a).data.cpu().numpy()

    # filter positions outside (-1, 1)
    indices = np.where((pos >= 0) & (pos <= 1))
    atom_indices = atom_indices[indices]
    pos = pos[indices]
    mags = mags[indices]


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
        return atom_indices, pos, mags


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
    vis = _nn_decode(recon[0], visualize=True)
    return vis


def vis_real():
    vis = _nn_decode(orig[0], visualize=True)
    return vis

def vis():
    _nn_decode(recon[0], visualize=True, save=False)
    _nn_decode(orig[0], visualize=True, save=True)

def vis_mags():
    _nn_decode(recon[0], visualize=True, save=False, plot_mags=True)
    _nn_decode(orig[0], visualize=True, save=True, plot_mags=True)

def listen():
    index = np.random.randint(0, len(recon))
    encoded = list(nn_decode(recon[index]))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def real():
    encoded = list(nn_decode(orig[0]))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded

def real_time_graph():
    g = 2 - np.abs((o[None, :, -1:] - o[:, None, -1:]).reshape((max_atoms, max_atoms)))
    indices = np.where(g > 1.8)
    sparse = np.zeros_like(g)
    sparse[indices] = g[indices]
    return sparse

def real_atom_graph():
    atoms = o[..., :-1]
    norms = np.linalg.norm(atoms, axis=-1, keepdims=True)
    normed = atoms / (norms + 1e-12)
    g = cdist(normed, normed)
    return g.max() - g

def fake_time_graph():
    g = 2 - np.abs((r[None, :, -1:] - r[:, None, -1:]).reshape((max_atoms, max_atoms)))
    indices = np.where(g > 1.8)
    sparse = np.zeros_like(g)
    sparse[indices] = g[indices]
    return sparse

def fake_atom_graph():
    atoms = r[..., :-1]
    norms = np.linalg.norm(atoms, axis=-1, keepdims=True)
    normed = atoms / (norms + 1e-12)
    g = cdist(normed, normed)
    return g.max() - g
    

# one-sided label smoothing
real_target = 0.9
fake_target = 0


def least_squares_generator_loss(j):
    return 0.5 * ((j - real_target) ** 2).mean()


def least_squares_disc_loss(r_j, f_j):
    return 0.5 * (((r_j - real_target) ** 2).mean() + ((f_j - fake_target) ** 2).mean())


def train_disc(batch):
    disc_optim.zero_grad()
    with torch.no_grad():
        z = latent()
        recon = gen.forward(z)
    rj = disc.forward(batch)
    fj = disc.forward(recon)
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    clip_grad_value_(disc.parameters(), 0.5)
    disc_optim.step()
    print('Disc: ', loss.item())
    return batch, recon


def train_gen(batch):
    gen_optim.zero_grad()
    z = latent()
    recon = gen.forward(z)
    fj = disc.forward(recon)
    loss = least_squares_generator_loss(fj)
    loss.backward()
    clip_grad_value_(gen.parameters(), 0.5)
    gen_optim.step()
    print('Gen: ', loss.item())


class BatchGenerator(object):
    def __init__(self, overfit=False):
        self.overfit = overfit
        self._iter = \
            cycle([next(iter_training_examples())]) \
            if overfit else iter_training_examples()

    def __call__(self, batch_size, max_atoms):

        if self.overfit:
            batch_size = 1

        examples = []
        for example in self._iter:
            encoded = decode(example, sparse_dict)
            x = nn_encode(encoded, max_atoms=max_atoms, pack=True)
            if x.shape[0] != max_atoms:
                continue
            examples.append(x)
            if len(examples) == batch_size:
                break

        output = torch.stack(examples)
        return output


get_batch = BatchGenerator(overfit=overfit)


class Turn(Enum):
    GEN = 'gen'
    DISC = 'disc'


if __name__ == '__main__':
    sparse_dict = learn_dict()

    torch.backends.cudnn.benchmark = True

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    if evaluate_disc:
        disc.load_state_dict(torch.load('disc.dat'))

        result = torch.nn.Parameter(torch.FloatTensor(batch_size, max_atoms, 18).uniform_(-1, 1).to(device))
        optim = Adam([result], lr=1e-3, betas=(0, 0.9))        
        while True:
            j = disc(result).mean()
            loss = least_squares_generator_loss(j)
            loss.backward()
            optim.step()
            recon = result
            print(loss.item())
    else:
        turn = cycle([Turn.GEN, Turn.DISC])

        for i, t in enumerate(turn):
            batch = get_batch(batch_size=batch_size, max_atoms=max_atoms)
            if t == Turn.GEN:
                train_gen(batch)
            elif t == Turn.DISC:
                orig, recon = train_disc(batch)
                o = orig[0].data.cpu().numpy()
                r = recon[0].data.cpu().numpy()
