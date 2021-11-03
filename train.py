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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

# embeddings = []
# for band, d in sparse_dict.items():
#     spec = unit_norm(np.abs(np.fft.rfft(d, norm='ortho')))
#     b = torch.zeros((spec.shape[0], len(sparse_dict)))
#     b[:, band] = 1
#     full = np.concatenate([spec, b], axis=-1)
#     embeddings.append(full)

# embeddings = np.concatenate(embeddings, axis=0)
# proj = np.random.normal(0, 1, (263, 32))
# embeddings = np.dot(embeddings, proj)
# embeddings = unit_norm(np.random.normal(0, 1, (3072, 32)))


max_atoms = 128
embedding_dim = 8
n_atoms = 512 * 6  # 512 atoms * 6 bands
mag_embedding_dim = 1
pos_embedding_dim = 1
total_vector_dim = embedding_dim + mag_embedding_dim + pos_embedding_dim
batch_size = 16
overfit = False

# OPTIONS
dense_judgements = True
gen_uses_disc_embeddings = False
one_hot = False

noise_level = 0.05

signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]

disc = Discriminator(
    128,
    dense_judgements,
    embedding_size,
    one_hot,
    noise_level,
    embeddings=None).to(device)
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


def nn_encode(encoded, max_atoms=100, pack=False, return_embedding=False):
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
        emb = disc.get_embeddings([atoms, positions, mags])
        if not return_embedding:
            return emb
        else:
            at = atoms.data.cpu().numpy()
            local_embed = np.zeros((len(at), 3072))
            local_embed[np.arange(len(at)), at] = mags.data.cpu().numpy()
            local_embed = np.sum(local_embed, axis=0)

            total_embed = np.zeros((3072, 3072))
            total_embed[at] = np.repeat(local_embed[None, ...], len(at), axis=0)
            return emb, total_embed
    else:
        return atoms, positions, mags


def _nn_decode(encoded, visualize=False, save=True, plot_mags=False):

    size = embedding_size if not one_hot else 3072
    if isinstance(encoded, list):
        a, p, m = encoded
    else:
        a, p, m = \
            encoded[:, :size], \
            encoded[:, -2:-1], \
            encoded[:, -1:]

    
    atom_indices = disc.get_atom_keys(a).data.cpu().numpy()
    # translate from embeddings to time and magnitude
    pos = (disc.get_times(p).data.cpu().numpy() + 1) / 2
    mags = disc.get_mags(m).data.cpu().numpy()

    # filter positions outside (-1, 1)
    indices = np.where((pos >= 0) & (pos <= 1))
    atom_indices = atom_indices[indices]
    pos = pos[indices]
    mags = mags[indices]


    if visualize:
        t = ((pos * signal_sizes[-1])).astype(np.int32)
        sizes = list(mags * 5)

        atom_colors = colors[atom_indices]

        plt.xlim([0, signal_sizes[-1]])

        if plot_mags:
            plt.ylim([0, 20])
            plt.scatter(t, mags, alpha=0.5)
        else:
            plt.ylim([0, 3072])
            plt.scatter(t, atom_indices, sizes, c=atom_colors)
        
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

def listen():
    index = np.random.randint(0, len(recon))
    encoded = list(nn_decode(recon[index]))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def real():
    encoded = list(nn_decode(orig[0]))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded

def real_time_dist():
    return np.sort(o[..., -2])

def fake_time_dist():
    return np.sort(r[..., -2])

def real_mag_dist():
    return np.sort(o[..., -1])

def fake_mag_dist():
    return np.sort(r[..., -1])


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
        recon, diff = gen.forward(z, batch)
    
    rj = disc.forward(batch)
    fj = disc.forward(recon)
    loss = least_squares_disc_loss(rj, fj)
    loss.backward()
    # clip_grad_value_(disc.parameters(), 0.5)
    disc_optim.step()
    print('Disc: ', loss.item())
    return batch, recon


def train_gen(batch):
    gen_optim.zero_grad()
    z = latent()
    recon, diff = gen.forward(z, batch)

    commitment_cost = (diff ** 2).mean() * 0

    fj = disc.forward(recon)
    loss = least_squares_generator_loss(fj) + commitment_cost
    loss.backward()
    # clip_grad_value_(gen.parameters(), 0.5)
    gen_optim.step()
    print('Gen: ', loss.item())

atom_embeddings = np.zeros((3072, 3072))

class BatchGenerator(object):
    def __init__(self, overfit=False):
        self.overfit = overfit
        self._iter = \
            cycle([next(iter_training_examples())]) \
            if overfit else iter_training_examples()

    def __call__(self, batch_size, max_atoms, return_embeddings=False):

        if self.overfit:
            batch_size = 1

        atom_embeddings = np.zeros((3072, 3072))
        examples = []
        for example in self._iter:
            encoded = decode(example, sparse_dict)
            if return_embeddings:
                x, local_embed = nn_encode(encoded, max_atoms=max_atoms, pack=True, return_embedding=True)
                atom_embeddings += local_embed
            else:
                x = nn_encode(encoded, max_atoms=max_atoms, pack=True, return_embedding=False)
            if x.shape[0] != max_atoms:
                continue
            examples.append(x)
            if len(examples) == batch_size:
                break

        output = torch.stack(examples)
        if return_embeddings:
            return output, atom_embeddings
        else:
            return output


get_batch = BatchGenerator(overfit=overfit)


class Turn(Enum):
    GEN = 'gen'
    DISC = 'disc'


if __name__ == '__main__':
    

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    for i in range(100):
        batch, at_emb = get_batch(batch_size=batch_size, max_atoms=max_atoms, return_embeddings=True)
        atom_embeddings += at_emb
        print(i)
    
    atom_embeddings /= (atom_embeddings.std() + 1e-12)
    pca = PCA(n_components=embedding_size)
    print('Starting to learn PCA')
    pca.fit(atom_embeddings)
    coeffs = pca.transform(atom_embeddings)
    recon_embeddings = pca.inverse_transform(coeffs)
    coeffs = unit_norm(coeffs)

    disc.atom_embedding.weight[:] = torch.from_numpy(coeffs).to(device)

    print('Done learning atom embeddings')

    tsne = TSNE(n_components=3)
    colors = tsne.fit_transform(coeffs)

    colors -= colors.min(axis=0, keepdims=True)
    colors /= colors.max(axis=0, keepdims=True)

    
    turn = cycle([
        Turn.GEN, 
        Turn.DISC
    ])

    for i, t in enumerate(turn):
        batch = get_batch(batch_size=batch_size, max_atoms=max_atoms, return_embeddings=False)
        if t == Turn.GEN:
            train_gen(batch)
        elif t == Turn.DISC:
            orig, recon = train_disc(batch)
            o = orig[0].data.cpu().numpy()
            r = recon[0].data.cpu().numpy()

            
                
            
