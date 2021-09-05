from collections import defaultdict
from modules4 import Discriminator, Generator, unit_norm
from sparse2 import freq_recompose
from multilevel_sparse import multilevel_sparse_decode
from get_encoded import iter_training_examples, learn_dict
import zounds
import numpy as np
import torch
from torch.optim import Adam
from itertools import cycle
from random import shuffle
from torch.nn import functional as F
from torch.nn import Embedding
from torch.nn.utils.clip_grad import clip_grad_value_


sr = zounds.SR22050()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
one_hot = False
max_atoms = 100
overfit = False


def get_trained_weights():
    with open('embedding.dat', 'rb') as f:
        embedding = Embedding(3072, 8).to(device)
        embedding.load_state_dict(torch.load(f))
        return embedding.weight.data.cpu().numpy()


# network = AutoEncoder(128, get_trained_weights()).to(device)


signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]


# optim = Adam(network.parameters(), lr=1e-4, betas=(0, 0.9))

embedding_weights = get_trained_weights()

gen = Generator(128, embedding_weights, one_hot).to(device)
gen_optim = Adam(gen.parameters(), lr=1e-4, betas=(0, 0.9))

disc = Discriminator(128, embedding_weights, one_hot).to(device)
disc_optim = Adam(disc.parameters(), lr=1e-4, betas=(0, 0.9))

if overfit:
    z_latent = torch.FloatTensor(1, 128).normal_(0, 1).to(device)

def latent():
    if overfit:
        return z_latent
    
    return torch.FloatTensor(1, 128).normal_(0, 1).to(device)



class Digitizer(object):
    def __init__(self, data, n_bins):
        super().__init__()
        self.data = np.sort(data)
        self.n_bins = n_bins
        self.edges = np.histogram_bin_edges(data, bins=n_bins)

    @property
    def std(self):
        return self.data.std()

    @property
    def mean(self):
        return self.data.mean()

    @property
    def max(self):
        return self.data.max()

    def __repr__(self):
        return f'''
Digitizer(
    n_bins={self.n_bins}, 
    range=[{self.data.min()}...{self.data.max()}, 
    std={self.std}, 
    mean={self.mean}])'''

    def __str__(self):
        return self.__repr__()

    def forward(self, x):
        return np.clip(np.digitize(x, self.edges), 0, self.n_bins - 1)

    def backward(self, x):
        return self.data[np.clip(np.searchsorted(self.data, x), 0, self.n_bins - 1)]


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


def build_digitizers(bins=256, n_examples=20):
    magintudes = defaultdict(list)

    for i, example in enumerate(iter_training_examples()):
        for instance in example:
            sig_size, _, _, mag = instance
            magintudes[sig_size].append(mag)
        if i >= n_examples:
            break

    # sig_sizes = sorted(magintudes.keys())
    sig_sizes = signal_sizes
    sig_size_indices = {ss: i for i, ss in enumerate(sig_sizes)}

    digitizers = {}

    for size, index in zip(sig_sizes, sig_size_indices):
        digitizers[index] = Digitizer(magintudes[size], bins)

    return digitizers


digitizers = build_digitizers()
for k, v in digitizers.items():
    print(k, v.mean, v.std)


def nn_encode(encoded, digitizers):
    """
    Transform the encoding into a format 
    suitable for the neural network to manipulate
    """
    atoms = []
    positions = []
    mags = []

    sig_sizes = sorted(digitizers.keys())
    sig_size_indices = {i: ss for i, ss in enumerate(sig_sizes)}

    for band_index, atom_dict in encoded.items():
        signal_size = sig_size_indices[band_index]

        for atom_index, atom_list in atom_dict.items():
            for atom, pos, mag, _ in atom_list:
                atoms.append(512 * band_index + atom)
                positions.append(pos / float(signal_size))
                # mags.extend(digitizers[signal_size].forward([mag]))
                # mags.append(mag / digitizers[signal_size].max)
                mags.append(mag / 20)

    atoms = np.array(atoms)
    positions = np.array(positions)
    mags = np.array(mags)

    # sort by time
    indices = np.argsort(mags)[::-1][:max_atoms]
    # indices = np.random.permutation(atoms.shape[0])
    atoms = atoms[indices]
    positions = positions[indices]
    mags = mags[indices]

    atoms = torch.from_numpy(atoms).long().to(device)
    positions = torch.from_numpy(positions).float().to(device)
    mags = torch.from_numpy(mags).float().to(device)

    return atoms, positions, mags


def nn_decode(encoded):
    """
    Transform the neural network encoding into one 
    """

    # encoded = network.flatten(encoded)

    if isinstance(encoded, list):
        a, p, m = encoded
    else:
        a, p, m = encoded[:, :8 if not one_hot else 3072], encoded[:, -2:-1], encoded[:, -1:]

    keys = sorted(digitizers.keys())

    atom_indices = disc.get_atom_keys(a).data.cpu().numpy()
    pos = np.clip(p.data.cpu().numpy().squeeze(), 0, 1)
    # mags = network.get_magnitude_keys(m).data.cpu().numpy()
    mags = np.clip(m.data.cpu().numpy().squeeze(), 0, 1)
    cmags = mags * 20

    band_indices = atom_indices // 512
    atom_indices = atom_indices % 512

    band_keys = np.array([keys[i] for i in band_indices])

    sample_pos = (pos * band_keys).astype(np.int32)

    # cmags = []
    # for m, k in zip(mags, band_keys):
    #     d = digitizers[k]
    #     # indices = d.backward(m)
    #     cmags.append(d.edges[int(m * 256)])

    for b, a, m, p in zip(band_indices, atom_indices, cmags, sample_pos):
        yield (keys[b], a, p, m)


def listen():
    print(recon.shape)
    encoded = list(nn_decode(recon))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def real():
    encoded = list(nn_decode(orig))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


real_target = 1
fake_target = 0


def least_squares_generator_loss(j):
    return 0.5 * ((j - real_target) ** 2).mean()


def least_squares_disc_loss(r_j, f_j):
    return 0.5 * (((r_j - real_target) ** 2).mean() + ((f_j - fake_target) ** 2).mean())



def train_disc(example1, example2):
    disc_optim.zero_grad()

    # train disc
    encoded = decode(example1, sparse_dict)
    a, p, m = nn_encode(encoded, digitizers)
    rl = torch.FloatTensor([a.shape[0] / max_atoms]).to(device)

    with torch.no_grad():
        z = latent()
        recon, l = gen.forward(z)

    rj1 = disc.forward([a, p, m], rl).view(-1)
    fj1 = disc.forward(recon, l).view(-1)

    # do it again
    encoded = decode(example2, sparse_dict)
    a, p, m = nn_encode(encoded, digitizers)
    rl = torch.FloatTensor([a.shape[0] / max_atoms]).to(device)

    with torch.no_grad():
        z = latent()
        recon, l = gen.forward(z)

    rj2 = disc.forward([a, p, m], rl).view(-1)
    fj2 = disc.forward(recon, l).view(-1)

    loss = least_squares_disc_loss(
        torch.cat([rj1, rj2]), torch.cat([fj1, fj2]))

    loss.backward()
    # clip_grad_value_(disc.parameters(), 0.5)
    disc_optim.step()
    print('Disc: ', loss.item(), a.shape[0], recon.shape[0])

    orig = disc.get_embeddings([a, p, m])

    return recon, orig


def train_gen(example):
    gen_optim.zero_grad()

    # train gen
    encoded = decode(example, sparse_dict)
    a, p, m = nn_encode(encoded, digitizers)

    # do two
    z = latent()
    recon, l = gen.forward(z)
    fj1 = disc.forward(recon, l)

    z = latent()
    recon2, l2 = gen.forward(z)
    fj2 = disc.forward(recon2, l2)

    loss = least_squares_generator_loss(torch.cat([fj1, fj2]))  # + diff

    loss.backward()

    # print('============================================')
    # for n, p in gen.named_parameters():
    #     if p.grad is None:
    #         continue
    #     print(n, p.grad.std().item())

    # clip_grad_value_(gen.parameters(), 0.5)
    gen_optim.step()
    print('Gen: ', loss.item(), a.shape[0], recon.shape[0])


if __name__ == '__main__':
    sparse_dict = learn_dict()

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    stream = enumerate(iter_training_examples())
    if overfit:
        stream = cycle([next(stream)])

    while True:
        _, example1 = next(stream)
        _, example2 = next(stream)
        recon, orig = train_disc(example1, example2)

        _, example3 = next(stream)
        train_gen(example3)
        o = orig.data.cpu().numpy()[:128]
        r = recon.data.cpu().numpy()[:128]
