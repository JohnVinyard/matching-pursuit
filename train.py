from collections import defaultdict
from modules2 import AutoEncoder
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


sr = zounds.SR22050()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = AutoEncoder(128).to(device)
signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]


optim = Adam(network.parameters(), lr=1e-3)


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

    # sort by magnitude descending
    # encoded = sorted(encoded, key=lambda item: item[-1], reverse=True)
    shuffle(encoded)

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
                mags.append(mag / digitizers[signal_size].max)

    atoms = np.array(atoms)
    positions = np.array(positions)
    mags = np.array(mags)

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
        a, p, m = encoded[:, :8], encoded[:, 8:9], encoded[:, 9:]

    keys = sorted(digitizers.keys())

    atom_indices = network.get_atom_keys(a).data.cpu().numpy()
    pos = np.clip(p.data.cpu().numpy().squeeze(), 0, 1)
    # mags = network.get_magnitude_keys(m).data.cpu().numpy()
    mags = np.clip(m.data.cpu().numpy().squeeze(), 0, 1)

    band_indices = atom_indices // 512
    atom_indices = atom_indices % 512

    band_keys = np.array([keys[i] for i in band_indices])

    sample_pos = (pos * band_keys).astype(np.int32)

    cmags = []
    for m, k in zip(mags, band_keys):
        d = digitizers[k]
        # indices = d.backward(m)
        cmags.append(d.edges[int(m * 256)])

    for b, a, m, p in zip(band_indices, atom_indices, cmags, sample_pos):
        yield (keys[b], a, p, m)


def listen():
    encoded = list(nn_decode(recon))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def real():
    encoded = list(nn_decode(orig))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


def break_apart(x):
    indices = network.get_atom_keys(x[:, :8])
    buckets = indices // 512

    d = {}
    for i, ss in enumerate(signal_sizes):
        idx = buckets == i
        d[ss] = x[idx]
    return d

def multiband_loss(a, b):
    ad = break_apart(a)
    bd = break_apart(b)

    l = 0
    for x, y in zip(ad.values(), bd.values()):
        l = l + loss_func(x, y)
    
    
    return l


def loss_func(a, b):
    """
    Align points/atoms with their best matches from the
    decoded signal and compute overall distance
    """

    l = max(a.shape[0], b.shape[0])
    a_diff = l - a.shape[0]
    b_diff = l - b.shape[0]


    a = F.pad(a, (0, 0, 0, a_diff))
    b = F.pad(b, (0, 0, 0, b_diff))

    bands = network.get_atom_keys(b[:, :8]) // 512
    stds = np.array([digitizers[signal_sizes[b]].std for b in bands])
    stds = torch.from_numpy(stds)[..., None].to(a.device)

    # align by atom embedding
    dist = torch.cdist(a, b)
    indices = torch.argmin(dist, dim=0)
    

    # l = (((a[indices] - b) ** 2) * stds).mean()
    # return l

    return F.mse_loss(a[indices], b)


# def weight_loss():
#     w = network.encoder.atom_embedding.weight
#     n = torch.norm(w, dim=1)
#     l = ((n - 1) ** 2).mean()
#     return l

if __name__ == '__main__':
    sparse_dict = learn_dict()

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    overfit = cycle([next(iter_training_examples())])

    # TODO: Pre-embed atoms
    for i, example in enumerate(overfit):
        optim.zero_grad()

        encoded = decode(example, sparse_dict)

        a, p, m = nn_encode(encoded, digitizers)

        # print('N ATOMS', a.shape, len(set(a.data.cpu().numpy())))

        if a.shape[0] == 0:
            print('WARNING 0 length')
            continue

        recon, latent = network([a, p, m])
        orig = network.get_embeddings([a, p, m])


        o = orig.data.cpu().numpy()[:128]
        r = recon.data.cpu().numpy()[:128]

        z = latent.data.cpu().numpy()

        loss = loss_func(recon, orig)
        # loss = multiband_loss(recon, orig)

        loss.backward()

        # clip_grad_value_(network.parameters(), 0.5)

        optim.step()

        print(loss.item())
