from collections import defaultdict
from modules import AutoEncoder, loss_func
from sparse2 import freq_recompose
from multilevel_sparse import multilevel_sparse_decode
from get_encoded import iter_training_examples, learn_dict
import zounds
import numpy as np
import torch
from torch.optim import Adam
from itertools import cycle

sr = zounds.SR22050()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

network = AutoEncoder().to(device)
optim = Adam(network.parameters(), lr=1e-4)

signal_sizes = [1024, 2048, 4096, 8192, 16384, 32768]


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
                mags.append(mag)

    atoms = np.array(atoms)
    positions = np.array(positions)
    mags = np.array(mags)

    return atoms, positions, mags


def nn_decode(encoded):
    """
    Transform the neural network encoding into one 
    """
    a, p, m = encoded


    keys = sorted(digitizers.keys())

    atom_indices = network.encoder.get_atom_keys(a).data.cpu().numpy()
    # pos = network.encoder.get_positions(p).data.cpu().numpy()
    # print(pos)
    # mags = network.encoder.get_magnitude_keys(m).data.cpu().numpy()
    m = m.data.cpu().numpy().squeeze()
    pos = p.data.cpu().numpy().squeeze()

    band_indices = atom_indices // 512
    atom_indices = atom_indices % 512

    band_keys = np.array([keys[i] for i in band_indices])

    sample_pos = (pos * band_keys).astype(np.int32)

    # cmags = []
    # for m, k in zip(mags, band_keys):
    #     d = digitizers[k]
    #     indices = d.backward(m)
    #     cmags.append(indices)

    for b, a, m, p in zip(band_indices, atom_indices, m, sample_pos):
        yield (keys[b], a, p, m)


def listen():
    encoded = list(nn_decode([pa, pp, pm]))
    decoded = decode(encoded, sparse_dict, return_audio=True)
    return decoded


if __name__ == '__main__':
    sparse_dict = learn_dict()

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    for i, example in enumerate(iter_training_examples()):
        optim.zero_grad()

        encoded = decode(example, sparse_dict)

        a, p, m = nn_encode(encoded, digitizers)

        a = torch.from_numpy(a).to(device).long()
        p = torch.from_numpy(p).to(device).float()
        m = torch.from_numpy(m).to(device).float()

        pa, pp, pm, z = network([a, p, m])

        # o = torch.cat([a, p, m], dim=-1).data.cpu().numpy()

        latent = z.data.cpu().numpy()

        orig = network.get_embeddings([a, p, m])
        recon = torch.cat([pa, pp, pm], dim=-1)

        r = recon.data.cpu().numpy()
        o = orig.data.cpu().numpy()

        loss = loss_func(
            recon, torch.clone(orig).detach())

        loss.backward()
        optim.step()
        print(loss.item())
