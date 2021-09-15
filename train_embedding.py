from modules4 import unit_norm
from re import M
from train import decode, nn_encode
from get_encoded import iter_training_examples, learn_dict
import torch
import bisect
import numpy as np
from torch import nn
from torch.optim import Adam
import zounds
from sklearn.manifold import TSNE
import argparse
from matplotlib import pyplot as plt

# TODO: What about training this using the method with no negative examples?
# Instead, we ensure that:
# - each feature in the batch has zero mean and unit variance (std=1)
# - Upper diagonal of the batch's covariance matrix is zero (features are independent)

segment_duration = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding = nn.Embedding(3072, 8).to(device)

with torch.no_grad():
    embedding.weight.normal_(0, 1)

optim = Adam(embedding.parameters(), lr=1e-2)
loss = nn.TripletMarginLoss(margin=0.01).to(device)


def get_trained_weights():
    with open('embedding.dat', 'rb') as f:
        embedding.load_state_dict(torch.load(f))
        return embedding.weight.data


def iter_sorted_by_time():
    sparse_dict = learn_dict()
    for example in iter_training_examples():
        encoded = decode(example, sparse_dict)
        a, p, m = nn_encode(encoded)

        # indices sorted by time
        indices = torch.argsort(p)

        # TODO: function to return all atoms within a window
        a = a[indices].data.cpu().numpy()
        p = p[indices].data.cpu().numpy()
        m = m[indices].data.cpu().numpy()

        yield a, p, m


def iter_examples():
    b = []

    prev = None

    for a, p, m in iter_sorted_by_time():
        start = np.random.random()
        stop = start + segment_duration

        start_index = bisect.bisect_left(p, start)
        stop_index = bisect.bisect_right(p, stop)
        atoms = a[start_index: stop_index]

        if len(atoms) == 0:
            continue

        if len(b) == 0:
            # anchor and positive example
            b.extend(np.random.choice(atoms, 2))
        elif len(b) == 2:
            # negative example
            b.append(np.random.choice(atoms))
        else:
            yield b
            b = []
            b.extend(np.random.choice(atoms, 2))


def iter_batches(batch_size=128):
    anchor = []
    positive = []
    negative = []

    for a, p, n in iter_examples():
        anchor.append(a)
        positive.append(p)
        negative.append(n)

        if (len(anchor) == batch_size):
            anc = torch.from_numpy(np.array(anchor)).long().to(device)
            pos = torch.from_numpy(np.array(positive)).long().to(device)
            neg = torch.from_numpy(np.array(negative)).long().to(device)
            yield anc, pos, neg

            anchor = []
            positive = []
            negative = []


def visualize():
    w = unit_norm(get_trained_weights().cpu().numpy())
    tsne = TSNE()
    points = tsne.fit_transform(w)
    plt.scatter(points[:, 0], points[:, 1], s=5, alpha=0.1)
    plt.savefig('embeddings.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    if args.train:
        app = zounds.ZoundsApp(locals=locals(), globals=globals())
        app.start_in_thread(9999)

        for anchor, positive, negative in iter_batches(batch_size=128):
            optim.zero_grad()

            a = embedding.forward(anchor)
            p = embedding.forward(positive)
            n = embedding.forward(negative)

            l = loss.forward(a, p, n)
            l.backward()
            optim.step()
            print(l.item())

    elif args.visualize:
        visualize()

    else:
        pass
