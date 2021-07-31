from train import decode, nn_encode, digitizers
from get_encoded import iter_training_examples, learn_dict
import torch
import bisect
import numpy as np
from torch import nn
from torch.optim import Adam
import zounds

segment_duration = 0.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embedding = nn.Embedding(3072, 8).to(device)
optim = Adam(embedding.parameters(), lr=1e-4)
loss = nn.MarginRankingLoss().to(device)


def iter_sorted_by_time():
    sparse_dict = learn_dict()
    for i, example in enumerate(iter_training_examples()):
        encoded = decode(example, sparse_dict)
        a, p, m = nn_encode(encoded, digitizers)

        # indices sorted by time
        indices = torch.argsort(p)

        # TODO: function to return all atoms within a window
        a = a[indices].data.cpu().numpy()
        p = p[indices].data.cpu().numpy()
        m = m[indices].data.cpu().numpy()

        yield a, p, m


def iter_examples():
    b = []
    for a, p, m in iter_sorted_by_time():
        start = np.random.random()
        stop = start + segment_duration

        start_index = bisect.bisect_left(p, start)
        stop_index = bisect.bisect_right(p, stop)
        atoms = a[start_index: stop_index]

        if len(atoms) == 0:
            continue

        if len(b) == 0:
            b.extend(np.random.choice(atoms, 2))
        elif len(b) == 2:
            b.append(np.random.choice(atoms))
        else:
            yield b
            b = []

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

if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(9999)

    for anchor, positive, negative in iter_batches(batch_size=32):
        a = embedding.forward(anchor)
        p = embedding.forward(positive)
        n = embedding.forward(negative)

        pos_dist = ((a - p) ** 2).mean()
        neg_dist = ((n - p) ** 2).mean()

        l = loss.forward(a, n, p)
        l.backward()
        optim.step()
        print(l.item())
