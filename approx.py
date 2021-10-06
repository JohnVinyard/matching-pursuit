from datastore import batch_stream
from dict_learning_step import unit_norm
from get_encoded import iter_training_examples, learn_dict
from sparse2 import freq_decompose
import zounds
import numpy as np

n_samples = 2 ** 15
test_band = 0
batch_size = 1
window_size = 512
sparsity = 0.8

# TODO: do this in the frequency domain!!!
def windowed(samples):
    _, sw = zounds.nputil.windowed(samples.squeeze(), 512, 1, dopad=False)
    return sw

def best_exact(samples, d):
    sw = windowed(samples)
    results = np.dot(sw, d.T)
    indices = np.argmax(results)
    indices = np.unravel_index(indices, results.shape)
    return indices


def best_approx(samples, d):
    sw = windowed(samples)
    n_samps = int(sparsity * window_size)
    index = np.sort(np.random.permutation(512)[:n_samps])
    sw = sw[:, index]
    d = d[:, index]
    results = np.dot(sw, d.T)
    indices = np.argmax(results)
    indices = np.unravel_index(indices, results.shape)
    return indices


if __name__ == '__main__':
    sparse_dict = learn_dict()

    atoms = unit_norm(sparse_dict[test_band])

    print('DICT', sparse_dict[0].shape)

    path = '/home/john/workspace/audio-data/musicnet/train_data'


    while True:
        results = []

        for batch in batch_stream(path, '*.wav', batch_size, n_samples):
            bands = freq_decompose(batch, 5)
            band_size = list(bands.keys())[0]
            band = bands[band_size]
            exact = best_exact(band, atoms)
            approx = best_approx(band, atoms)
            results.append(exact == approx)

            if len(results) == 100:
                break
        
        matching = np.where(np.array(results) == True)[0]

        print(f'{matching.shape[0] / len(results)}')
        input('Next round...')
