from modules import pos_encode
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist


if __name__ == '__main__':
    n_samples = 512
    a = np.linspace(-1, 1, n_samples)[None, :]
    b = pos_encode(1, n_samples, 16)

    adist = cdist(a, a)
    bdist = cdist(b, b)

    plt.matshow(adist)
    plt.show()

    plt.matshow(bdist)
    plt.show()