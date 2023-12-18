import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    signal = np.ones((128,))
    diffs = []
    for i in range(128):
        pad = np.zeros(i)
        cat = np.concatenate([pad, signal])[:128]
        diff = ((signal - cat) ** 2).mean()
        diffs.append(diff)
    
    plt.plot(diffs)
    plt.show()
        