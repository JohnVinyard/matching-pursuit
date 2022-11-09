import numpy as np
from matplotlib import pyplot as plt


ratings = [
  3,
  5,
  5,
  4,
  4.5,
  5,
  5,
  5,
  5,
  5,
  4,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  4,
  5,
  5,
  4,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  5,
  4,
  4,
  4,
  5,
  4,
  3,
  5,
  5,
  4,
  5,
  4,
  5,
  5,
  5,
  5,
  5,
  4,
  5,
  5,
  5,
  4,
  4,
  5,
  4,
  4,
  5,
  4,
  5,
  5,
  5,
  3,
  5,
  5,
  5,
  5,
  5,
  5,
  4,
  5,
  4,
  4,
  5,
  5,
  4,
  5,
  4,
  5,
  4,
  4,
  5,
  5,
  5,
  4,
  5,
  5,
  5,
  5,
  4,
  5,
  5,
  4,
  4,
  5,
  5
]

if __name__ == '__main__':
    # previous = np.array([3.5] * 100)

    previous = ratings
    new = np.array([5] * 14)

    all_ratings = np.concatenate([previous, new])
    
    current_snapshot = all_ratings[-100:]

    # print(np.mean(current_snapshot))

    weighting = np.linspace(0, 1, 100) ** 14
    weighting = weighting / np.sum(weighting)

    # new_avg = np.dot(current_snapshot, weighting)
    new_avg = np.sum(current_snapshot * weighting)
    print(new_avg)

    # plt.plot(weighting)
    # plt.show()

    