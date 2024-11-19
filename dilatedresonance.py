from typing import List
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F

from util import playable
from util.playable import listen_to_sound

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib import pyplot as plt


def conv(signal: torch.Tensor, mix: torch.Tensor, filters: List[torch.Tensor]):
    _, _, n_samples = signal.shape
    output = [signal]
    current = signal
    for i, f in enumerate(filters):
        dilation = 2 ** i
        current = F.pad(current, (dilation, 0))
        current = F.conv1d(current, weight=f.view(1, 1, 2), stride=1, dilation=dilation)[..., :n_samples]
        output.append(current)

    mix = torch.softmax(mix, dim=-1)
    output = torch.stack(output, dim=-1)
    output = output * mix[None, None, None, :]
    output = torch.sum(output, dim=-1)
    return output


if __name__ == '__main__':
    n_samples = 2 ** 16

    impulse_size = 256
    n_conv_layers = 16

    signal = torch.zeros(1, 1, n_samples)
    signal[:, :, :impulse_size] = torch.zeros(impulse_size).uniform_(-1, 1) * torch.hann_window(impulse_size)

    layers = [torch.zeros(1, 1, 2).uniform_(-1, 1) for _ in range(n_conv_layers)]

    mix = torch.zeros(n_conv_layers + 1).uniform_(-1, 1)
    result = conv(signal, mix, layers)
    result = result / torch.abs(result).max()

    plt.plot(result.view(-1).data.cpu().numpy()[:])
    plt.show()

    samples = playable(result, samplerate=22050, normalize=True)
    listen_to_sound(samples)
