import zounds
import numpy as np
import torch
from modules.waveguide import waveguide_synth
from torch.nn import functional as F

n_samples = 2 ** 15
samplerate = zounds.SR22050()


class DelayLine(object):

    def __init__(self, n_samples, decay=0.9):
        super().__init__()
        self.n_samples = n_samples
        self.decay = decay
        self.buffer = []

    def append(self, x):
        self.buffer.append(x)

    def forward(self, x):
        output = 0
        if len(self.buffer) > self.n_samples:
            output = self.buffer[0] * self.decay
            self.buffer = self.buffer[1:]
        # self.buffer.append(x)
        return output





if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)

    delay = torch.from_numpy(np.array([128, 130, 132, 130])).float().view(1, 1, -1)
    damping = torch.from_numpy(np.array([0.99, 0.99, 0.99, 0.99])).float().view(1, 1, -1)
    filter_size = torch.from_numpy(np.array([2, 10, 10, 3])).float().view(1, 1, -1)

    delay = F.interpolate(delay, size=n_samples, mode='linear').squeeze().long().data.cpu().numpy()
    damping = F.interpolate(damping, size=n_samples, mode='linear').squeeze().data.cpu().numpy()
    filter_size = F.interpolate(filter_size, size=n_samples, mode='linear').squeeze().long().data.cpu().numpy()

    impulse = np.zeros(n_samples)
    impulse[:32] = np.random.uniform(-1, 1, 32)

    samples = waveguide_synth(impulse, delay, damping, filter_size)
    samples = zounds.AudioSamples(samples, samplerate).pad_with_silence()

    input('Whatever...')
