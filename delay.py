import zounds
import numpy as np
from time import perf_counter

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

class Filter(object):
    def __init__(self, n_samples):
        super().__init__()
        self.buffer = []
    
    def append(self, x):
        self.buffer.append(x)
    
    def forward(self, x):
        output = 0
        if len(self.buffer) > self.n_samples:
            output = sum(self.buffer) / n_samples
        return output

if __name__ == '__main__':

    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(8888)


    start = perf_counter()

    o = []
    n = np.random.uniform(-1, 1, 32)
    delay = DelayLine(450, decay=0.95)
    filt = Filter(4)

    for i in range(n_samples):
        s = 0
        if i < len(n):
            s += n[i]
        
        d = delay.forward(s)
        d = filt.forward(d)
        s += d
        delay.append(s)
        o.append(s)
    
    
    samples = np.array(o)
    end = perf_counter()

    samples = zounds.AudioSamples(samples, samplerate).pad_with_silence()

    input(f'{end - start} seconds to generate {n_samples / int(samplerate)} seconds of audio')

