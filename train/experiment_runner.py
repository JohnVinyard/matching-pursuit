from config.experiment import Experiment
from util import playable
import numpy as np
import zounds

class BaseExperimentRunner(object):
    def __init__(self, stream, train, exp: Experiment):
        super().__init__()
        self.stream = stream
        self.exp = exp
        self.real = None
        self.fake = None
        self.train = train

    def orig(self):
        return playable(self.real, self.exp.samplerate)
    
    def real_spec(self):
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self):
        return playable(self.fake, self.exp.samplerate)
    
    def fake_spec(self):
        return np.abs(zounds.spectral.stft(self.listen()))

    def hard_pos(self):
        return self.pos.data.cpu().numpy().reshape((-1, self.exp.n_frames))

    def run(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, self.exp.n_samples)
            self.real = item

            l, r = self.train(item)
            self.fake = r

            print(l.item())
