class Experiment(object):
    def __init__(self, samplerate, n_samples):
        super().__init__()
        self.samplerate = samplerate
        self.n_samples = n_samples