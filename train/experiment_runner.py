from typing import Union
from config.experiment import Experiment
from util import playable
import numpy as np
import zounds
from conjure import LmdbCollection

from pathlib import Path


class BaseExperimentRunner(object):
    def __init__(self, stream, train, exp: Experiment, port: Union[str, int] = None):
        super().__init__()
        self.stream = stream
        self.exp = exp
        self.real = None
        self.fake = None
        self.train = train

        if port is not None:
            self.port = port
            self.collection = LmdbCollection(
                str(self.experiment_path).encode(), port=self.port)

    @property
    def experiment_dir_name(self):
        return self.__module__.split('.')[1]

    @property
    def experiment_path(self):
        return Path('experiments') / Path(self.experiment_dir_name) / Path('experiment_data')

    @property
    def conjure_funcs(self):
        return []

    def orig(self) -> zounds.AudioSamples:
        return playable(self.real, self.exp.samplerate)

    def real_spec(self) -> np.ndarray:
        return np.abs(zounds.spectral.stft(self.orig()))

    def listen(self) -> zounds.AudioSamples:
        return playable(self.fake, self.exp.samplerate)

    def fake_spec(self) -> np.ndarray:
        return np.abs(zounds.spectral.stft(self.listen()))

    def iter_items(self):
        for i, item in enumerate(self.stream):
            item = item.view(-1, 1, self.exp.n_samples)
            yield item

    def after_training_iteration(self, l):
        pass

    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            l, r = self.train(item, i)
            self.fake = r

            print(i, l.item())

            self.after_training_iteration(l)
