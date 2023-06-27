from typing import Union
from config.experiment import Experiment
from util import playable
import numpy as np
import zounds
from conjure import LmdbCollection

from pathlib import Path
from conjure import time_series_conjure, audio_conjure, numpy_conjure, SupportedContentType


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
        self.loss_func = None

        self._orig_audio = None
        self._orig_spec = None
        self._fake_audio = None
        self._fake_spec = None

    def after_training_iteration(self, l):
        self.loss_func(l)
        self._orig_audio(self.orig())
        self._orig_spec(self.real_spec())
        self._fake_audio(self.listen())
        self._fake_spec(self.fake_spec())
    
    @property
    def batch_size(self):
        return self.stream.batch_size
    
    @property
    def overfit(self):
        return self.stream.overfit

    @property
    def conjure_funcs(self):

        d = {
            'values': np.zeros((1,))
        }

        # TODO: Move into base class
        @time_series_conjure(self.collection, 'loss')
        def loss_func(l):
            """
            Append values to a time series
            """
            d['values'] = np.concatenate(
                [d['values'], l.data.cpu().numpy().reshape((1,))], axis=-1)
            return d['values'].reshape((1, -1))

        @audio_conjure(self.collection)
        def orig_audio(x: zounds.AudioSamples):
            bio = x.encode()
            return bio.read()

        @numpy_conjure(self.collection, content_type=SupportedContentType.Spectrogram.value)
        def orig_spec(x: np.ndarray):
            return (x / (x.max() + 1e-12))

        @audio_conjure(self.collection)
        def fake_audio(x: zounds.AudioSamples):
            bio = x.encode()
            return bio.read()

        @numpy_conjure(self.collection, content_type=SupportedContentType.Spectrogram.value)
        def fake_spec(x: np.ndarray):
            return (x / (x.max() + 1e-12))

        self.loss_func = loss_func
        self._orig_audio = orig_audio
        self._orig_spec = orig_spec
        self._fake_audio = fake_audio
        self._fake_spec = fake_spec

        return [
            loss_func,
            orig_audio,
            orig_spec,
            fake_audio,
            fake_spec
        ]

    @property
    def experiment_dir_name(self):
        return self.__module__.split('.')[1]

    @property
    def experiment_path(self):
        return Path('experiments') / Path(self.experiment_dir_name) / Path('experiment_data')

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

    def run(self):
        for i, item in enumerate(self.iter_items()):
            self.real = item

            l, r = self.train(item, i)
            self.fake = r
            print(i, l.item())
            self.after_training_iteration(l)


