from typing import Union

import torch
from config.experiment import Experiment
from util import playable
import numpy as np
import zounds
from conjure import LmdbCollection

from pathlib import Path
from conjure import time_series_conjure, audio_conjure, numpy_conjure, SupportedContentType


def build_target_value_conjure_funcs(experiment):

    @audio_conjure(experiment.collection)
    def orig_audio(data: torch.Tensor):
        samples = playable(data, experiment.exp.samplerate)
        bio = samples.encode()
        return bio.read()

    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def orig_spec(data: torch.Tensor):
        samples = playable(data, experiment.exp.samplerate)
        x = np.abs(zounds.spectral.stft(samples))
        return (x / (x.max() + 1e-12))

    return orig_audio, orig_spec


class MonitoredValueDescriptor(object):

    def __init__(self, build_conjure_funcs):
        super().__init__()
        self.build_conjure_funcs = build_conjure_funcs

    def get_conjure_funcs(self, experiment):
        return self.build_conjure_funcs(experiment)

    def __set_name__(self, owner, name):
        self.name = name

    # def __get__(self, obj, t=None):
    #     return self

    def __set__(self, obj, value):
        funcs = self.build_conjure_funcs(obj)

        # run all the conjure funcs
        for func in funcs:
            func(value)



class BaseExperimentRunner(object):

    real = MonitoredValueDescriptor(build_target_value_conjure_funcs)

    def __init__(self, stream, train, exp: Experiment, port: Union[str, int] = None):
        super().__init__()
        self.stream = stream
        self.exp = exp

        # self.real = None

        self.fake = None
        self.train = train

        if port is not None:
            self.port = port
            self.collection = LmdbCollection(
                str(self.experiment_path).encode(), port=self.port)
        self.loss_func = None

        # self._orig_audio = None
        # self._orig_spec = None

        self._fake_audio = None
        self._fake_spec = None

    def after_training_iteration(self, l):
        self.loss_func(l)
        # self._orig_audio(self.orig())
        # self._orig_spec(self.real_spec())
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

        @time_series_conjure(self.collection, 'loss')
        def loss_func(l):
            """
            Append values to a time series
            """
            d['values'] = np.concatenate(
                [d['values'], l.data.cpu().numpy().reshape((1,))], axis=-1)
            return d['values'].reshape((1, -1))

        # @audio_conjure(self.collection)
        # def orig_audio(x: zounds.AudioSamples):
        #     bio = x.encode()
        #     return bio.read()

        # @numpy_conjure(self.collection, content_type=SupportedContentType.Spectrogram.value,)
        # def orig_spec(x: np.ndarray):
        #     return (x / (x.max() + 1e-12))

        @audio_conjure(self.collection)
        def fake_audio(x: zounds.AudioSamples):
            bio = x.encode()
            return bio.read()

        @numpy_conjure(self.collection, content_type=SupportedContentType.Spectrogram.value)
        def fake_spec(x: np.ndarray):
            return (x / (x.max() + 1e-12))

        self.loss_func = loss_func
        # self._orig_audio = orig_audio
        # self._orig_spec = orig_spec
        self._fake_audio = fake_audio
        self._fake_spec = fake_spec

        funcs = []

        # print('===========================')
        # print(dir(self))
        # print(dir(self.__class__))

        for key in dir(self.__class__):
            item = getattr(self.__class__, key)
            if isinstance(item, MonitoredValueDescriptor):
                others = item.get_conjure_funcs(self)
                funcs.extend(others)

        funcs.extend([loss_func, fake_audio, fake_spec])

        # funcs = [
        #     loss_func,
        #     # orig_audio,
        #     # orig_spec,
        #     fake_audio,
        #     fake_spec
        # ]

        return funcs

    @property
    def experiment_dir_name(self):
        return self.__module__.split('.')[1]

    @property
    def experiment_path(self):
        return Path('experiments') / Path(self.experiment_dir_name) / Path('experiment_data')

    # def orig(self) -> zounds.AudioSamples:
    #     return playable(self.real, self.exp.samplerate)

    # def real_spec(self) -> np.ndarray:
    #     return np.abs(zounds.spectral.stft(self.orig()))

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
