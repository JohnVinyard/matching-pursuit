from typing import Union

import torch
from config.experiment import Experiment
from util import playable
import numpy as np
import zounds
from conjure import LmdbCollection

from pathlib import Path
from conjure import time_series_conjure, audio_conjure, numpy_conjure, SupportedContentType


# TODO: the following two functions should be collapsed into one
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


def build_recon_value_conjure_funcs(experiment):

    @audio_conjure(experiment.collection)
    def fake_audio(data: torch.Tensor):
        samples = playable(data, experiment.exp.samplerate)
        bio = samples.encode()
        return bio.read()
    

    @numpy_conjure(experiment.collection, content_type=SupportedContentType.Spectrogram.value)
    def fake_spec(data: torch.Tensor):
        samples = playable(data, experiment.exp.samplerate)
        x = np.abs(zounds.spectral.stft(samples))
        return (x / (x.max() + 1e-12))
    
    return fake_audio, fake_spec



class MonitoredValueDescriptor(object):

    def __init__(self, build_conjure_funcs):
        super().__init__()
        self.build_conjure_funcs = build_conjure_funcs

    def get_conjure_funcs(self, experiment):
        return self.build_conjure_funcs(experiment)

    def __set_name__(self, owner, name):
        self.name = name


    def __set__(self, obj, value):
        funcs = self.build_conjure_funcs(obj)

        # run all the conjure funcs
        for func in funcs:
            func(value)



class BaseExperimentRunner(object):

    real = MonitoredValueDescriptor(build_target_value_conjure_funcs)
    fake = MonitoredValueDescriptor(build_recon_value_conjure_funcs)

    def __init__(self, stream, train, exp: Experiment, port: Union[str, int] = None):
        super().__init__()
        self.stream = stream
        self.exp = exp

        self.train = train

        if port is not None:
            self.port = port
            self.collection = LmdbCollection(
                str(self.experiment_path).encode(), port=self.port)
        self.loss_func = None
        

    def after_training_iteration(self, l):
        self.loss_func(l)

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

        
        self.loss_func = loss_func
        

        funcs = []

        for key in dir(self.__class__):
            item = getattr(self.__class__, key)
            if isinstance(item, MonitoredValueDescriptor):
                others = item.get_conjure_funcs(self)
                funcs.extend(others)

        funcs.extend([loss_func])
        

        return funcs

    @property
    def experiment_dir_name(self):
        return self.__module__.split('.')[1]

    @property
    def experiment_path(self):
        return Path('experiments') / Path(self.experiment_dir_name) / Path('experiment_data')

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
