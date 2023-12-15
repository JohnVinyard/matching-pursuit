from typing import Union

import torch
from torch import nn
from config.experiment import Experiment
from util.playable import playable
import numpy as np
import zounds
from conjure import LmdbCollection

from pathlib import Path
from conjure import time_series_conjure, audio_conjure, numpy_conjure, SupportedContentType
import os.path

def build_funcs(prefix):

    def build_target_value_conjure_funcs(experiment):

        @audio_conjure(
            experiment.collection,
            identifier=f'{prefix}audio',
        )
        def audio(data: torch.Tensor):
            samples = playable(data, zounds.SR22050())
            bio = samples.encode()
            return bio.read()

        @numpy_conjure(
            experiment.collection,
            content_type=SupportedContentType.Spectrogram.value,
            identifier=f'{prefix}spec',
        )
        def spec(data: torch.Tensor):
            samples = playable(data, zounds.SR22050())
            x = np.abs(zounds.spectral.stft(samples))
            return (x / (x.max() + 1e-12))

        return audio, spec

    return build_target_value_conjure_funcs


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

    real = MonitoredValueDescriptor(build_funcs('orig'))
    fake = MonitoredValueDescriptor(build_funcs('fake'))

    def __init__(
        self, 
        stream, 
        train, 
        exp: Experiment, 
        port: Union[str, int] = None, 
        save_weights: bool = False, 
        load_weights: bool = False,
        model: nn.Module = None):
        
        super().__init__()
        
        self.stream = stream
        self.exp = exp
        self.save_weights = save_weights
        self.model = model
        self.load_weights = load_weights

        self.train = train

        if port is not None:
            self.port = port
            self.collection = LmdbCollection(
                str(self.experiment_path).encode(), port=self.port)
        self.loss_func = None
        
        print(f'Has Model {model is not None}, save weights: {save_weights}, load weights: {load_weights}')
        
        if self.model is not None and self.load_weights:
            try:
                self.model.load_state_dict(torch.load(self.trained_weights_path_and_filename))
                print(f'Loaded model weights from {self.trained_weights_path_and_filename}')
            except IOError as e:
                print(f'Could not load {self.trained_weights_path_and_filename} due to {str(e)}')

    def after_training_iteration(self, l, iteration: int = None):
        self.loss_func(l)
        
        if self.save_weights and self.model is not None:
            
            if bool(iteration) and iteration % 1000 == 0:
                if not os.path.exists(self.trained_weights_path):
                    os.mkdir(self.trained_weights_path)
                torch.save(self.model.state_dict(), self.trained_weights_path_and_filename)
                print(f'Saved model weights to {self.trained_weights_path_and_filename}')

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
    
    @property
    def trained_weights_path(self):
        return Path('experiments') / Path(self.experiment_dir_name) / Path('trained_weights')
    
    @property
    def trained_weights_path_and_filename(self):
        return self.trained_weights_path / Path('weights.dat')
    
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
            self.after_training_iteration(l, i)
