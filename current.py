from config import config_values
import json
from data.audioiter import AudioIterator
from experiments import Current
import os
import zounds
import argparse
from datetime import datetime
import os
import numpy as np
import torch
from conjure.serve import serve_conjure

from train.experiment_runner import BaseExperimentRunner

torch.backends.cudnn.benchmark = True


def templatized_experiment(class_name):
    experiment_template = f'''
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass

@readme
class {class_name}(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
    '''
    return experiment_template


def new_experiment(class_name=None, postfix=''):
    dt = datetime.now()

    if postfix and not postfix.startswith('_'):
        postfix = '_' + postfix

    dirname = f'e_{dt.year}_{dt.month}_{dt.day}{postfix}'

    path, _ = os.path.split(__file__)

    exp_path = os.path.join(path, 'experiments', dirname)

    if os.path.exists(exp_path):
        print(
            f'Experiment {exp_path} already exists.  Remove it if you want to create a new one.')
        return

    os.mkdir(exp_path)

    with open(os.path.join(exp_path, '__init__.py'), 'w'):
        pass

    with open(os.path.join(exp_path, 'experiment.py'), 'w') as f:
        if class_name:
            f.write(templatized_experiment(class_name))

    with open(os.path.join(exp_path, 'readme.md'), 'w'):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--nsamples', type=int, default=14)
    parser.add_argument('--classname', type=str, default=None)
    parser.add_argument('--postfix', type=str, default='')

    args = parser.parse_args()

    if args.new:
        new_experiment(args.classname, args.postfix)
    else:
        # app = zounds.ZoundsApp(locals=locals(), globals=globals())
        # app.start_in_thread(os.environ['PORT'])

        port = os.environ['PORT']

        stream = AudioIterator(
            args.batch_size,
            2**args.nsamples,
            zounds.SR22050(),
            args.normalize,
            args.overfit)

        exp: BaseExperimentRunner = Current(stream, port=port)

        serve_conjure(
            exp.conjure_funcs,
            port=port,
            n_workers=2
        )

        if exp.__doc__ is None or exp.__doc__.strip() == '':
            raise ValueError('Please write a little about your experiment')

        print('\n\nTODAY\'S EXPERIMENT ==============================\n\n')
        print(exp.__doc__)
        print('config: ')
        print(json.dumps(config_values, indent=4))
        print('\n\n==================================================\n\n')

        exp.run()

        input('Check it out...')
