from config import config_values, Config
import json
from data.audioiter import AudioIterator
import os
import zounds
import argparse
from datetime import datetime
import os
import torch
from conjure.serve import serve_conjure
from pathlib import Path

from train.experiment_runner import BaseExperimentRunner
from util.store_trained_weights_remotely import store_trained_weights_remoteley

torch.backends.cudnn.benchmark = True

def templatized_init(class_name):
    return f'''from .experiment import {class_name}'''

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
    samplerate=22050,
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass

@readme
class {class_name}(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
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
            f'Experiment {exp_path} already exists.  Appending to dirname.')
        exp_path += 'b'    

    os.mkdir(exp_path)

    with open(os.path.join(exp_path, '__init__.py'), 'w') as f:
        if class_name:
            f.write(templatized_init(class_name))

    with open(os.path.join(exp_path, 'experiment.py'), 'w') as f:
        if class_name:
            f.write(templatized_experiment(class_name))

    with open(os.path.join(exp_path, 'readme.md'), 'w'):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--push-trained-weights', action='store_true')
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--nsamples', type=int, default=14)
    parser.add_argument('--classname', type=str, default=None)
    parser.add_argument('--postfix', type=str, default='')
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--pattern', type=str, default='*.wav')
    parser.add_argument('--save-weights', action='store_true')
    parser.add_argument('--load-weights', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--indices', action='store_true')
    

    args = parser.parse_args()

    if args.new:
        new_experiment(args.classname, args.postfix)
    elif args.push_trained_weights:
        from experiments import Current
        
        port = os.environ['PORT']

        print(args)
        
        path = Current.__module__
        
        
        stream = AudioIterator(
            args.batch_size,
            2**args.nsamples,
            zounds.SR22050(),
            args.normalize,
            args.overfit,
            step_size=args.step,
            pattern=args.pattern,
            return_indices=args.indices)
        
        exp: BaseExperimentRunner = Current(
            stream, 
            port=port, 
            save_weights=args.save_weights, 
            load_weights=args.load_weights)

        model = exp.model
        if model is None:
            raise ValueError(f'Experiment {Current.__class__.__name__} does not have an associated model')
        
        
        store_trained_weights_remoteley(
            path, model, device='cpu', s3_bucket=Config.s3_bucket())
    else:
        from experiments import Current
        port = os.environ['PORT']
        
        path = Current.__module__
        exp, date, _ = path.split('.')
        base_path = os.path.join(exp, date)
        
        # TODO: This should probably be a method on the base 
        # experiment class
        if args.clean:
            try:
                data_path = Path(base_path) / Path('experiment_data/data.mdb')
                os.remove(data_path)
                print('Removed old experiment data')
            except IOError:
                print('No old experiment data to remove')

        print(args)
        
        stream = AudioIterator(
            args.batch_size,
            2**args.nsamples,
            zounds.SR22050(),
            args.normalize,
            args.overfit,
            step_size=args.step,
            pattern=args.pattern)

        exp: BaseExperimentRunner = Current(
            stream, 
            port=port, 
            save_weights=args.save_weights, 
            load_weights=args.load_weights)

        funcs = exp.conjure_funcs

        serve_conjure(
            funcs,
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
