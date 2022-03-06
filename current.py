from config import config_values
import json
from experiments import InstaneousFreqExperiment
import os
import zounds
import argparse
from datetime import datetime
import os


def new_experiment():
    dt = datetime.now()
    dirname = f'e_{dt.year}_{dt.month}_{dt.day}'

    path, _ = os.path.split(__file__)

    exp_path = os.path.join(path, 'experiments', dirname)

    if os.path.exists(exp_path):
        print(
            f'Experiment {exp_path} already exists.  Remove it if you want to create a new one.')
        return

    os.mkdir(exp_path)

    with open(os.path.join(exp_path, '__init__.py'), 'w'):
        pass

    with open(os.path.join(exp_path, 'experiment.py'), 'w'):
        pass

    with open(os.path.join(exp_path, 'readme.md'), 'w'):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    args = parser.parse_args()

    if args.new:
        new_experiment()
    else:
        app = zounds.ZoundsApp(locals=locals(), globals=globals())
        app.start_in_thread(os.environ['PORT'])

        exp = InstaneousFreqExperiment(
            overfit=args.overfit, batch_size=4)

        if exp.__doc__ is None or exp.__doc__.strip() == '':
            raise ValueError('Please write a little about your experiment')

        print('\n\nTODAY\'S EXPERIMENT ==============================\n\n')
        print(exp.__doc__)
        print('config: ')
        print(json.dumps(config_values, indent=4))
        print('\n\n==================================================\n\n')

        exp.run()

        input('Check it out...')
