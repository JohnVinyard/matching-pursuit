from config import config_values
import json
from experiments import FNetAutoencoder
import os
import zounds


if __name__ == '__main__':
    app = zounds.ZoundsApp(locals=locals(), globals=globals())
    app.start_in_thread(os.environ['PORT'])

    exp = FNetAutoencoder(overfit=True)

    print('\n\nTODAY\'S EXPERIMENT ==============================\n\n')
    print(exp.__doc__)
    print('config: ')
    print(json.dumps(config_values, indent=4))
    print('\n\n==================================================\n\n')

    exp.run()
