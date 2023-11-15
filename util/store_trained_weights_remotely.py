from torch import nn
import os
from pathlib import Path
import boto3
from warnings import warn


def store_trained_weights_remoteley(
    module_init_path: str,
    model: nn.Module, 
    device: str = 'cpu',
    s3_bucket: str = None):
    
    exp, date, _ = module_init_path.split('.')
    base_path = os.path.join(exp, date)
    
    weights_path = Path(base_path) / Path('trained_weights/weights.dat')
    
    experiment_date = date
    s3_key = f'{experiment_date}_weights.dat'
    s3_client = boto3.client('s3')
    
    try:
        with open(weights_path, 'rb') as f:
            data = f.read()
            print(f'Pushing {s3_bucket}/{s3_key}')
            resp = s3_client.put_object(
                Bucket=s3_bucket, 
                Key=s3_key, 
                ACL='public-read', 
                Body=data)
            print('Done pushing')
            print(resp)
    except IOError:
        warn('No local or remote saved weights')