from torch import nn
from typing import Callable
from pathlib import Path
import os
import boto3
import torch
from warnings import warn

def load_trained_weights_for_inference(
    module_init_path: str, 
    model_constructor: Callable[[], nn.Module], 
    device: str = 'cpu', 
    weights_filename = 'weights.dat',
    s3_bucket: str = None):
    
    """
    Load trained weights and return the model.  If the weights
    are available locally, use them.  If not, check S3 for weights,
    download, and load them, if available.
    """
    
    model = model_constructor()
    
    directory, filename = os.path.split(module_init_path)
    weights_path = Path(directory) / Path('trained_weights') / Path(weights_filename)
    model = model.to(device)
    
    if not os.path.exists(weights_path) and s3_bucket:
        _, experiment_date = os.path.split(directory)
        s3_key = f'{experiment_date}_weights.dat'
        s3_client = boto3.client('s3')
        
        print(f'Downloading weights to {weights_path}')
        with open(weights_path, 'wb') as f:
            s3_client.download_fileobj(s3_bucket, s3_key, f)
        print(f'Downloaded weights to {weights_path}')
    
    
    print(f'[INFO] - loaded weights from {weights_path}')
    
    # Now, we either have a local copy via previous training, via S3, or there
    # are no local weights to load
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except IOError:
        warn(f'[WARNING] - No local or remote saved weights')
    
    return model
    
    
 
    
    

