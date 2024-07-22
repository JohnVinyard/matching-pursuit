
from config.dotenv import Config
from data.datastore import batch_stream
from util import device
import torch
import numpy as np


def audio_stream(
        batch_size,
        n_samples,
        overfit=False,
        normalize=False,
        as_torch=True,
        step_size=1,
        pattern='*.wav',
        return_indices=False):

    stream = batch_stream(
        Config.audio_path(),
        pattern,
        batch_size,
        n_samples,
        overfit,
        normalize,
        step_size=step_size,
        return_indices=return_indices)

    for item in stream:
        if return_indices:
            item, indices = item
            
            if not as_torch:
                yield indices, item.astype(np.float32)
            else:
                yield indices, torch.from_numpy(item).float().to(device)
        else:
            if not as_torch:
                yield item.astype(np.float32)
            else:
                yield torch.from_numpy(item).float().to(device)
