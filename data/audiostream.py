
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
        pattern='*.wav'):

    stream = batch_stream(
        Config.audio_path(),
        pattern,
        batch_size,
        n_samples,
        overfit,
        normalize,
        step_size=step_size)

    for item in stream:
        if not as_torch:
            yield item.astype(np.float32)
        else:
            yield torch.from_numpy(item).float().to(device)
