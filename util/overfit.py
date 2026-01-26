from os import PathLike
from typing import Callable, List

from torch import nn
import torch

from data import get_one_audio_segment
from .device import device
from modules import max_norm
from .playable import encode_audio
from torch.optim import Adam
import conjure
from conjure import serve_conjure
from itertools import count


LossFunc = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

LoggerFactory = Callable[[conjure.LmdbCollection], List[conjure.Conjure]]

TrainingLoopHook = Callable[[int, List[conjure.Conjure], nn.Module], None]

def default_training_loop_hook(iteration: int, loggers: List[conjure.Conjure], model: nn.Module):
    pass

def add_loggers(collection: conjure.LmdbCollection) -> List[conjure.Conjure]:
    return []

def overfit_model(
        n_samples: int,
        model: nn.Module,
        loss_func: LossFunc,
        collection_name: PathLike,
        learning_rate: float = 1e-3,
        logger_factory: LoggerFactory = add_loggers,
        training_loop_hook: TrainingLoopHook = default_training_loop_hook,
        device=device):

    target = get_one_audio_segment(n_samples).to(device)

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    collection = conjure.LmdbCollection(path=collection_name)

    t, r = conjure.loggers(
        ['target', 'recon', ],
        'audio/wav',
        encode_audio,
        collection)

    other_loggers = logger_factory(collection)

    serve_conjure(
        [t, r, *other_loggers],
        port=9999,
        n_workers=1,
        web_components_version='0.0.101')

    t(max_norm(target))

    for i in count():
        optimizer.zero_grad()
        recon = model.forward()
        r(max_norm(recon))
        loss = loss_func(recon, target)
        loss.backward()
        optimizer.step()
        training_loop_hook(i, other_loggers, model)
        print(i, loss.item())
