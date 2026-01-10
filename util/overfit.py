from os import PathLike
from typing import Callable

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


def overfit_model(
        n_samples: int,
        model: nn.Module,
        loss_func: LossFunc,
        collection_name: PathLike,
        learning_rate: float = 1e-3,
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

    serve_conjure(
        [t, r],
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
        print(i, loss.item())
