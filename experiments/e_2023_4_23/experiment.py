
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.ddsp import overlap_add
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
from experiments.e_2023_3_8.experiment import model


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**17,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)




def window(x: torch.Tensor, window_size: int, step_size: int) -> torch.Tensor:

    x = F.pad(x, (0, step_size))
    windowed = x.unfold(-1, window_size, step_size)

    lap = window_size - step_size

    fade = torch.ones_like(windowed)
    ramp = torch.linspace(0, 1, lap, device=x.device)[None, None, None, :]
    fade[..., :lap] = fade[..., :lap] * ramp
    fade[..., -lap:] = fade[..., -lap:] * ramp.flip(-1)

    windowed = windowed * fade
    return windowed


def sparse_code(
        x: torch.Tensor, 
        window_size: int, 
        step_size: int, 
        n_steps: int = 100) -> torch.Tensor:

    windowed = window(x, window_size, step_size)
    batch, _, n_frames, time = windowed.shape
    windowed = windowed.permute(0, 2, 1, 3).view(batch * n_frames, 1, time)

    e, scatter, _ = model.encode(windowed, steps=n_steps)
    decoded = scatter()


def window_and_crossfade(x: torch.Tensor, window_size: int, step_size: int) -> torch.Tensor:
    orig = x
    batch, _, time = x.shape

    windowed = window(x, window_size, step_size)    

    final = torch.zeros_like(x)

    for i in range(windowed.shape[2]):
        start = i * step_size
        end = start + window_size

        final[:, :, start:end] += windowed[:, :, i, :]
    final = final[..., :time]
    return final


def train(batch, i):
    recon = window_and_crossfade(batch, 32768, 32768 - 512)
    return torch.zeros(1).fill_(0), recon

@readme
class StitchingAlgorithm(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
    