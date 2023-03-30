
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass

@readme
class BandFilteredImpulseResponse(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
    