
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.phase import AudioCodec, MelScale
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme


scale = MelScale()
codec = AudioCodec(scale)

def spectrogram(audio: torch.Tensor):
    spec = codec.to_frequency_domain(audio)
    return spec

exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


def train(batch, i):
    pass

@readme
class ConjureDemo(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
    