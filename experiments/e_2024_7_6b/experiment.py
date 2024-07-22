
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.pos_encode import pos_encoded
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme

n_samples = 2**15

exp = Experiment(
    samplerate=22050,
    n_samples=n_samples,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)

class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.l1 = nn.Linear(in_channels, out_channels)
        self.l2 = nn.Linear(in_channels, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x



class NERF(nn.Module):
    def __init__(self, channels, layers):
        super().__init__()
        self.channels = channels
        self.layers = layers

def train(batch, i):
    batch, indices = batch
    batch = batch.view(-1, 1, exp.n_samples)

@readme
class FullSongNERF(BaseExperimentRunner):
    def __init__(self, stream, port=None, save_weights=False, load_weights=False):
        super().__init__(stream, train, exp, port=port, save_weights=save_weights, load_weights=load_weights)
    
    
    # def run(self):
    #     for i, item in enumerate(self.iter_items()):
    #         batch, indices = item
    #         batch = batch.view(-1, 1, exp.n_samples)
    #         loss, recon = 