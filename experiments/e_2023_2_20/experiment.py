from typing import Callable, Type
from config.experiment import Experiment
from modules.dilated import DilatedStack
from modules.fft import fft_convolve
from modules.normalization import ExampleNorm
from modules.pos_encode import pos_encoded
from modules.sparse import sparsify
from modules.stft import stft
from perceptual.feature import NormalizedSpectrogram
from train.experiment_runner import BaseExperimentRunner
from train.optim import optimizer
from util.readmedocs import readme
import zounds
from torch import Tensor, nn
from util import device, playable
from torch.nn import functional as F
import torch
import numpy as np
from torch.jit._script import ScriptModule, script_method
import time


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class ApproximateConvolution(ScriptModule):
    def __init__(self):
        super().__init__()
    
    @script_method
    def forward(self, a, b, percent_sparse):
        n_samples = a.shape[-1]

        a = F.pad(a, (0, a.shape[-1]))
        b = F.pad(b, (0, b.shape[-1]))

        n_coeffs = ((a.shape[-1] // 2) + 1)
        n_elements = int(n_coeffs * percent_sparse)

        a_spec = torch.fft.rfft(a, dim=-1, norm='ortho')[..., :n_elements]
        b_spec = torch.fft.rfft(b, dim=-1, norm='ortho')[..., :n_elements]

        x = a_spec * b_spec

        x = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], n_coeffs - n_elements, dtype=x.dtype, device=x.device)], dim=-1)
        
        x = torch.fft.irfft(x, dim=-1, norm='ortho')[..., :n_samples]

        return x



approx = ApproximateConvolution()

# def approx_conv(a, b, percent_sparse):

#     n_samples = a.shape[-1]


#     a = F.pad(a, (0, a.shape[-1]))
#     b = F.pad(b, (0, b.shape[-1]))

#     n_coeffs = ((a.shape[-1] // 2) + 1)
#     n_elements = int(n_coeffs * percent_sparse)
#     # indices = torch.randperm(n_coeffs, device=a.device)[:n_elements]
#     indices = torch.arange(0, n_elements)

#     a_spec = torch.fft.rfft(a, dim=-1, norm='ortho')
#     b_spec = torch.fft.rfft(b, dim=-1, norm='ortho')

#     x = a_spec * b_spec

#     x = torch.zeros_like(a_spec)
#     for index in indices:
#         x[:, :, index] = a_spec[:, :, index] * b_spec[:, :, index]
    
#     x = torch.fft.irfft(x, dim=-1, norm='ortho')[..., :n_samples]

#     return x

def train(batch):
    batch_size = batch.shape[0]

    weights = torch.zeros(1, 256, batch.shape[-1], device=batch.device)

    # shuffled = batch[batch_size // 2:]
    # batch = batch[:batch_size // 2]
    # indices = torch.randperm(batch.shape[0], device=batch.device)
    # shuffled = batch[indices]

    start = time.monotonic()
    real_fm = fft_convolve(batch, weights)
    best = torch.argmax(real_fm, dim=-1)
    print(best.data.cpu().numpy().squeeze()) 
    print(f'Full convolution took {time.monotonic() - start}')

    start = time.monotonic()
    approx_fm = approx.forward(
        batch, weights, torch.zeros(1, device=batch.device).fill_(0.1))
    approx_best = torch.argmax(approx_fm, dim=-1)
    print(approx_best.data.cpu().numpy().squeeze())
    print(f'Approx. convolution took {time.monotonic() - start}')

    
    input('Waiting...')

    return torch.zeros(1).fill_(0), None


@readme
class ApproximateConvolution(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)
        self.real = None
        self.fake = None
    