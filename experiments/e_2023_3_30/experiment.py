
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from modules.fft import fft_convolve
from scalar_scheduling import fft_shift
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
from scipy.signal import morlet
import numpy as np


exp = Experiment(
    samplerate=zounds.SR22050(),
    n_samples=2**15,
    weight_init=0.1,
    model_dim=128,
    kernel_size=512)


class Model(nn.Module):
    def __init__(self, signal, initial_position_value):
        super().__init__()
        self.register_buffer('signal', signal)
        self.position = nn.Parameter(torch.zeros(1).fill_(initial_position_value))
        self.activation = lambda x: x
    
    @property
    def current_position_value(self):
        with torch.no_grad():
            return self.activation(self.position).item()
    
    def forward(self):
        return fft_shift(
            self.signal, 
            self.activation(self.position)
        )

def sampling_kernel(size, device=None):
    rng = torch.linspace(0, 1, size, device=device)
    kernel = rng
    return kernel

def dirac_impulse(size, position, device=None):
    x = torch.zeros(size, device=device)
    x[position] = 1
    return x

def differentiable_index_test(size, position, device=None):
    imp = dirac_impulse(size, position, device=device)
    kernel = sampling_kernel(size, device=device)
    index = (kernel @ imp)
    return index


def differentiable_index(impulse):
    kernel = sampling_kernel(impulse.shape[-1], device=impulse.device)
    index = (kernel @ impulse)
    return index


def best_match(signal, search_pattern):
    affinity_map = F.conv1d(signal.view(1, 1, -1), search_pattern.view(1, 1, -1)).view(-1)
    # given a feature map, take the softmax, then multiply by any number of channels
    # with different sampling kernels

    # in this case, your target(s) are indices or positions, OR you have
    # a way to infer these (e.g., computing the best matching positions)
    
    affinity_map = torch.softmax(affinity_map, dim=-1)
    index = torch.argmax(affinity_map, dim=-1)
    forward = dirac_impulse(
        affinity_map.shape[-1], position=index, device=affinity_map.device)
    backward = affinity_map
    output = backward + (forward - backward).detach()
    return output

def generate_signal(size, padded_size=None, device=None):
    p = morlet(size).real
    p = torch.from_numpy(p).to(device).float()

    if padded_size:
        p = torch.cat([
            p, 
            torch.zeros(padded_size - p.shape[-1], device=p.device)
        ])
    return p

def experiment(
        name,
        total_size, 
        signal_size, 
        target_position, 
        starting_position, 
        iterations, 
        loss,
        device=None):
    
    print(f'--- {name} ---------------------------------')
    print(f'Experiment using {loss}, target pos {target_position:.2f} starting pos {starting_position:.2f}')
    
    target = generate_signal(signal_size, total_size, device=device)
    target = fft_shift(target, target_position)

    model = Model(
        signal=generate_signal(signal_size, total_size, device=device), 
        initial_position_value=starting_position).to(device)
    optim = optimizer(model, lr=1e-3)

    for i in range(iterations):
        optim.zero_grad()
        estimate = model.forward()
        l = loss(estimate, target)
        l.backward()
        optim.step()

        if l == 0:
            break

        if i % 500 == 0:
            print(f'For iteration {i}, target {target_position:4f}, current position {model.current_position_value:.4f}, loss is {l.item()}')
    
    with torch.no_grad():
        print(f'final model position is {model.current_position_value:.2f} at iteration {i}')
        final_estimate = model.forward()
        overlay = target + final_estimate
        return overlay

def index_loss(a, b):
    return torch.abs(a - b).sum()

def positional_loss(a, b, pattern):
    a_impulse = best_match(a, pattern)
    b_impulse = best_match(b, pattern)

    a_index = differentiable_index(a_impulse)
    b_index = differentiable_index(b_impulse)

    loss = index_loss(a_index, b_index)

    return loss


def self_similarity_loss(a: torch.Tensor, b: torch.Tensor):

    a = a.unfold(-1, 128, 64)
    b = b.unfold(-1, 128, 64)
    
    a_sim = torch.cdist(a, b)
    b_sim = torch.cdist(b, b)

    # a_sim = fft_convolve(a, b)
    # b_sim = fft_convolve(b, b)
    # a_sim = a[None, ...] * b[..., None]
    # b_sim = b[None, ...] * b[..., None]

    loss = torch.abs(a_sim - b_sim).sum()
    return loss

def compute_positional_losses(size):
    positions = torch.arange(0, size)
    diracs = [dirac_impulse(size, p) for p in positions]

    loss = np.zeros((size, size))

    for i in range(len(diracs)):
        for j in range(0, len(diracs)):
            a = diracs[i]
            b = diracs[j]
            a_index = differentiable_index(a)
            b_index = differentiable_index(b)
            loss[i, j] = index_loss(a_index, b_index).item()
    
    return loss

def train(batch, i):
    pass

@readme
class ScalarPositioning(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)

        self.mse_far_overlay = None
        self.mse_close_overlay = None
        self.sampling_kernel_overlay = None
        self.example_signal = generate_signal(64, 8192).data.cpu().numpy()
        self.kernel = sampling_kernel(16, device=device).data.cpu().numpy()
        self.indices = [differentiable_index_test(64, i, device=device).item() for i in range(64)]
        self.losses = compute_positional_losses(64)
    
    def run(self):

        total_size = 8192
        signal_size = 64
        overlap_range = signal_size / total_size
        iterations = 10000

        search_signal = generate_signal(signal_size, device=device)

        # hypothesis - total failure
        self.mse_far_overlay = experiment(
            name='start higher, MSE loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.9,
            iterations=iterations,
            loss=lambda a, b: F.mse_loss(a, b),
            device=device
        )

        # hypothesis - total failure
        self.mse_far_overlay = experiment(
            name='start lower, MSE loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.1,
            iterations=iterations,
            loss=lambda a, b: F.mse_loss(a, b),
            device=device
        )

        # hypothesis - perfect success
        self.mse_close_overlay = experiment(
            name='start close, MSE loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.5 + (overlap_range * 0.5),
            iterations=iterations,
            loss=lambda a, b: F.mse_loss(a, b),
            device=device
        )


        # hypothesis - success
        self.mse_far_overlay = experiment(
            name='start higher, positional loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.9,
            iterations=iterations,
            loss=lambda a, b: positional_loss(a, b, search_signal),
            device=device
        )

        # hypothesis - success
        self.mse_far_overlay = experiment(
            name='start lower, positional loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.1,
            iterations=iterations,
            loss=lambda a, b: positional_loss(a, b, search_signal),
            device=device
        )

        # hypothesis - success
        self.mse_far_overlay = experiment(
            name='start higher, self-similarity loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.9,
            iterations=iterations,
            loss=lambda a, b: self_similarity_loss(a, b),
            device=device
        )

        # hypothesis - success
        self.mse_far_overlay = experiment(
            name='start lower, self-similarity loss',
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.1,
            iterations=iterations,
            loss=lambda a, b: self_similarity_loss(a, b),
            device=device
        )
    