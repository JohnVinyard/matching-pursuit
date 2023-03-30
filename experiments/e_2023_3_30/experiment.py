
import torch
from torch import nn
from torch.nn import functional as F
import zounds
from config.experiment import Experiment
from scalar_scheduling import fft_shift
from time_distance import optimizer
from train.experiment_runner import BaseExperimentRunner
from util import device
from util.readmedocs import readme
from scipy.signal import gausspulse


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
        self.position = nn.Parameter(torch.zeros(1)._fill(initial_position_value))
    
    def forward(self):
        return fft_shift(
            self.signal, 
            torch.clamp(self.position, 0, 1)
        )

def generate_signal(size, padded_size=None, device=None):
    p = gausspulse(size)
    p = torch.from_numpy(p, device=device)

    if padded_size:
        p = torch.cat([
            p, 
            torch.zeros(padded_size - p.shape[-1], device=p.device)
        ])
    
    return p


def experiment(
        total_size, 
        signal_size, 
        target_position, 
        starting_position, 
        iterations, 
        loss,
        device=None):
    
    print('---------------------------------')
    print(f'Experiment using {loss}, target pos {target_position:.2f} starting pos {starting_position:.2f}')
    
    pulse = generate_signal(signal_size, device=device)
    model = Model(pulse, )

    target = generate_signal(signal_size, total_size, device=device)
    target = fft_shift(target, target_position)

    model = Model(pulse, initial_position_value=starting_position).to(device)
    optim = optimizer(model, lr=1e-3)

    for i in range(iterations):
            
        estimate = model.forward()
        l = loss(estimate, target)
        l.backward()
        optim.step()

        if i % 10 == 0:
            print(f'For target {target_position:.2f}, starting_position {starting_position:.2f}, loss is {l.item()}')
    
    with torch.no_grad():
        print(f'final model position is {model.position.item():.2f}')
        final_estimate = model.forward()
        overlay = target + final_estimate
        return overlay


def train(batch, i):
    pass

@readme
class ScalarPositioning(BaseExperimentRunner):
    def __init__(self, stream):
        super().__init__(stream, train, exp)

        self.mse_far_overlay = None
        self.mse_close_overlay = None
        self.sampling_kernel_overlay = None
    
    def run(self):

        total_size = 8192
        signal_size = 64
        overlap_range = signal_size / total_size
        device = 'cpu'

        # hypothesis - total failure
        self.mse_far_overlay = experiment(
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.9,
            iterations=1000,
            loss=F.mse_loss,
            device=device
        )

        # hypothesis - perfect success
        self.mse_close_overlay = experiment(
            total_size=total_size,
            signal_size=signal_size,
            target_position=0.5,
            starting_position=0.5 + (overlap_range * 0.5),
            iterations=1000,
            loss=F.mse_loss,
            device=device
        )
    