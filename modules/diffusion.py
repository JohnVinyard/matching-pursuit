from typing import Tuple
import torch
from torch import nn


class DiffusionProcess(object):
    def __init__(self, total_steps, variance_per_step):
        super().__init__()
        self.total_steps = total_steps
        self.variance_per_step = variance_per_step

    def forward_process(self, x: torch.Tensor, steps: int) -> Tuple[torch.Tensor, float, torch.Tensor]:
        """
        Add noise to a signal N times in small steps
        """
        orig_steps = steps
        steps = 1 + int(steps[0, 0] * self.total_steps)
        for i in range(steps):
            noise = torch.normal(0, self.variance_per_step, x.shape).to(x.device)
            x = x + noise

        # current_step = i / self.total_steps
        return x, orig_steps, noise

    def backward_process(self, x: torch.Tensor, current_step: float, model: nn.Module, *args, **kwargs):
        """
        Allow the model to perform one denoisng step, given the noisy signal and
        potentially some conditioning information
        """
        x = model(x, current_step, *args, **kwargs)
        return x
    

    def get_steps(self, batch: int, device, steps=None):
        if steps is None:
            steps = torch.rand(1)
        
        steps = torch.zeros((batch, 1)).fill_(float(steps)).to(device)
        return steps
    
    def generate(self, shape: Tuple[int], model: nn.Module, device, *args, **kwargs):
        with torch.no_grad():   
            start = torch.normal(0, 1.8, shape).to(device)

            for i in range(self.total_steps):
                curr = 1 - torch.zeros((shape[0], 1)).fill_(i / self.total_steps).to(device)
                noise_pred = self.backward_process(start, curr, model, *args, **kwargs)
                start = start - noise_pred
            
            return start
    
