from typing import Tuple
import torch
from torch import nn

class DiffusionProcess(object):
    def __init__(self, total_steps, variance_per_step):
        super().__init__()
        self.total_steps = total_steps
        self.variance_per_step = variance_per_step
    
    def forward_process(self, x: torch.Tensor, steps: int) -> Tuple[torch.Tensor, float, torch.Tensor]:
        orig_steps = steps
        steps = 1 + int(steps[0, 0] * self.total_steps)
        for i in range(steps):
            noise = torch.normal(0, self.variance_per_step, x.shape).to(x.device)
            x = x + noise
        
        # current_step = i / self.total_steps
        return x, orig_steps, noise
    

    def backward_process(self, x: torch.Tensor, current_step: float, model: nn.Module):
        x = model(x, current_step)
        return x
