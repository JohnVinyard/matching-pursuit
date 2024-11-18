from torch import nn
from typing import Dict, Tuple
import torch
import numpy as np
from modules.eventgenerators.generator import ShapeSpec


class MultiHeadTransform(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            hidden_channels: int,
            shapes: ShapeSpec,
            n_layers: int):
        super().__init__()

        self.latent_dim = latent_dim
        self.shapes = shapes
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.shapes = shapes

        modules = {
            name: nn.Linear(latent_dim, np.prod(shapes[name]))
            for name, shape in shapes.items()}

        self.mods = nn.ModuleDict(modules)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, n_events, latent = x.shape

        return {
            name: module.forward(x).view(batch, n_events, *self.shapes[name])
            for name, module
            in self.mods.items()
        }


