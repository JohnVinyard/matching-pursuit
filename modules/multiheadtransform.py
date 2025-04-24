from torch import nn
from typing import Dict, Tuple, Union
import torch
import numpy as np

from modules import LinearOutputStack
from modules.eventgenerators.generator import ShapeSpec


class MultiHeadTransform(nn.Module):

    def __init__(
            self,
            latent_dim: int,
            hidden_channels: int,
            shapes: ShapeSpec,
            n_layers: int,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.shapes = shapes
        self.n_layers = n_layers
        self.hidden_channels = hidden_channels
        self.shapes = shapes

        modules = {
            # name: custom_modules.get(name, None) or nn.Linear(latent_dim, np.prod(shapes[name]))
            name: LinearOutputStack(
                channels=self.hidden_channels,
                layers=n_layers,
                in_channels=latent_dim,
                out_channels=np.prod(shapes[name]),
                norm=lambda channels: nn.LayerNorm([channels,])
            )
            for name, shape in shapes.items()
        }

        self.mods = nn.ModuleDict(modules)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch, n_events, latent = x.shape

        return {
            name: module.forward(x).view(batch, n_events, *self.shapes[name])
            for name, module
            in self.mods.items()
        }
