import torch
from torch import nn

_NORM_LAYERS = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    nn.LayerNorm, nn.GroupNorm,
)


def make_initializer(init_value):
    def init_weights(p):
        # norm layers' .weight/.bias are the affine scale/shift (default
        # 1/0), not projection weights -- reinitializing them to a small
        # uniform value (mean ~0 for the scale) silently crushes everything
        # that passes through, as soon as the layer's norm is actually
        # applied (e.g. toggling a do_norm flag on)
        if isinstance(p, _NORM_LAYERS):
            return

        with torch.no_grad():
            try:
                p.weight.uniform_(-init_value, init_value)
            except AttributeError:
                pass

            try:
                p.bias.fill_(0)
            except AttributeError:
                pass
    return init_weights
