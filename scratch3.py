import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm
import torch.nn.functional as F
import numpy as np


def sparsify2(x: torch.Tensor, n_to_keep: int = 8):
    """
    input: (batch, channels, time)

    outputs:
        sparse:  (batch, channels, time)
        packed:  (batch, n_to_keep, time)
        one_hot: (batch, n_to_keep, channels)
    
        
    - one_hot can be used to generate events
    - packed can be used to convolve generated events with 
      original activations
    """
    batch, channels, time = x.shape
    # orig = x

    x = x.reshape(batch, -1)
    values, indices = torch.topk(x, k=n_to_keep, dim=-1)
    print('2', indices)

    ch = indices // time
    t = indices % time


    packed_range = torch.arange(0, n_to_keep, step=1, device=x.device)
    packed_indices = (packed_range[None, :] * time) + t

    context_indices = (packed_range[None, :] * channels) + ch

    sparse = torch.zeros_like(x)
    sparse = torch.scatter(sparse, dim=-1, index=indices, src=values)
    sparse = sparse.view(batch, channels, time)

    context = torch.zeros(batch, n_to_keep * channels, device=x.device)
    context = torch.scatter(context, dim=-1, index=context_indices, src=values)
    context = context.view(batch, n_to_keep, channels)
    
    packed = torch.zeros(batch, n_to_keep * time, device=x.device)
    packed = torch.scatter(packed, dim=-1, index=packed_indices, src=values)
    packed = packed.view(batch, n_to_keep, time)

    return sparse, packed, context


def sparsify3(x: torch.Tensor, n_to_keep: int = 8):
    """
    input: (batch, channels, time)

    outputs:
        sparse:  (batch, channels, time)
        packed:  (batch, n_to_keep, time)
        one_hot: (batch, n_to_keep, channels)
    
        
    - one_hot can be used to generate events
    - packed can be used to convolve generated events with 
      original activations
    """
    batch, channels, time = x.shape
    # orig = x

    x = x.reshape(batch, -1)
    
    # values, indices = torch.topk(x, k=n_to_keep, dim=-1)
    indices = torch.nonzero(x, as_tuple=True)
    print('3', indices)
    
    values = x[indices]

    ch = indices // time
    t = indices % time


    packed_range = torch.arange(0, n_to_keep, step=1, device=x.device)
    packed_indices = (packed_range[None, :] * time) + t

    context_indices = (packed_range[None, :] * channels) + ch

    sparse = torch.zeros_like(x)
    sparse = torch.scatter(sparse, dim=-1, index=indices, src=values)
    sparse = sparse.view(batch, channels, time)

    context = torch.zeros(batch, n_to_keep * channels, device=x.device)
    context = torch.scatter(context, dim=-1, index=context_indices, src=values)
    context = context.view(batch, n_to_keep, channels)
    
    packed = torch.zeros(batch, n_to_keep * time, device=x.device)
    packed = torch.scatter(packed, dim=-1, index=packed_indices, src=values)
    packed = packed.view(batch, n_to_keep, time)

    return sparse, packed, context

if __name__ == '__main__':
    # I need a known, fixed size number of non-zero indices
    
    x = torch.zeros(4, 16, 128)
    x[:, 0, -4:] = 1
    
    
    s1, p1, c1 = sparsify2(x, n_to_keep=4)
    s2, p2, c2 = sparsify3(x)
    
    torch.testing.assert_allclose(s1, s2)
    torch.testing.assert_allclose(p1, p2)
    torch.testing.assert_allclose(c1, c2)


    