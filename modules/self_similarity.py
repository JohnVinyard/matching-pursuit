
'''
def self_similarity(x, ws=None, ss=None, dim=-1):
    batch = x.shape[0]

    x = x.view(-1, n_samples)

    elements = int((ws * (ws - 1)) / 2)
    x = F.pad(x, (0, ss))
    x = x.unfold(dim, ws, ss)
    window = torch.hamming_window(ws).to(x.device)
    x = x * window
    x = x[..., None, :] * x[..., :, None]


    batch_shape = x.shape[:-2]
    sim_shape = x.shape[-2:]
    x = x.reshape(np.prod(batch_shape), *sim_shape)
    row, col = torch.triu_indices(ws, ws, 1)
    x = x[:, row, col]
    a = x = x.reshape(*batch_shape, elements)
    b = x @ x.permute(0, 2, 1)

    row, col = torch.triu_indices(b.shape[-1], b.shape[-1], 1)
    b = b[:, row, col]
    
    x = torch.cat([
        a.view(batch, -1), 
        # b.view(batch, -1)
    ], dim=1)
    return x

'''

import torch
from torch import nn
from torch.nn import functional as F

from modules.linear import LinearOutputStack

def self_sim(x, reduction=None, return_full=False):
    if reduction is not None:
        x = x[..., None]
    
    mat = x @ x.transpose(-1, -2)
    if return_full:
        return mat
    orig_shape = mat.shape
    size = mat.shape[-1]

    if reduction is not None:
        return reduction(mat, dim=-1)
    
    mat = mat.view(-1, size, size)
    row, col = torch.triu_indices(size, size, 1)
    mat = mat[..., row, col]
    mat = mat.view(*orig_shape[:-2], -1)
    return mat



class SelfSimNetwork(nn.Module):
    def __init__(self, channels, layers, window=512, step=256, input_size=2**14):
        super().__init__()
        n_frames = input_size // step
        # how many elements in the upper triangular matrix
        self.size = int((n_frames * (n_frames - 1)) / 2)
        self.net = LinearOutputStack(channels, layers, in_channels=self.size)
        self.window = window
        self.step = step

    def forward(self, x, return_full=False):
        # assume audio in the shape (batch, 1, 16384)
        x = F.pad(x, (0, self.step))
        x = x.unfold(-1, self.window, self.step)
        x = x * torch.hamming_window(self.window).to(x.device)
        x = y = self_sim(x, reduction=torch.mean)
        x = self_sim(x, return_full=return_full)
        if return_full:
            return x
        x = x.view(-1, self.size)
        x = self.net(x)
        return x, y



    