import torch
import numpy as np

def pdf(x, mean, sd):
    var = sd ** 2
    denom = (2* np.pi * var) ** .5
    num = torch.exp(-(x-mean) ** 2/ (2*var))
    return num / denom
