import torch
import numpy as np

def pdf(x, mean, sd, epsilon=1e-8):
    var = (sd ** 2) + epsilon
    denom = (2* np.pi * var) ** .5
    num = torch.exp(-(x-mean) ** 2/ (2*var))
    return num / denom
