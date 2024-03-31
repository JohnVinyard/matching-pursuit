import torch
import numpy as np
from torch.distributions import Normal


def pdf(x, mean, sd, epsilon=1e-8):
    var = (sd ** 2) + epsilon
    denom = (2 * np.pi * var) ** .5
    num = torch.exp(-(x-mean) ** 2 / (2*var))
    return num / denom

'''
dist = Normal(
        torch.zeros(16).uniform_(0, 1)[:, None],
        torch.zeros(16).uniform_(0.01, 0.06)[:, None]
    )
    prob = dist.log_prob(torch.linspace(0, 1, 2**15)[None, :])
    print(prob.shape)
    prob = torch.exp(prob)
    prob = prob / prob.max(dim=-1, keepdim=True)[0]
    
    for p in prob:
        plt.plot(p.data.cpu().numpy())
    
    plt.show()
'''

def  pdf2(means, stds, n_elements):
    
    dist = Normal(means[..., None], stds[..., None])
    prob = dist.log_prob(torch.linspace(0, 1, n_elements, device=means.device).view(*([1] * len(stds.shape)), n_elements))
    prob = torch.exp(prob)
    prob = prob / (prob.max(dim=-1, keepdim=True)[0] + 1e-8)
    return prob
                         